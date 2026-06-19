#include <fstream>
#include <gltfio/materials/uberarchive.h>
#include "core/scene_bridge.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <utils/EntityManager.h>
#include <math/mat4.h>
#include <math/vec3.h>
#include <math/quat.h>

namespace vf_mujoco {

static constexpr int SPHERE_SLICES = 32;
static constexpr int SPHERE_STACKS = 16;
static constexpr int CAPSULE_SLICES = 24;
static constexpr int CAPSULE_STACKS = 8;
static constexpr int CYLINDER_SLICES = 32;
static constexpr float GROUND_PLANE_SIZE = 20.0f;
static constexpr int GROUND_PLANE_SUBDIVS = 64;

SceneBridge::SceneBridge(Renderer& renderer)
    : renderer_(renderer)
{
    material_manager_ = std::make_unique<MaterialManager>(renderer.engine());
    light_manager_ = std::make_unique<LightManager>(renderer.engine(), renderer.scene());
}

SceneBridge::~SceneBridge() {
    clear();
}

void SceneBridge::load_model(const mjModel* model) {
    clear();

    // Initialize default PBR materials
    material_manager_->initialize();

    // Lighting: if the MJCF/XML model declares its own <light> elements, honour
    // them (so a native MuJoCo scene's lights actually drive the render); else
    // fall back to a generic sun + fill. Either way keep the IBL/ambient probe
    // so PBR materials have indirect light + reflections.
    int n = sync_lights_from_model(model);
    if (n == 0) {
        light_manager_->setup_default_lighting();
    } else {
        light_manager_->setup_indirect_fallback();
    }

    // If the MJCF/XML defines its own skybox (e.g. MuJoCo's gradient sky), use
    // it as the background so the scene matches MuJoCo's look (and the floor
    // reflects that sky instead of a bright neutral one).
    for (int t = 0; t < model->ntex; ++t) {
        if (model->tex_type[t] == mjTEXTURE_SKYBOX) {
            int tw = model->tex_width[t];
            int th = model->tex_height[t];
            int nch = model->tex_nchannel[t];
            const uint8_t* tdata = model->tex_data + model->tex_adr[t];
            // Use a distinct cache key range so it doesn't collide with material
            // textures (which key on the texture id directly).
            filament::Texture* sky = material_manager_->get_or_create_texture_cube(
                100000 + t, tw, th, nch, tdata);
            if (sky) light_manager_->set_skybox_cubemap(sky);
            break;
        }
    }

    // Create Filament renderables for each MuJoCo geom
    for (int i = 0; i < model->ngeom; ++i) {
        create_geom(model, i);
    }
}

int SceneBridge::sync_lights_from_model(const mjModel* model) {
    const int nlight = model->nlight;
    if (nlight <= 0) return 0;

    int created = 0;
    for (int i = 0; i < nlight; ++i) {
        if (model->light_active && !model->light_active[i]) continue;

        // Global pose at qpos0 (correct for the common worldbody lights; static).
        const mjtNum* p = model->light_pos0 + i * 3;
        const mjtNum* d = model->light_dir0 + i * 3;
        const float* col = model->light_diffuse + i * 3;
        float r = col[0], g = col[1], b = col[2];
        // Skip black lights.
        if (r + g + b <= 1e-4f) continue;

        int type = model->light_type[i];
        bool shadow = model->light_castshadow && model->light_castshadow[i];

        // MuJoCo diffuse is a 0..1 OpenGL-style colour. Map its magnitude onto
        // Filament's physical scale (lux for directional, lumens for punctual).
        // These constants suit normal MJCF scenes (tabletop to room scale); the
        // warehouse drives its own much-brighter lights through the Python API,
        // so it is unaffected by these values.
        float mag = std::max({r, g, b, 1e-3f});
        // Normalize colour so intensity carries the brightness.
        float nr = r / mag, ng = g / mag, nb = b / mag;

        switch (type) {
            case mjLIGHT_DIRECTIONAL: {
                float dx = (float)d[0], dy = (float)d[1], dz = (float)d[2];
                float dl = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dl > 1e-6f) { dx/=dl; dy/=dl; dz/=dl; } else { dz = -1.0f; }
                // A perfectly vertical key light casts its shadow straight down,
                // hidden directly under the object. MuJoCo/Isaac-style scenes
                // read better with the shadow spread to the side, so nudge a
                // near-vertical light slightly off-axis (keeps the lighting look
                // essentially the same but grounds objects with a visible shadow).
                if (std::abs(dx) < 0.15f && std::abs(dy) < 0.15f) {
                    dx += 0.32f; dy += 0.22f;
                    float nl = std::sqrt(dx*dx + dy*dy + dz*dz);
                    dx/=nl; dy/=nl; dz/=nl;
                }
                light_manager_->add_directional_light(
                    dx, dy, dz, nr, ng, nb, 13000.0f * mag, shadow);
                ++created;
                break;
            }
            case mjLIGHT_POINT: {
                light_manager_->add_point_light(
                    (float)p[0], (float)p[1], (float)p[2],
                    nr, ng, nb, 60000.0f * mag, 8.0f, false);
                ++created;
                break;
            }
            case mjLIGHT_SPOT: {
                float dx = (float)d[0], dy = (float)d[1], dz = (float)d[2];
                float dl = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dl > 1e-6f) { dx/=dl; dy/=dl; dz/=dl; } else { dz = -1.0f; }
                // MuJoCo cutoff is the spot half-angle in degrees; clamp to
                // Filament's <=90 requirement and give a soft inner cone.
                float outer = 45.0f;
                if (model->light_cutoff) outer = model->light_cutoff[i];
                if (outer <= 1.0f || outer > 90.0f) outer = std::min(outer, 88.0f);
                outer = std::min(std::max(outer, 5.0f), 88.0f);
                float inner = outer * 0.6f;
                light_manager_->add_spot_light(
                    (float)p[0], (float)p[1], (float)p[2], dx, dy, dz,
                    nr, ng, nb, 150000.0f * mag, 10.0f, inner, outer,
                    shadow, false);
                ++created;
                break;
            }
            default:
                break;  // image-based handled via IBL elsewhere
        }
    }
    return created;
}

void SceneBridge::create_geom(const mjModel* model, int geom_id) {
    int geom_type = model->geom_type[geom_id];
    const mjtNum* geom_size = model->geom_size + geom_id * 3;
    const float* geom_rgba = model->geom_rgba + geom_id * 4;

    // Skip invisible geoms (collision-only group, or fully transparent).
    int geom_group = model->geom_group[geom_id];
    if (geom_group >= 3) return;        // collision-only
    if (geom_rgba[3] < 0.01f) return;   // fully transparent

    ResolvedMaterial rm = resolve_material(model, geom_id);

    // Render MuJoCo geoms directly so a standard MJCF (primitives, mesh assets,
    // height fields) renders with full material/texture coverage — no GLB needed.
    switch (geom_type) {
        case mjGEOM_PLANE:
        case mjGEOM_SPHERE:
        case mjGEOM_BOX:
        case mjGEOM_CAPSULE:
        case mjGEOM_CYLINDER:
        case mjGEOM_ELLIPSOID:
            create_primitive(geom_id, geom_type, geom_size, rm);
            break;
        case mjGEOM_MESH: {
            int mesh_id = model->geom_dataid[geom_id];
            if (mesh_id >= 0) create_mesh(model, geom_id, mesh_id, rm);
            break;
        }
        case mjGEOM_HFIELD: {
            int hfield_id = model->geom_dataid[geom_id];
            if (hfield_id >= 0) create_hfield(model, geom_id, hfield_id, geom_size, rm);
            break;
        }
        default:
            // sdf etc. are not yet supported natively.
            break;
    }
}

ResolvedMaterial SceneBridge::resolve_material(const mjModel* model, int geom_id) {
    ResolvedMaterial rm;
    const float* geom_rgba = model->geom_rgba + geom_id * 4;
    int geom_type = model->geom_type[geom_id];
    for (int k = 0; k < 4; ++k) rm.rgba[k] = geom_rgba[k];

    int matid = model->geom_matid[geom_id];
    if (matid >= 0) {
        rm.has_mat = true;
        // MuJoCo leaves metallic/roughness at -1 ("unset") unless authored for
        // PBR. Most MJCFs (e.g. MuJoCo Menagerie robots) ship NO PBR data — just
        // a flat colour + legacy shininess — which renders as dull plastic. To
        // get an Isaac-style finish out of the box we apply smart PBR defaults
        // based on the base colour when the author left PBR unset. If the author
        // DID set metallic/roughness, we always respect it.
        float mj_metallic = model->mat_metallic[matid];
        float mj_roughness = model->mat_roughness[matid];
        const float* mc = model->mat_rgba + matid * 4;

        // Decide the surface colour first (geom rgba overrides the material's
        // unless the geom is left at the default grey).
        bool geom_default = (geom_rgba[0] == 0.5f && geom_rgba[1] == 0.5f &&
                             geom_rgba[2] == 0.5f && geom_rgba[3] == 1.0f);
        if (geom_default) for (int k = 0; k < 4; ++k) rm.rgba[k] = mc[k];

        const bool pbr_authored = (mj_metallic >= 0.0f) || (mj_roughness >= 0.0f);

        if (pbr_authored) {
            rm.metallic  = (mj_metallic  >= 0.0f) ? mj_metallic  : 0.0f;
            rm.roughness = (mj_roughness >= 0.0f) ? mj_roughness
                          : std::clamp(1.0f - model->mat_shininess[matid], 0.04f, 1.0f);
            rm.reflectance = model->mat_specular[matid];
        } else {
            // --- Smart defaults for plain (non-PBR) MJCF materials ----------
            float r = rm.rgba[0], g = rm.rgba[1], b = rm.rgba[2];
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            float mx = std::max({r, g, b}), mn = std::min({r, g, b});
            float chroma = mx - mn;          // ~0 = grey/black/white
            float shininess = model->mat_shininess[matid];  // legacy hint, def 0.5

            // Default: smooth semi-gloss dielectric (clean injection-moulded
            // plastic) — the look most robot parts actually have.
            rm.metallic = 0.0f;
            rm.roughness = 0.42f;
            rm.reflectance = 0.6f;

            if (lum > 0.75f && chroma < 0.10f) {
                // Bright neutral (white robot shells): smooth satin plastic.
                rm.roughness = 0.38f;
                rm.reflectance = 0.65f;
            } else if (lum < 0.28f && chroma < 0.12f) {
                // Dark neutral (joint covers, rubber, trim — MuJoCo "black" is
                // often ~0.2 grey): glossy black plastic. Deepen the base colour
                // so the bright studio IBL doesn't lift it to a washed grey; the
                // gloss highlight is what reads it as a surface, not the albedo.
                float deep = 0.45f;  // pull toward black
                rm.rgba[0] *= deep; rm.rgba[1] *= deep; rm.rgba[2] *= deep;
                rm.roughness = 0.28f;
                rm.reflectance = 0.6f;
            } else if (chroma < 0.08f && lum >= 0.35f && lum <= 0.75f) {
                // Mid grey: likely brushed/painted metal -> a touch metallic.
                rm.metallic = 0.6f;
                rm.roughness = 0.40f;
                rm.reflectance = 0.7f;
            } else {
                // Saturated colour (accents, coloured plastic): glossy plastic.
                rm.roughness = 0.40f;
                rm.reflectance = 0.6f;
            }
            // Nudge by the legacy shininess hint if the author set a non-default.
            if (shininess > 0.0f && shininess != 0.5f) {
                rm.roughness = std::clamp(rm.roughness * (1.3f - shininess), 0.06f, 1.0f);
            }
        }
        rm.emissive = model->mat_emission[matid];

        // NOTE on MuJoCo `mat_reflectance`: MuJoCo implements it as a PLANAR
        // mirror (a second mirrored scene pass), not an environment reflection.
        // Driving Filament roughness down here only makes the floor reflect the
        // IBL/sky (a bright sheen) — NOT the scene geometry — which looks like a
        // washed-out hotspot, not a mirror. So we keep the surface MATTE; a true
        // planar reflection pass is the correct way to mirror the scene.

        rm.uvscale[0] = model->mat_texrepeat[matid * 2 + 0];
        rm.uvscale[1] = model->mat_texrepeat[matid * 2 + 1];

        // Resolve a base-color texture (RGB or RGBA role).
        int texid = model->mat_texid[matid * mjNTEXROLE + mjTEXROLE_RGB];
        if (texid < 0) texid = model->mat_texid[matid * mjNTEXROLE + mjTEXROLE_RGBA];
        if (texid >= 0 && texid < model->ntex) {
            int tw = model->tex_width[texid];
            int th = model->tex_height[texid];
            int nch = model->tex_nchannel[texid];
            const uint8_t* tdata = model->tex_data + model->tex_adr[texid];
            int ttype = model->tex_type[texid];
            if (ttype == mjTEXTURE_2D) {
                rm.albedo_2d = material_manager_->get_or_create_texture_2d(
                    texid, tw, th, nch, tdata);
            } else { // CUBE or SKYBOX used as a geom texture
                rm.cube = material_manager_->get_or_create_texture_cube(
                    texid, tw, th, nch, tdata);
            }
        }
    } else {
        // No MuJoCo material: keep a tasteful brightness heuristic for scalars so
        // bare MJCF colors still look like plausible materials.
        float brightness = (rm.rgba[0] + rm.rgba[1] + rm.rgba[2]) / 3.0f;
        rm.roughness = 0.5f; rm.metallic = 0.0f; rm.reflectance = 0.2f;
        if (brightness < 0.25f) { rm.roughness = 0.45f; rm.metallic = 0.4f; rm.reflectance = 0.35f; }
        else if (brightness > 0.8f) { rm.roughness = 0.4f; rm.reflectance = 0.25f; }
        else if (rm.rgba[0] > 0.7f && rm.rgba[1] < 0.3f) { rm.roughness = 0.4f; rm.reflectance = 0.25f; }
    }

    // Floor planes read best MATTE: a glossy floor reflects the bright IBL/sky
    // and washes out (we can't planar-mirror the scene here anyway). Force any
    // ground plane fully diffuse regardless of material so it stays clean.
    if (geom_type == mjGEOM_PLANE) {
        rm.roughness = 1.0f;
        rm.metallic = 0.0f;
        rm.reflectance = 0.0f;
    }
    return rm;
}

filament::MaterialInstance* SceneBridge::make_instance(const ResolvedMaterial& rm) {
    if (rm.textured()) {
        return material_manager_->create_mujoco_textured_instance(
            rm.rgba[0], rm.rgba[1], rm.rgba[2], rm.rgba[3],
            rm.roughness, rm.metallic, rm.reflectance, rm.emissive,
            rm.uvscale[0], rm.uvscale[1], rm.albedo_2d, rm.cube);
    }
    return material_manager_->create_pbr_instance(
        rm.rgba[0], rm.rgba[1], rm.rgba[2], rm.rgba[3],
        rm.roughness, rm.metallic, rm.reflectance, rm.emissive);
}

void SceneBridge::create_primitive(int geom_id, int geom_type,
                                    const mjtNum* size, const ResolvedMaterial& rm) {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;

    switch (geom_type) {
        case mjGEOM_SPHERE:
            build_sphere(size[0], SPHERE_SLICES, SPHERE_STACKS, vertices, indices);
            break;
        case mjGEOM_ELLIPSOID:
            build_ellipsoid(size[0], size[1], size[2], SPHERE_SLICES,
                            SPHERE_STACKS, vertices, indices);
            break;
        case mjGEOM_BOX:
            build_box(size[0], size[1], size[2], vertices, indices);
            break;
        case mjGEOM_CAPSULE:
            build_capsule(size[0], size[1], CAPSULE_SLICES, CAPSULE_STACKS,
                         vertices, indices);
            break;
        case mjGEOM_CYLINDER:
            build_cylinder(size[0], size[1], CYLINDER_SLICES, vertices, indices);
            break;
        case mjGEOM_PLANE:
            build_plane(GROUND_PLANE_SIZE, vertices, indices);
            break;
        default:
            return;
    }

    // UVs are only needed for the textured material. The plane gets WORLD-space
    // planar UVs so a checker/texture tiles uniformly regardless of the (large)
    // plane size — matching MuJoCo's `texuniform` floor. Other geoms use the
    // material's object-space cube sampling, so a zero UV buffer just satisfies
    // the uv0 requirement.
    ResolvedMaterial rm_local = rm;
    std::vector<float> uvs;
    if (rm.textured()) {
        const uint32_t vcount = static_cast<uint32_t>(vertices.size() / 6);
        uvs.resize(static_cast<size_t>(vcount) * 2, 0.0f);
        if (geom_type == mjGEOM_PLANE) {
            // Bake a world-space tile density directly into the UVs (~0.67
            // texture repeats per metre -> roughly MuJoCo's ~0.75 m checker
            // tiles). mat_texrepeat tunes it around that baseline. The shader's
            // uvscale is neutralised to 1 so density isn't applied twice.
            const float density = 0.13f * (rm.uvscale[0] > 0.0f ? rm.uvscale[0] : 1.0f);
            const float density_y = 0.13f * (rm.uvscale[1] > 0.0f ? rm.uvscale[1] : 1.0f);
            for (uint32_t i = 0; i < vcount; ++i) {
                float x = vertices[i * 6 + 0], y = vertices[i * 6 + 1];
                uvs[i * 2 + 0] = x * density;
                uvs[i * 2 + 1] = y * density_y;
            }
            rm_local.uvscale[0] = 1.0f;
            rm_local.uvscale[1] = 1.0f;
        }
    }

    auto* mat_inst = make_instance(rm_local);
    // The ground plane only RECEIVES shadows; it shouldn't cast (a flat plane
    // self-shadows into acne). All other primitives cast.
    bool cast = (geom_type != mjGEOM_PLANE);
    auto renderable = create_renderable(vertices, indices, mat_inst, geom_id, uvs, cast);
    renderable.mj_geom_type = geom_type;
    geom_renderables_.push_back(std::move(renderable));
}

void SceneBridge::create_mesh(const mjModel* model, int geom_id, int mesh_id,
                              const ResolvedMaterial& rm) {
    // MuJoCo stores mesh data in shared arrays, addressed per mesh. Crucially,
    // vertices and normals have INDEPENDENT counts/indices: faces index vertices
    // via mesh_face and normals via mesh_facenormal (which may differ). So we
    // expand to a non-indexed (per-face-corner) buffer to get correct normals
    // regardless of how MuJoCo deduplicated them.
    int vert_start = model->mesh_vertadr[mesh_id];
    int norm_start = model->mesh_normaladr[mesh_id];
    int face_start = model->mesh_faceadr[mesh_id];
    int face_count = model->mesh_facenum[mesh_id];

    if (face_count == 0) return;

    const float* mj_verts = model->mesh_vert;            // (nmeshvert x 3)
    const float* mj_normals = model->mesh_normal;        // (nmeshnormal x 3)
    const int* mj_faces = model->mesh_face;              // (nmeshface x 3) vert idx
    const int* mj_facenorm = model->mesh_facenormal;     // (nmeshface x 3) norm idx
    const bool have_normals = (model->mesh_normalnum[mesh_id] > 0);

    // Texture coordinates (optional): faces index texcoords via mesh_facetexcoord.
    int tc_start = model->mesh_texcoordadr[mesh_id];
    const bool have_uv = (tc_start >= 0 && rm.textured());
    const float* mj_tc = model->mesh_texcoord;           // (nmeshtexcoord x 2)
    const int* mj_facetc = model->mesh_facetexcoord;     // (nmeshface x 3) tc idx

    // Interleaved position (3) + normal (3) = 6 floats per corner, 3 corners/face.
    std::vector<float> vertices;
    vertices.reserve(static_cast<size_t>(face_count) * 3 * 6);
    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(face_count) * 3);
    std::vector<float> uvs;
    if (have_uv) uvs.reserve(static_cast<size_t>(face_count) * 3 * 2);

    uint32_t out_idx = 0;
    for (int f = 0; f < face_count; ++f) {
        const int* fv = mj_faces + (face_start + f) * 3;
        const int* fn = mj_facenorm + (face_start + f) * 3;
        const int* ft = have_uv ? mj_facetc + (face_start + f) * 3 : nullptr;

        // Compute a geometric face normal as a fallback (and for meshes without
        // stored normals). Vertices are local indices within this mesh.
        const float* p0 = mj_verts + (vert_start + fv[0]) * 3;
        const float* p1 = mj_verts + (vert_start + fv[1]) * 3;
        const float* p2 = mj_verts + (vert_start + fv[2]) * 3;
        float e1x = p1[0]-p0[0], e1y = p1[1]-p0[1], e1z = p1[2]-p0[2];
        float e2x = p2[0]-p0[0], e2y = p2[1]-p0[1], e2z = p2[2]-p0[2];
        float fnx = e1y*e2z - e1z*e2y;
        float fny = e1z*e2x - e1x*e2z;
        float fnz = e1x*e2y - e1y*e2x;
        float flen = std::sqrt(fnx*fnx + fny*fny + fnz*fnz);
        if (flen > 1e-12f) { fnx/=flen; fny/=flen; fnz/=flen; }
        else { fnx=0; fny=0; fnz=1; }

        for (int k = 0; k < 3; ++k) {
            const float* p = mj_verts + (vert_start + fv[k]) * 3;
            vertices.push_back(p[0]);
            vertices.push_back(p[1]);
            vertices.push_back(p[2]);
            if (have_normals) {
                const float* nrm = mj_normals + (norm_start + fn[k]) * 3;
                vertices.push_back(nrm[0]);
                vertices.push_back(nrm[1]);
                vertices.push_back(nrm[2]);
            } else {
                vertices.push_back(fnx);
                vertices.push_back(fny);
                vertices.push_back(fnz);
            }
            if (have_uv) {
                const float* uv = mj_tc + (tc_start + ft[k]) * 2;
                // MuJoCo texcoords use a top-left origin; flip V for Filament.
                uvs.push_back(uv[0]);
                uvs.push_back(1.0f - uv[1]);
            }
            indices.push_back(out_idx++);
        }
    }

    auto* mat_inst = make_instance(rm);
    auto renderable = create_renderable(vertices, indices, mat_inst, geom_id, uvs);
    renderable.mj_geom_type = mjGEOM_MESH;
    geom_renderables_.push_back(std::move(renderable));
}

void SceneBridge::create_hfield(const mjModel* model, int geom_id, int hfield_id,
                                const mjtNum* /*size*/, const ResolvedMaterial& rm) {
    // MuJoCo height field: an nrow x ncol grid of elevations in [0,1], scaled to
    // the geom's local box (radius_x, radius_y, elevation_z, base_z). Build a
    // triangulated surface mesh (pos+normal) plus optional planar UVs.
    int nrow = model->hfield_nrow[hfield_id];
    int ncol = model->hfield_ncol[hfield_id];
    if (nrow < 2 || ncol < 2) return;
    const mjtNum* hsize = model->hfield_size + hfield_id * 4;  // x, y, z_top, z_base
    const float rx = static_cast<float>(hsize[0]);
    const float ry = static_cast<float>(hsize[1]);
    const float rz = static_cast<float>(hsize[2]);
    const float* hdata = model->hfield_data + model->hfield_adr[hfield_id];

    auto elev = [&](int r, int c) -> float {
        return hdata[r * ncol + c] * rz;  // [0,1] -> world height
    };
    auto px = [&](int c) -> float { return -rx + 2.0f * rx * c / (ncol - 1); };
    auto py = [&](int r) -> float { return -ry + 2.0f * ry * r / (nrow - 1); };

    std::vector<float> vertices;
    vertices.reserve(static_cast<size_t>(nrow) * ncol * 6);
    std::vector<float> uvs;
    const bool tex = rm.textured();
    if (tex) uvs.reserve(static_cast<size_t>(nrow) * ncol * 2);

    for (int r = 0; r < nrow; ++r) {
        for (int c = 0; c < ncol; ++c) {
            float x = px(c), y = py(r), z = elev(r, c);
            // Central-difference normal from neighbours.
            int cm = c > 0 ? c - 1 : c, cp = c < ncol - 1 ? c + 1 : c;
            int rm_ = r > 0 ? r - 1 : r, rp = r < nrow - 1 ? r + 1 : r;
            float dzdx = (elev(r, cp) - elev(r, cm)) / (px(cp) - px(cm) + 1e-9f);
            float dzdy = (elev(rp, c) - elev(rm_, c)) / (py(rp) - py(rm_) + 1e-9f);
            float nx = -dzdx, ny = -dzdy, nz = 1.0f;
            float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            nx/=nl; ny/=nl; nz/=nl;
            vertices.push_back(x); vertices.push_back(y); vertices.push_back(z);
            vertices.push_back(nx); vertices.push_back(ny); vertices.push_back(nz);
            if (tex) {
                uvs.push_back(static_cast<float>(c) / (ncol - 1));
                uvs.push_back(static_cast<float>(r) / (nrow - 1));
            }
        }
    }

    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(nrow - 1) * (ncol - 1) * 6);
    auto idx = [&](int r, int c) { return static_cast<uint32_t>(r * ncol + c); };
    for (int r = 0; r < nrow - 1; ++r) {
        for (int c = 0; c < ncol - 1; ++c) {
            indices.push_back(idx(r, c));   indices.push_back(idx(r, c + 1));   indices.push_back(idx(r + 1, c));
            indices.push_back(idx(r + 1, c)); indices.push_back(idx(r, c + 1)); indices.push_back(idx(r + 1, c + 1));
        }
    }

    auto* mat_inst = make_instance(rm);
    auto renderable = create_renderable(vertices, indices, mat_inst, geom_id, uvs);
    renderable.mj_geom_type = mjGEOM_HFIELD;
    geom_renderables_.push_back(std::move(renderable));
}

GeomRenderable SceneBridge::create_renderable(
    const std::vector<float>& vertices,
    const std::vector<uint32_t>& indices,
    filament::MaterialInstance* mat_inst,
    int geom_id,
    const std::vector<float>& uvs,
    bool cast_shadows)
{
    auto* engine = renderer_.engine();

    uint32_t vertex_count = static_cast<uint32_t>(vertices.size() / 6);
    const bool has_uv = !uvs.empty();
    uint32_t index_count = static_cast<uint32_t>(indices.size());

    // Vertex buffer: position (buffer 0) + tangents as SHORT4 (buffer 1)
    // + optional UV0 (buffer 2). When no UVs are needed the fast 2-buffer
    // position+tangent path is used (identical to the untextured fast path).
    const uint8_t buffer_count = has_uv ? 3 : 2;
    auto vb_builder = filament::VertexBuffer::Builder()
        .vertexCount(vertex_count)
        .bufferCount(buffer_count)
        .attribute(filament::VertexAttribute::POSITION, 0,
                   filament::VertexBuffer::AttributeType::FLOAT3, 0, 24)
        .attribute(filament::VertexAttribute::TANGENTS, 1,
                   filament::VertexBuffer::AttributeType::SHORT4, 0, 8)
        .normalized(filament::VertexAttribute::TANGENTS);
    if (has_uv) {
        vb_builder.attribute(filament::VertexAttribute::UV0, 2,
                             filament::VertexBuffer::AttributeType::FLOAT2, 0, 8);
    }
    auto* vb = vb_builder.build(*engine);

    // Upload position data (buffer 0)
    auto vert_data = new float[vertices.size()];
    std::memcpy(vert_data, vertices.data(), vertices.size() * sizeof(float));
    vb->setBufferAt(*engine, 0,
        filament::VertexBuffer::BufferDescriptor(
            vert_data, vertices.size() * sizeof(float),
            [](void* buf, size_t, void*) { delete[] static_cast<float*>(buf); }
        ));

    // Compute tangent quaternions from normals (buffer 1)
    auto tangent_data = new int16_t[vertex_count * 4];
    for (uint32_t i = 0; i < vertex_count; i++) {
        float nx = vertices[i*6+3], ny = vertices[i*6+4], nz = vertices[i*6+5];
        // Normalize
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 0.0001f) { nx/=len; ny/=len; nz/=len; }
        else { nx=0; ny=0; nz=1; }
        // Compute tangent frame from normal using Frisvad's method
        float tx, ty, tz, bx, by, bz;
        if (nz < -0.9999f) {
            tx = 0; ty = -1; tz = 0;
            bx = -1; by = 0; bz = 0;
        } else {
            float a = 1.0f / (1.0f + nz);
            float b = -nx * ny * a;
            tx = 1.0f - nx * nx * a; ty = b; tz = -nx;
            bx = b; by = 1.0f - ny * ny * a; bz = -ny;
        }
        // Build rotation matrix -> quaternion
        // mat = [tx,bx,nx; ty,by,ny; tz,bz,nz]
        float trace = tx + by + nz;
        float qw, qx, qy, qz;
        if (trace > 0) {
            float s = 0.5f / std::sqrt(trace + 1.0f);
            qw = 0.25f / s;
            qx = (bz - ny) * s;
            qy = (nx - tz) * s; 
            qz = (ty - bx) * s;
        } else if (tx > by && tx > nz) {
            float s = 2.0f * std::sqrt(1.0f + tx - by - nz);
            qw = (bz - ny) / s;
            qx = 0.25f * s;
            qy = (ty + bx) / s;
            qz = (nx + tz) / s;
        } else if (by > nz) {
            float s = 2.0f * std::sqrt(1.0f + by - tx - nz);
            qw = (nx - tz) / s;
            qx = (ty + bx) / s;
            qy = 0.25f * s;
            qz = (bz + ny) / s;
        } else {
            float s = 2.0f * std::sqrt(1.0f + nz - tx - by);
            qw = (ty - bx) / s;
            qx = (nx + tz) / s;
            qy = (bz + ny) / s;
            qz = 0.25f * s;
        }
        // Normalize quaternion
        float qlen = std::sqrt(qw*qw+qx*qx+qy*qy+qz*qz);
        if (qlen > 0) { qw/=qlen; qx/=qlen; qy/=qlen; qz/=qlen; }
        // Pack as SHORT4 (int16 normalized)
        tangent_data[i*4+0] = (int16_t)(qx * 32767.0f);
        tangent_data[i*4+1] = (int16_t)(qy * 32767.0f);
        tangent_data[i*4+2] = (int16_t)(qz * 32767.0f);
        tangent_data[i*4+3] = (int16_t)(qw * 32767.0f);
    }
    vb->setBufferAt(*engine, 1,
        filament::VertexBuffer::BufferDescriptor(
            tangent_data, vertex_count * 4 * sizeof(int16_t),
            [](void* buf, size_t, void*) { delete[] static_cast<int16_t*>(buf); }
        ));

    // UV0 data (buffer 2), only when a textured material needs it.
    if (has_uv) {
        auto uv_data = new float[vertex_count * 2];
        // uvs may be shorter than expected only if mismatched; guard the copy.
        size_t n = std::min(uvs.size(), static_cast<size_t>(vertex_count) * 2);
        std::memcpy(uv_data, uvs.data(), n * sizeof(float));
        for (size_t i = n; i < static_cast<size_t>(vertex_count) * 2; ++i) uv_data[i] = 0.0f;
        vb->setBufferAt(*engine, 2,
            filament::VertexBuffer::BufferDescriptor(
                uv_data, vertex_count * 2 * sizeof(float),
                [](void* buf, size_t, void*) { delete[] static_cast<float*>(buf); }
            ));
    }

    // Index buffer
    auto* ib = filament::IndexBuffer::Builder()
        .indexCount(index_count)
        .bufferType(filament::IndexBuffer::IndexType::UINT)
        .build(*engine);

    auto idx_data = new uint32_t[indices.size()];
    std::memcpy(idx_data, indices.data(), indices.size() * sizeof(uint32_t));
    ib->setBuffer(*engine,
        filament::IndexBuffer::BufferDescriptor(
            idx_data, indices.size() * sizeof(uint32_t),
            [](void* buf, size_t, void*) { delete[] static_cast<uint32_t*>(buf); }
        ));

    // Create entity and renderable component
    auto entity = utils::EntityManager::get().create();

    // Set explicit bounding box to avoid empty AABB crash
    filament::Box aabb;
    // Compute from vertices
    float minx=1e10,miny=1e10,minz=1e10,maxx=-1e10,maxy=-1e10,maxz=-1e10;
    for (uint32_t i = 0; i < vertex_count; i++) {
        float x = vertices[i*6+0], y = vertices[i*6+1], z = vertices[i*6+2];
        if(x<minx)minx=x; if(y<miny)miny=y; if(z<minz)minz=z;
        if(x>maxx)maxx=x; if(y>maxy)maxy=y; if(z>maxz)maxz=z;
    }
    aabb.center = filament::math::float3{(minx+maxx)/2,(miny+maxy)/2,(minz+maxz)/2};
    aabb.halfExtent = filament::math::float3{(maxx-minx)/2+0.001f,(maxy-miny)/2+0.001f,(maxz-minz)/2+0.001f};

    filament::RenderableManager::Builder(1)
        .geometry(0, filament::RenderableManager::PrimitiveType::TRIANGLES, vb, ib)
        .material(0, mat_inst)
        .boundingBox(aabb)
        .receiveShadows(true)
        .castShadows(cast_shadows)
        .culling(false)
        .build(*engine, entity);

    renderer_.scene()->addEntity(entity);

    GeomRenderable gr;
    gr.entity = entity;
    gr.vertex_buffer = vb;
    gr.index_buffer = ib;
    gr.material_instance = mat_inst;
    gr.mj_geom_id = geom_id;
    return gr;
}

void SceneBridge::sync_transforms(const mjModel* model, const mjData* data) {
    auto& tcm = renderer_.engine()->getTransformManager();

    for (auto& gr : geom_renderables_) {
        if (gr.mj_geom_id < 0) continue;

        const double* pos = data->geom_xpos + gr.mj_geom_id * 3;
        // MuJoCo stores rotation as 3x3 matrix (row-major)
        const double* mat = data->geom_xmat + gr.mj_geom_id * 9;

        // Build Filament column-major 4x4 transform
        filament::math::mat4f transform(
            filament::math::float4{(float)mat[0], (float)mat[3], (float)mat[6], 0.0f},
            filament::math::float4{(float)mat[1], (float)mat[4], (float)mat[7], 0.0f},
            filament::math::float4{(float)mat[2], (float)mat[5], (float)mat[8], 0.0f},
            filament::math::float4{(float)pos[0], (float)pos[1], (float)pos[2], 1.0f}
        );

        auto inst = tcm.getInstance(gr.entity);
        tcm.setTransform(inst, transform);
    }
}

void SceneBridge::sync_camera(const mjModel* model, const mjData* data, int cam_id) {
    auto* camera = renderer_.camera();

    if (cam_id < 0 || cam_id >= model->ncam) {
        // Use default free camera
        return;
    }

    const double* pos = data->cam_xpos + cam_id * 3;
    const double* mat = data->cam_xmat + cam_id * 9;

    // MuJoCo camera looks along -Z in camera frame
    filament::math::float3 eye{(float)pos[0], (float)pos[1], (float)pos[2]};

    // Forward direction is -Z column of rotation matrix
    filament::math::float3 forward{
        -(float)mat[2], -(float)mat[5], -(float)mat[8]
    };
    filament::math::float3 up{
        (float)mat[1], (float)mat[4], (float)mat[7]
    };

    filament::math::float3 target = eye + forward;

    camera->lookAt(eye, target, up);

    // Set FOV from MuJoCo camera
    float fovy = static_cast<float>(model->cam_fovy[cam_id]);
    float aspect = static_cast<float>(renderer_.width()) /
                   static_cast<float>(renderer_.height());
    camera->setProjection(fovy, aspect, 0.05f, 200.0f);
}

void SceneBridge::render_batch_rgb(const mjModel* model,
                                   const std::vector<const mjData*>& datas,
                                   int cam_id, uint8_t* out_rgb) {
    const size_t n = datas.size();
    if (n == 0 || !out_rgb) return;

    const uint32_t w = renderer_.width();
    const uint32_t h = renderer_.height();
    const size_t rgba_per = static_cast<size_t>(w) * h * 4;
    if (batch_scratch_.size() < n * rgba_per) batch_scratch_.resize(n * rgba_per);

    // For each env: sync its state, render, and QUEUE an async readback into its
    // own slice of the scratch buffer — no per-env GPU wait.
    for (size_t i = 0; i < n; ++i) {
        sync_transforms(model, datas[i]);
        if (cam_id >= 0) sync_camera(model, datas[i], cam_id);
        renderer_.render_readback_async(batch_scratch_.data() + i * rgba_per);
    }

    // Single GPU sync for the whole batch.
    renderer_.finish();

    // Strip RGBA -> RGB for all envs.
    const size_t px = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* src = batch_scratch_.data() + i * rgba_per;
        uint8_t* dst = out_rgb + i * px * 3;
        for (size_t p = 0; p < px; ++p) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst += 3;
            src += 4;
        }
    }
}

void SceneBridge::set_free_camera(float eye_x, float eye_y, float eye_z,
                                   float target_x, float target_y, float target_z) {
    renderer_.camera()->lookAt(
        filament::math::float3{eye_x, eye_y, eye_z},
        filament::math::float3{target_x, target_y, target_z},
        filament::math::float3{0.0f, 0.0f, 1.0f}
    );
}

void SceneBridge::clear() {
    auto* engine = renderer_.engine();
    if (!engine) return;

    for (auto& gr : geom_renderables_) {
        renderer_.scene()->remove(gr.entity);
        engine->destroy(gr.entity);
        if (gr.vertex_buffer) engine->destroy(gr.vertex_buffer);
        if (gr.index_buffer) engine->destroy(gr.index_buffer);
        if (gr.material_instance) engine->destroy(gr.material_instance);
    }
    geom_renderables_.clear();

    if (light_manager_) light_manager_->clear();
}

filament::math::mat4f SceneBridge::mj_to_filament_transform(
    const double* pos, const double* quat)
{
    // MuJoCo quaternion: w, x, y, z
    float w = static_cast<float>(quat[0]);
    float x = static_cast<float>(quat[1]);
    float y = static_cast<float>(quat[2]);
    float z = static_cast<float>(quat[3]);

    // Quaternion to rotation matrix
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    return filament::math::mat4f(
        filament::math::float4{1 - 2*(yy+zz), 2*(xy+wz), 2*(xz-wy), 0},
        filament::math::float4{2*(xy-wz), 1 - 2*(xx+zz), 2*(yz+wx), 0},
        filament::math::float4{2*(xz+wy), 2*(yz-wx), 1 - 2*(xx+yy), 0},
        filament::math::float4{(float)pos[0], (float)pos[1], (float)pos[2], 1}
    );
}


void SceneBridge::load_glb(const std::string& path) {
    auto* engine = renderer_.engine();
    
    // Read GLB file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return;
    }
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    
    // Create material provider (uses ubershader)
    auto* materials = filament::gltfio::createUbershaderProvider(engine,
        UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
    
    // Create asset loader
    filament::gltfio::AssetConfiguration config = {};
    config.engine = engine;
    config.materials = materials;
    auto* loader = filament::gltfio::AssetLoader::create(config);
    
    // Parse GLB
    auto* asset = loader->createAsset(buffer.data(), buffer.size());
    if (!asset) {
        return;
    }
    
    // Load resources (textures, buffers)
    // Register texture providers for PNG/JPEG decoding
    auto* stbDecoder = filament::gltfio::createStbProvider(engine);
    filament::gltfio::ResourceLoader resourceLoader({
        .engine = engine,
        .normalizeSkinningWeights = true
    });
    resourceLoader.addTextureProvider("image/png", stbDecoder);
    resourceLoader.addTextureProvider("image/jpeg", stbDecoder);
    resourceLoader.loadResources(asset);
    
    // Add to scene
    auto* scene = renderer_.scene();
    scene->addEntities(asset->getEntities(), asset->getEntityCount());
    
    // Note: asset must be kept alive for the duration of rendering
    // In production code, store the asset pointer
}

void SceneBridge::load_glb_at(const std::string& path, float x, float y, float z,
                               float scale) {
    auto* engine = renderer_.engine();
    
    // Read GLB file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return;
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    
    auto* materials = filament::gltfio::createUbershaderProvider(engine,
        UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
    
    filament::gltfio::AssetConfiguration config = {};
    config.engine = engine;
    config.materials = materials;
    auto* loader = filament::gltfio::AssetLoader::create(config);
    
    auto* asset = loader->createAsset(buffer.data(), buffer.size());
    if (!asset) return;
    
    // Register texture providers for PNG/JPEG decoding
    auto* stbDecoder = filament::gltfio::createStbProvider(engine);
    filament::gltfio::ResourceLoader resourceLoader({
        .engine = engine,
        .normalizeSkinningWeights = true
    });
    resourceLoader.addTextureProvider("image/png", stbDecoder);
    resourceLoader.addTextureProvider("image/jpeg", stbDecoder);
    resourceLoader.loadResources(asset);
    
    // Apply transform to the root entity
    auto& tcm = engine->getTransformManager();
    auto root = asset->getRoot();
    auto inst = tcm.getInstance(root);
    
    filament::math::mat4f transform = filament::math::mat4f::scaling(scale) *
        filament::math::mat4f(1.0f);
    transform[3] = filament::math::float4{x, y, z, 1.0f};
    
    tcm.setTransform(inst, transform);
    
    auto* scene = renderer_.scene();
    scene->addEntities(asset->getEntities(), asset->getEntityCount());
}

void SceneBridge::load_glb_xform(const std::string& path, const float* mat4x4) {
    auto* engine = renderer_.engine();
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return;
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();
    
    auto* materials = filament::gltfio::createUbershaderProvider(engine,
        UBERARCHIVE_DEFAULT_DATA, UBERARCHIVE_DEFAULT_SIZE);
    
    filament::gltfio::AssetConfiguration config = {};
    config.engine = engine;
    config.materials = materials;
    auto* loader = filament::gltfio::AssetLoader::create(config);
    
    auto* asset = loader->createAsset(buffer.data(), buffer.size());
    if (!asset) return;
    
    auto* stbDecoder = filament::gltfio::createStbProvider(engine);
    filament::gltfio::ResourceLoader resourceLoader({
        .engine = engine,
        .normalizeSkinningWeights = true
    });
    resourceLoader.addTextureProvider("image/png", stbDecoder);
    resourceLoader.addTextureProvider("image/jpeg", stbDecoder);
    resourceLoader.loadResources(asset);
    
    // Apply full 4x4 transform (column-major)
    auto& tcm = engine->getTransformManager();
    auto root = asset->getRoot();
    auto inst = tcm.getInstance(root);
    
    filament::math::mat4f transform(
        filament::math::float4{mat4x4[0], mat4x4[1], mat4x4[2], mat4x4[3]},
        filament::math::float4{mat4x4[4], mat4x4[5], mat4x4[6], mat4x4[7]},
        filament::math::float4{mat4x4[8], mat4x4[9], mat4x4[10], mat4x4[11]},
        filament::math::float4{mat4x4[12], mat4x4[13], mat4x4[14], mat4x4[15]}
    );
    
    tcm.setTransform(inst, transform);
    
    auto* scene = renderer_.scene();
    scene->addEntities(asset->getEntities(), asset->getEntityCount());
}

void SceneBridge::load_ibl(const std::string& ibl_path, const std::string& skybox_path,
                           bool with_skybox) {
    light_manager_->load_ibl(ibl_path, skybox_path, with_skybox);
}

void SceneBridge::clear_dynamic_lights() {
    light_manager_->clear_dynamic_lights();
}

void SceneBridge::set_ambient_intensity(float intensity) {
    light_manager_->set_indirect_light_intensity(intensity);
}

void SceneBridge::add_directional_light(float dx, float dy, float dz,
                                        float r, float g, float b,
                                        float intensity, bool cast_shadows) {
    light_manager_->add_directional_light(dx, dy, dz, r, g, b,
                                          intensity, cast_shadows);
}

void SceneBridge::add_point_light(float x, float y, float z,
                                  float r, float g, float b,
                                  float intensity, float falloff) {
    light_manager_->add_point_light(x, y, z, r, g, b, intensity, falloff, false);
}

void SceneBridge::add_spot_light(float x, float y, float z,
                                 float dx, float dy, float dz,
                                 float r, float g, float b,
                                 float intensity, float falloff,
                                 float inner_deg, float outer_deg,
                                 bool focused) {
    light_manager_->add_spot_light(x, y, z, dx, dy, dz, r, g, b,
                                   intensity, falloff, inner_deg, outer_deg,
                                   false, focused);
}

// ============================================================================
// Primitive geometry builders
// ============================================================================

void SceneBridge::build_sphere(float radius, int slices, int stacks,
                                std::vector<float>& vertices,
                                std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    for (int i = 0; i <= stacks; ++i) {
        float phi = M_PI * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);

        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float st = std::sin(theta), ct = std::cos(theta);

            float nx = sp * ct, ny = sp * st, nz = cp;
            // position
            vertices.push_back(radius * nx);
            vertices.push_back(radius * ny);
            vertices.push_back(radius * nz);
            // normal
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);
            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
    }
}

void SceneBridge::build_ellipsoid(float rx, float ry, float rz, int slices,
                                  int stacks, std::vector<float>& vertices,
                                  std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    for (int i = 0; i <= stacks; ++i) {
        float phi = M_PI * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);

        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float st = std::sin(theta), ct = std::cos(theta);

            // Unit-sphere direction.
            float ux = sp * ct, uy = sp * st, uz = cp;
            // Scaled position.
            vertices.push_back(rx * ux);
            vertices.push_back(ry * uy);
            vertices.push_back(rz * uz);
            // Correct ellipsoid normal = normalize(u / r^2) (inverse-transpose).
            float nx = ux / (rx * rx);
            float ny = uy / (ry * ry);
            float nz = uz / (rz * rz);
            float nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nl > 1e-12f) { nx/=nl; ny/=nl; nz/=nl; }
            else { nx=0; ny=0; nz=1; }
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);
            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
    }
}

void SceneBridge::build_box(float hx, float hy, float hz,
                             std::vector<float>& vertices,
                             std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    // 6 faces, each with 4 vertices (for proper normals)
    struct Face {
        float nx, ny, nz;
        float v[4][3];
    };

    Face faces[6] = {
        // +X
        {1,0,0, {{hx,-hy,-hz}, {hx,hy,-hz}, {hx,hy,hz}, {hx,-hy,hz}}},
        // -X
        {-1,0,0, {{-hx,hy,-hz}, {-hx,-hy,-hz}, {-hx,-hy,hz}, {-hx,hy,hz}}},
        // +Y
        {0,1,0, {{hx,hy,-hz}, {-hx,hy,-hz}, {-hx,hy,hz}, {hx,hy,hz}}},
        // -Y
        {0,-1,0, {{-hx,-hy,-hz}, {hx,-hy,-hz}, {hx,-hy,hz}, {-hx,-hy,hz}}},
        // +Z
        {0,0,1, {{-hx,-hy,hz}, {hx,-hy,hz}, {hx,hy,hz}, {-hx,hy,hz}}},
        // -Z
        {0,0,-1, {{hx,-hy,-hz}, {-hx,-hy,-hz}, {-hx,hy,-hz}, {hx,hy,-hz}}},
    };

    for (int f = 0; f < 6; ++f) {
        uint32_t base = static_cast<uint32_t>(vertices.size() / 6);
        for (int v = 0; v < 4; ++v) {
            vertices.push_back(faces[f].v[v][0]);
            vertices.push_back(faces[f].v[v][1]);
            vertices.push_back(faces[f].v[v][2]);
            vertices.push_back(faces[f].nx);
            vertices.push_back(faces[f].ny);
            vertices.push_back(faces[f].nz);
        }
        indices.push_back(base);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
}

void SceneBridge::build_capsule(float radius, float half_length,
                                 int slices, int stacks,
                                 std::vector<float>& vertices,
                                 std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    // Top hemisphere
    for (int i = 0; i <= stacks; ++i) {
        float phi = (M_PI / 2.0f) * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float nx = cp * std::cos(theta);
            float ny = cp * std::sin(theta);
            float nz = sp;
            vertices.push_back(radius * nx);
            vertices.push_back(radius * ny);
            vertices.push_back(half_length + radius * nz);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    uint32_t cyl_base = static_cast<uint32_t>(vertices.size() / 6);

    // Cylinder body
    for (int i = 0; i <= 1; ++i) {
        float z = (i == 0) ? half_length : -half_length;
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float nx = std::cos(theta), ny = std::sin(theta);
            vertices.push_back(radius * nx);
            vertices.push_back(radius * ny);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(0.0f);
        }
    }

    uint32_t bot_base = static_cast<uint32_t>(vertices.size() / 6);

    // Bottom hemisphere
    for (int i = 0; i <= stacks; ++i) {
        float phi = (M_PI / 2.0f) * static_cast<float>(i) / static_cast<float>(stacks);
        float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float nx = cp * std::cos(theta);
            float ny = cp * std::sin(theta);
            float nz = -sp;
            vertices.push_back(radius * nx);
            vertices.push_back(radius * ny);
            vertices.push_back(-half_length + radius * nz);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    // Top hemisphere indices
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            indices.push_back(a); indices.push_back(b); indices.push_back(a+1);
            indices.push_back(a+1); indices.push_back(b); indices.push_back(b+1);
        }
    }

    // Cylinder indices
    for (int j = 0; j < slices; ++j) {
        uint32_t a = cyl_base + j;
        uint32_t b = a + slices + 1;
        indices.push_back(a); indices.push_back(b); indices.push_back(a+1);
        indices.push_back(a+1); indices.push_back(b); indices.push_back(b+1);
    }

    // Bottom hemisphere indices. The bottom hemisphere is generated equator->pole
    // (nz goes 0 -> -1), the opposite vertical direction from the top, so the
    // triangle winding must be reversed to keep front-faces outward (otherwise
    // the bottom cap is back-facing and renders black under backface culling).
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            uint32_t a = bot_base + i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            indices.push_back(a); indices.push_back(a+1); indices.push_back(b);
            indices.push_back(a+1); indices.push_back(b+1); indices.push_back(b);
        }
    }
}

void SceneBridge::build_cylinder(float radius, float half_length, int slices,
                                  std::vector<float>& vertices,
                                  std::vector<uint32_t>& indices) {
    vertices.clear();
    indices.clear();

    // Side
    for (int i = 0; i <= 1; ++i) {
        float z = (i == 0) ? half_length : -half_length;
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
            float nx = std::cos(theta), ny = std::sin(theta);
            vertices.push_back(radius * nx);
            vertices.push_back(radius * ny);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(0.0f);
        }
    }

    for (int j = 0; j < slices; ++j) {
        uint32_t a = j, b = a + slices + 1;
        indices.push_back(a); indices.push_back(b); indices.push_back(a+1);
        indices.push_back(a+1); indices.push_back(b); indices.push_back(b+1);
    }

    // Top cap
    uint32_t center_top = static_cast<uint32_t>(vertices.size() / 6);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(half_length);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    for (int j = 0; j <= slices; ++j) {
        float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
        vertices.push_back(radius * std::cos(theta));
        vertices.push_back(radius * std::sin(theta));
        vertices.push_back(half_length);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(1);
    }
    for (int j = 0; j < slices; ++j) {
        indices.push_back(center_top);
        indices.push_back(center_top + 1 + j);
        indices.push_back(center_top + 2 + j);
    }

    // Bottom cap
    uint32_t center_bot = static_cast<uint32_t>(vertices.size() / 6);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(-half_length);
    vertices.push_back(0); vertices.push_back(0); vertices.push_back(-1);
    for (int j = 0; j <= slices; ++j) {
        float theta = 2.0f * M_PI * static_cast<float>(j) / static_cast<float>(slices);
        vertices.push_back(radius * std::cos(theta));
        vertices.push_back(radius * std::sin(theta));
        vertices.push_back(-half_length);
        vertices.push_back(0); vertices.push_back(0); vertices.push_back(-1);
    }
    for (int j = 0; j < slices; ++j) {
        indices.push_back(center_bot);
        indices.push_back(center_bot + 2 + j);
        indices.push_back(center_bot + 1 + j);
    }
}

void SceneBridge::build_plane(float size,
                               std::vector<float>& vertices,
                               std::vector<uint32_t>& indices) {
    // Simple quad — no subdivision needed for matte materials
    vertices = {
        -size, -size, 0.0f,  0, 0, 1,
         size, -size, 0.0f,  0, 0, 1,
         size,  size, 0.0f,  0, 0, 1,
        -size,  size, 0.0f,  0, 0, 1,
    };
    indices = {0, 1, 2, 0, 2, 3};
}

} // namespace vf_mujoco
