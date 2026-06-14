#include <fstream>
#include <gltfio/materials/uberarchive.h>
#include "core/scene_bridge.h"

#include <cmath>
#include <cstring>
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

    // Setup default lighting (sun + ambient)
    light_manager_->setup_default_lighting();

    // Create Filament renderables for each MuJoCo geom
    for (int i = 0; i < model->ngeom; ++i) {
        create_geom(model, i);
    }
}

void SceneBridge::create_geom(const mjModel* model, int geom_id) {
    int geom_type = model->geom_type[geom_id];
    const mjtNum* geom_size = model->geom_size + geom_id * 3;
    const float* geom_rgba = model->geom_rgba + geom_id * 4;

    // Skip invisible geoms (collision-only, group >= 3, or transparent)
    int geom_group = model->geom_group[geom_id];
    if (geom_group >= 3) return;  // collision-only
    if (geom_rgba[3] < 0.01f) return;  // fully transparent

    // If the geom references a MuJoCo material, honor its PBR metallic/roughness
    // (and color) directly instead of guessing from brightness. This lets MJCF
    // authors drive real PBR (metal/glossy/rough) for primitive scenes.
    bool has_mat = false;
    float mat_metallic = 0.0f, mat_roughness = 0.5f;
    const float* mat_color = geom_rgba;
    int matid = model->geom_matid[geom_id];
    if (matid >= 0) {
        has_mat = true;
        mat_metallic = model->mat_metallic[matid];
        mat_roughness = model->mat_roughness[matid];
        // geom_rgba defaults to (0.5,0.5,0.5,1); if a material is set and the
        // geom color is the default, use the material's rgba.
        mat_color = model->mat_rgba + matid * 4;
    }

    // Render MuJoCo primitive geoms directly. The GLB-based warehouse only used
    // the plane from MuJoCo (everything else came from GLBs), but primitive
    // scenes (e.g. MJWarp worlds) need box/sphere/capsule/cylinder too.
    switch (geom_type) {
        case mjGEOM_PLANE:
        case mjGEOM_SPHERE:
        case mjGEOM_BOX:
        case mjGEOM_CAPSULE:
        case mjGEOM_CYLINDER:
            create_primitive(geom_id, geom_type, geom_size, mat_color,
                             has_mat, mat_metallic, mat_roughness);
            break;
        default:
            // mesh / ellipsoid / hfield etc. still come from GLBs when used
            break;
    }
}

void SceneBridge::create_primitive(int geom_id, int geom_type,
                                    const mjtNum* size, const float* rgba,
                                    bool has_mat, float mat_metallic,
                                    float mat_roughness) {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;

    switch (geom_type) {
        case mjGEOM_SPHERE:
            build_sphere(size[0], SPHERE_SLICES, SPHERE_STACKS, vertices, indices);
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

    // Determine PBR properties from color
    float brightness = (rgba[0] + rgba[1] + rgba[2]) / 3.0f;
    float roughness = 0.4f;  // default
    float metallic = 0.0f;
    float reflectance = 0.5f;

    // Dark surfaces = metal/rubber (robot joints)
    if (brightness < 0.25f) {
        roughness = 0.45f;
        metallic = 0.4f;
        reflectance = 0.35f;
    }
    // White/light grey = matte plastic (robot body)
    else if (brightness > 0.8f) {
        roughness = 0.4f;
        metallic = 0.0f;
        reflectance = 0.25f;
    }
    // Red accents
    else if (rgba[0] > 0.7f && rgba[1] < 0.3f) {
        roughness = 0.4f;
        metallic = 0.0f;
        reflectance = 0.25f;
    }
    // Mid-tone = general objects
    else {
        roughness = 0.5f;
        metallic = 0.0f;
        reflectance = 0.2f;
    }

    // A real MuJoCo material overrides the color-brightness heuristic.
    if (has_mat) {
        roughness = mat_roughness;
        metallic = mat_metallic;
        reflectance = 0.5f;
    }

    float final_r = rgba[0], final_g = rgba[1], final_b = rgba[2], final_a = rgba[3];
    if (geom_type == mjGEOM_PLANE && !has_mat) {
        // Untextured/material-less plane: honor color but keep it fully diffuse
        // (roughness 1, no specular) so IBL doesn't band on the large surface.
        roughness = 1.0f;
        metallic = 0.0f;
        reflectance = 0.0f;
    }
    auto* mat_inst = material_manager_->create_pbr_instance(
        final_r, final_g, final_b, final_a,
        roughness,
        metallic,
        reflectance
    );

    auto renderable = create_renderable(vertices, indices, mat_inst, geom_id);
    renderable.mj_geom_type = geom_type;
    geom_renderables_.push_back(std::move(renderable));
}

void SceneBridge::create_mesh(const mjModel* model, int geom_id, int mesh_id) {
    // Extract mesh data from MuJoCo
    int vert_start = model->mesh_vertadr[mesh_id];
    int vert_count = model->mesh_vertnum[mesh_id];
    int face_start = model->mesh_faceadr[mesh_id];
    int face_count = model->mesh_facenum[mesh_id];

    if (vert_count == 0 || face_count == 0) return;

    const float* mj_verts = model->mesh_vert + vert_start * 3;
    const float* mj_normals = model->mesh_normal + vert_start * 3;
    const int* mj_faces = model->mesh_face + face_start * 3;

    // Interleaved vertex data: position (3) + normal (3) = 6 floats per vertex
    std::vector<float> vertices(vert_count * 6);
    for (int i = 0; i < vert_count; ++i) {
        vertices[i * 6 + 0] = mj_verts[i * 3 + 0];
        vertices[i * 6 + 1] = mj_verts[i * 3 + 1];
        vertices[i * 6 + 2] = mj_verts[i * 3 + 2];
        vertices[i * 6 + 3] = mj_normals[i * 3 + 0];
        vertices[i * 6 + 4] = mj_normals[i * 3 + 1];
        vertices[i * 6 + 5] = mj_normals[i * 3 + 2];
    }

    std::vector<uint32_t> indices(face_count * 3);
    for (int i = 0; i < face_count * 3; ++i) {
        indices[i] = static_cast<uint32_t>(mj_faces[i]);
    }

    const float* geom_rgba = model->geom_rgba + geom_id * 4;
    // YCB objects: slightly rough, non-metallic
    auto* mat_inst = material_manager_->create_pbr_instance(
        geom_rgba[0], geom_rgba[1], geom_rgba[2], geom_rgba[3],
        0.4f,   // roughness
        0.0f,   // metallic
        0.5f    // reflectance
    );

    auto renderable = create_renderable(vertices, indices, mat_inst, geom_id);
    renderable.mj_geom_type = mjGEOM_MESH;
    geom_renderables_.push_back(std::move(renderable));
}

GeomRenderable SceneBridge::create_renderable(
    const std::vector<float>& vertices,
    const std::vector<uint32_t>& indices,
    filament::MaterialInstance* mat_inst,
    int geom_id)
{
    auto* engine = renderer_.engine();

    uint32_t vertex_count = static_cast<uint32_t>(vertices.size() / 6);
    uint32_t index_count = static_cast<uint32_t>(indices.size());

    // Vertex buffer: position (buffer 0) + tangents as SHORT4 (buffer 1)
    auto* vb = filament::VertexBuffer::Builder()
        .vertexCount(vertex_count)
        .bufferCount(2)
        .attribute(filament::VertexAttribute::POSITION, 0,
                   filament::VertexBuffer::AttributeType::FLOAT3, 0, 24)
        .attribute(filament::VertexAttribute::TANGENTS, 1,
                   filament::VertexBuffer::AttributeType::SHORT4, 0, 8)
        .normalized(filament::VertexAttribute::TANGENTS)
        .build(*engine);

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
        .castShadows(false)
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

void SceneBridge::load_ibl(const std::string& ibl_path, const std::string& skybox_path) {
    light_manager_->load_ibl(ibl_path, skybox_path);
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
