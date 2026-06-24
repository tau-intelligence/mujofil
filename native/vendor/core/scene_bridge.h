#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>
#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/MaterialInstance.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/RenderableManager.h>
#include <filament/InstanceBuffer.h>
#include <filament/TransformManager.h>
#include <utils/Entity.h>
#include <math/mat4.h>

#include "core/renderer.h"
#include <gltfio/AssetLoader.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/MaterialProvider.h>
#include <gltfio/TextureProvider.h>
#include "core/material_manager.h"
#include "core/light_manager.h"

namespace vf_mujoco {

/// Holds Filament resources for a single MuJoCo geometry.
struct GeomRenderable {
    utils::Entity entity;
    filament::VertexBuffer* vertex_buffer = nullptr;
    filament::IndexBuffer* index_buffer = nullptr;
    filament::MaterialInstance* material_instance = nullptr;
    filament::InstanceBuffer* instance_buffer = nullptr;  // layered mode: N world poses
    int mj_geom_id = -1;
    int mj_geom_type = -1;
    // STATIC instanced env mesh (e.g. a GLB photoreal environment ingested into the
    // layered instanced path): its world pose is FIXED (base_xform), the same in
    // every world, view-folded per world in egocentric mode. Unlike MuJoCo geoms
    // it is not driven by mj_geom_id (which stays -1).
    bool static_instanced = false;
    filament::math::mat4f base_xform{};  // identity by default
};

/// Fully resolved appearance for one geom: PBR scalars + optional MuJoCo texture.
/// Computed once in create_geom and consumed by create_primitive/create_mesh.
struct ResolvedMaterial {
    float rgba[4] = {0.5f, 0.5f, 0.5f, 1.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    float reflectance = 0.5f;
    float emissive = 0.0f;
    float uvscale[2] = {1.0f, 1.0f};
    bool has_mat = false;                 // a MuJoCo <material> drives PBR scalars
    filament::Texture* albedo_2d = nullptr;  // 2D texture (planes/hfields/meshes)
    filament::Texture* cube = nullptr;       // cube texture (other geoms)
    bool textured() const { return albedo_2d != nullptr || cube != nullptr; }
};

/// Bridges MuJoCo simulation state to Filament scene graph.
/// Handles: mesh conversion, transform syncing, material assignment.
class SceneBridge {
public:
    SceneBridge(Renderer& renderer);
    ~SceneBridge();

    /// Load a MuJoCo model and build the corresponding Filament scene.
    /// Call once after loading the model.
    void load_model(const mjModel* model);

    /// Sync transforms from MuJoCo simulation data to Filament entities.
    /// Call every frame before rendering.
    void sync_transforms(const mjModel* model, const mjData* data);

    /// Enable LAYERED instanced mode BEFORE load_model: each geom renderable is
    /// built instanced n_worlds times (one InstanceBuffer of n_worlds transforms),
    /// so a single render() draws all worlds at once (forked gl_Layer routing).
    void enable_layered(int n_worlds);

    /// Layered mode per-frame sync: fill each geom's InstanceBuffer with that
    /// geom's world pose across all N datas (datas.size() must == n_worlds).
    void sync_transforms_layered(const mjModel* model,
                                 const std::vector<const mjData*>& datas);

    /// Layered EGOCENTRIC setup: give each world its OWN camera (e.g. a robot-
    /// mounted MuJoCo camera) in the single instanced draw. Computes each world's
    /// view matrix V_w and binds a shared projection-only camera; sync_transforms_
    /// layered then folds V_w into each world's InstanceBuffer transform. cam_id < 0
    /// disables egocentric (shared-camera path, byte-identical). Call BEFORE
    /// sync_transforms_layered.
    void sync_cameras_layered(const mjModel* model,
                              const std::vector<const mjData*>& datas,
                              int cam_id);

    /// Sync the camera to match a MuJoCo camera.
    /// cam_id: MuJoCo camera ID, or -1 for free camera.
    void sync_camera(const mjModel* model, const mjData* data, int cam_id = -1);

    /// Batched render of N environments that share this model but have distinct
    /// mjData states, into a single (N, H, W, 3) RGB buffer. Each env's
    /// transforms are synced and the frame is rendered + queued for async
    /// readback; a SINGLE GPU sync is performed at the end. This amortizes the
    /// per-frame fence/submit cost (the dominant overhead at RL resolutions)
    /// across the whole batch. cam_id < 0 keeps the current free camera.
    void render_batch_rgb(const mjModel* model,
                          const std::vector<const mjData*>& datas,
                          int cam_id, uint8_t* out_rgb);

    /// Set free camera position and orientation.
    void set_free_camera(float eye_x, float eye_y, float eye_z,
                         float target_x, float target_y, float target_z);

    /// Load a GLB file and add to scene.
    void load_glb(const std::string& path);

    /// Load a GLB file at a specific position/rotation.
    void load_glb_at(const std::string& path, float x, float y, float z,
                     float scale = 1.0f);

    /// Load a GLB file with a full 4x4 transform matrix (column-major).
    void load_glb_xform(const std::string& path, const float* mat4x4);

    /// Ingest ONE mesh of a GLB environment as an INSTANCED LAYERED renderable so
    /// it participates in the single-draw parallel-batch path AND in per-world
    /// egocentric cameras (view-folding) -- unlike load_glb*, which adds the GLB as
    /// a single shared non-instanced backdrop that cannot be egocentric. Geometry
    /// is given in WORLD space (caller bakes the scene transform into the verts).
    /// positions/normals: vert_count*3 floats each. uvs: vert_count*2 or null.
    /// indices: index_count uint32. albedo_rgba: tex_w*tex_h*4 bytes or null.
    /// Requires layered mode (enable_layered) to have been set before load_model.
    void add_layered_env_mesh(
        const float* positions, const float* normals, int vert_count,
        const float* uvs, const float* tangents4,
        const uint32_t* indices, int index_count,
        float r, float g, float b, float a,
        float roughness, float metallic,
        float emissive_r, float emissive_g, float emissive_b,
        const uint8_t* albedo_rgba, int albedo_w, int albedo_h,
        const uint8_t* normal_rgba, int normal_w, int normal_h,
        const uint8_t* mr_rgba, int mr_w, int mr_h,
        const uint8_t* emissive_rgba, int emissive_w, int emissive_h);

    /// Load IBL environment map from KTX files (cmgen output).
    void load_ibl(const std::string& ibl_path, const std::string& skybox_path,
                  bool with_skybox = true);

    /// Remove the default directional/point/spot lights (keeps IBL + skybox).
    void clear_dynamic_lights();

    /// Set the indirect (ambient) light intensity.
    void set_ambient_intensity(float intensity);

    /// Add a directional light (like an even overhead/sun fill). Unlike point and
    /// spot lights it has no position, no distance falloff and is never froxel-
    /// culled, so it lights the whole scene evenly regardless of camera position.
    void add_directional_light(float dx, float dy, float dz,
                               float r, float g, float b,
                               float intensity, bool cast_shadows = false);

    /// Add a point light at a world position.
    void add_point_light(float x, float y, float z,
                         float r, float g, float b,
                         float intensity, float falloff);

    /// Add a downward (or directional) spot light.
    /// focused=false uses Filament's Type::SPOT (even spread across the cone);
    /// focused=true uses Type::FOCUSED_SPOT (physically correct, can hotspot).
    void add_spot_light(float x, float y, float z,
                        float dx, float dy, float dz,
                        float r, float g, float b,
                        float intensity, float falloff,
                        float inner_deg, float outer_deg,
                        bool focused = true);

    /// Clear all renderables from the scene.
    void clear();

    /// Get number of loaded geom renderables.
    size_t geom_count() const { return geom_renderables_.size(); }

private:
    /// Create Filament renderable for a MuJoCo geom.
    void create_geom(const mjModel* model, int geom_id);

    /// Resolve the full appearance (PBR scalars + textures) for a geom.
    ResolvedMaterial resolve_material(const mjModel* model, int geom_id);

    /// Build a Filament material instance from a resolved material.
    filament::MaterialInstance* make_instance(const ResolvedMaterial& rm);

    /// Create Filament lights matching the MuJoCo model's <light> elements.
    /// Returns the number of lights created (0 if the model has none).
    int sync_lights_from_model(const mjModel* model);

    /// Create primitive geometry (box, sphere, capsule, cylinder, plane, ellipsoid).
    void create_primitive(int geom_id, int geom_type,
                          const mjtNum* size, const ResolvedMaterial& rm);

    /// Create mesh geometry from MuJoCo mesh data.
    void create_mesh(const mjModel* model, int geom_id, int mesh_id,
                     const ResolvedMaterial& rm);

    /// Create a height-field (terrain) renderable from MuJoCo hfield data.
    void create_hfield(const mjModel* model, int geom_id, int hfield_id,
                       const mjtNum* size, const ResolvedMaterial& rm);

    /// Build a sphere vertex/index buffer.
    void build_sphere(float radius, int slices, int stacks,
                      std::vector<float>& vertices,
                      std::vector<uint32_t>& indices);

    /// Build an ellipsoid (per-axis scaled sphere) vertex/index buffer.
    void build_ellipsoid(float rx, float ry, float rz, int slices, int stacks,
                         std::vector<float>& vertices,
                         std::vector<uint32_t>& indices);

    /// Build a box vertex/index buffer.
    void build_box(float hx, float hy, float hz,
                   std::vector<float>& vertices,
                   std::vector<uint32_t>& indices);

    /// Build a capsule vertex/index buffer.
    void build_capsule(float radius, float half_length, int slices, int stacks,
                       std::vector<float>& vertices,
                       std::vector<uint32_t>& indices);

    /// Build a cylinder vertex/index buffer.
    void build_cylinder(float radius, float half_length, int slices,
                        std::vector<float>& vertices,
                        std::vector<uint32_t>& indices);

    /// Build a ground plane.
    void build_plane(float size,
                     std::vector<float>& vertices,
                     std::vector<uint32_t>& indices);

    /// Convert MuJoCo position + quaternion to Filament mat4.
    filament::math::mat4f mj_to_filament_transform(const double* pos,
                                                     const double* quat);

    /// Upload vertex + index data and create a renderable entity. When uvs is
    /// non-empty (2 floats per vertex) a UV0 attribute is added so textured
    /// materials can sample; otherwise the fast position+tangent path is used.
    GeomRenderable create_renderable(const std::vector<float>& vertices,
                                     const std::vector<uint32_t>& indices,
                                     filament::MaterialInstance* mat_inst,
                                     int geom_id,
                                     const std::vector<float>& uvs = {},
                                     bool cast_shadows = true,
                                     const std::vector<float>& tangents4 = {});

    Renderer& renderer_;
    std::unique_ptr<MaterialManager> material_manager_;
    std::unique_ptr<LightManager> light_manager_;
    std::vector<GeomRenderable> geom_renderables_;
    std::vector<uint8_t> batch_scratch_;  // reused RGBA readback buffer for render_batch_rgb
    bool layered_ = false;                // instanced/gl_Layer parallel-batch mode
    int n_worlds_ = 1;                    // # worlds (= array layers) in layered mode
    std::vector<filament::math::mat4f> world_scratch_;  // reused N-transform buffer

    // Egocentric (per-world camera) layered support via VIEW-FOLDING: each world's
    // view matrix V_w is folded into that world's InstanceBuffer transform
    // (localTransform_w = V_w * geomPose_w) and a shared projection-only camera is
    // bound, so position = P * V_w * geomPose_w * vert renders each world from its
    // own camera in one instanced draw -- using only the proven per-instance
    // transform + frame-uniform projection (no per-instance clip matrix, which
    // miscompiles in the layered vertex shader on this GL driver). cam_scratch_
    // camera builds V_w / P; egocentric_view_ holds the per-world V_w used by
    // sync_transforms_layered.
    filament::Camera* cam_scratch_camera_ = nullptr;
    utils::Entity cam_scratch_entity_;
    std::vector<filament::math::mat4f> egocentric_view_;   // per-world V_w
    bool egocentric_ = false;
    // Saved bound-camera state captured on the shared->egocentric transition and
    // restored on egocentric->shared, so mixing the two modes on one renderer is
    // robust (egocentric binds a projection-only camera that would otherwise leak
    // into a later shared-camera render).
    bool have_saved_cam_ = false;
    filament::math::mat4 saved_cam_model_;
    filament::math::mat4 saved_cam_proj_;
    double saved_cam_near_ = 0.05;
    double saved_cam_far_ = 200.0;
    int env_tex_key_ = -100000;   // unique cache keys for ingested GLB-env textures
};

} // namespace vf_mujoco
