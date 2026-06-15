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
    int mj_geom_id = -1;
    int mj_geom_type = -1;
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

    /// Load IBL environment map from KTX files (cmgen output).
    void load_ibl(const std::string& ibl_path, const std::string& skybox_path);

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

    /// Create primitive geometry (box, sphere, capsule, cylinder, plane).
    void create_primitive(int geom_id, int geom_type,
                          const mjtNum* size, const float* rgba,
                          bool has_mat = false, float mat_metallic = 0.0f,
                          float mat_roughness = 0.5f);

    /// Create mesh geometry from MuJoCo mesh data. Honors the geom's MuJoCo
    /// material (metallic/roughness/color) when has_mat is set.
    void create_mesh(const mjModel* model, int geom_id, int mesh_id,
                     const float* rgba, bool has_mat = false,
                     float mat_metallic = 0.0f, float mat_roughness = 0.5f);

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

    /// Upload vertex + index data and create a renderable entity.
    GeomRenderable create_renderable(const std::vector<float>& vertices,
                                     const std::vector<uint32_t>& indices,
                                     filament::MaterialInstance* mat_inst,
                                     int geom_id);

    Renderer& renderer_;
    std::unique_ptr<MaterialManager> material_manager_;
    std::unique_ptr<LightManager> light_manager_;
    std::vector<GeomRenderable> geom_renderables_;
    std::vector<uint8_t> batch_scratch_;  // reused RGBA readback buffer for render_batch_rgb
};

} // namespace vf_mujoco
