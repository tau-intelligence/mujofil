#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <filament/Engine.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/Texture.h>
#include <math/vec3.h>
#include <math/vec4.h>

namespace vf_mujoco {

/// Manages PBR materials for the Filament renderer.
/// Creates and caches Material objects and produces MaterialInstances
/// with per-geom parameters (base color, roughness, metallic).
class MaterialManager {
public:
    explicit MaterialManager(filament::Engine* engine);
    ~MaterialManager();

    /// Load the default PBR material from compiled .filamat data.
    void initialize();

    /// Load a compiled material from a .filamat file on disk.
    void load_material(const std::string& name, const std::string& filepath);

    /// Load a material from in-memory .filamat data.
    void load_material(const std::string& name, const uint8_t* data, size_t size);

    /// Create a PBR material instance with the given parameters. When alpha < 1
    /// a transparent (fade-blended) material is used so MuJoCo's semi-transparent
    /// geoms render correctly; otherwise the fast opaque material is used.
    filament::MaterialInstance* create_pbr_instance(
        float r, float g, float b, float a,
        float roughness = 0.5f,
        float metallic = 0.0f,
        float reflectance = 0.5f,
        float emissive = 0.0f);

    /// Create a textured PBR instance that samples a MuJoCo texture. Exactly one
    /// of albedo_2d / cube may be set; uvscale applies MuJoCo's texrepeat.
    filament::MaterialInstance* create_mujoco_textured_instance(
        float r, float g, float b, float a,
        float roughness, float metallic, float reflectance, float emissive,
        float uvscale_x, float uvscale_y,
        filament::Texture* albedo_2d, filament::Texture* cube);

    /// Create a layered-env textured PBR instance with the full glTF map set
    /// (albedo + normal + metallic-roughness + emissive). Any map may be null;
    /// the corresponding use* flag is set automatically. Used by the GLB-ingest
    /// (load_glb_layered) path so a photoreal environment keeps its surface detail
    /// in the instanced layered draw.
    filament::MaterialInstance* create_env_textured_instance(
        float r, float g, float b, float a,
        float roughness, float metallic, float reflectance,
        float emissive_r, float emissive_g, float emissive_b,
        filament::Texture* albedo_2d,
        filament::Texture* normal_2d,
        filament::Texture* mr_2d,
        filament::Texture* emissive_2d);

    /// Get (or create + cache) a Filament 2D texture from MuJoCo pixel data.
    /// key is the MuJoCo texture id (for caching). nchannel is 1/3/4. srgb=false
    /// uploads a LINEAR texture (for data maps: normal / metallic-roughness).
    filament::Texture* get_or_create_texture_2d(
        int key, int width, int height, int nchannel, const uint8_t* data,
        bool srgb = true);

    /// Get (or create + cache) a Filament cubemap from MuJoCo cube texture data
    /// (6 square faces stacked vertically, height == 6*width).
    filament::Texture* get_or_create_texture_cube(
        int key, int width, int height, int nchannel, const uint8_t* data);

    /// Create a textured PBR material instance.
    filament::MaterialInstance* create_textured_instance(
        filament::Texture* albedo_map,
        filament::Texture* normal_map = nullptr,
        filament::Texture* roughness_map = nullptr,
        float roughness = 0.5f,
        float metallic = 0.0f);

    /// Destroy all materials and instances.
    void clear();

    /// Get the default PBR material.
    filament::Material* default_material() const { return default_material_; }

private:
    void create_default_material();
    filament::Material* load_named_material(const std::string& filename);
    filament::Texture* dummy_2d();
    filament::Texture* dummy_cube();
    filament::Texture* dummy_normal();   // 1x1 flat tangent normal (0.5,0.5,1) linear
    filament::Texture* dummy_white_lin();// 1x1 white LINEAR (unused MR/emissive map)

    filament::Engine* engine_;
    filament::Material* default_material_ = nullptr;
    filament::Material* textured_material_ = nullptr;
    filament::Material* blend_material_ = nullptr;
    std::unordered_map<std::string, filament::Material*> materials_;
    std::vector<filament::MaterialInstance*> instances_;
    std::unordered_map<int, filament::Texture*> texture_cache_;
    filament::Texture* dummy_2d_ = nullptr;     // 1x1 white, for unused sampler2d
    filament::Texture* dummy_cube_ = nullptr;   // 1x1 white cube, for unused samplerCube
    filament::Texture* dummy_normal_ = nullptr; // 1x1 flat normal, linear
    filament::Texture* dummy_white_lin_ = nullptr; // 1x1 white, linear
};

} // namespace vf_mujoco
