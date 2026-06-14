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

    /// Create a PBR material instance with the given parameters.
    filament::MaterialInstance* create_pbr_instance(
        float r, float g, float b, float a,
        float roughness = 0.5f,
        float metallic = 0.0f,
        float reflectance = 0.5f);

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

    filament::Engine* engine_;
    filament::Material* default_material_ = nullptr;
    std::unordered_map<std::string, filament::Material*> materials_;
    std::vector<filament::MaterialInstance*> instances_;
};

} // namespace vf_mujoco
