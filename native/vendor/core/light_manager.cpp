#include "core/light_manager.h"

#include <fstream>
#include <vector>
#include <filament/IndirectLight.h>
#include <filament/Skybox.h>
#include <filament/Texture.h>
#include <image/Ktx1Bundle.h>
#include <ktxreader/Ktx1Reader.h>
#include <utils/EntityManager.h>

namespace vf_mujoco {

LightManager::LightManager(filament::Engine* engine, filament::Scene* scene)
    : engine_(engine), scene_(scene) {}

LightManager::~LightManager() {
    clear();
}

void LightManager::setup_default_lighting() {
    // Clear existing lights but keep IBL/skybox if already loaded
    for (auto entity : light_entities_) {
        scene_->remove(entity);
        engine_->destroy(entity);
    }
    light_entities_.clear();

    // Primary overhead light — mimics warehouse ceiling lamps
    sun_entity_ = add_directional_light(
        -0.1f, -0.1f, -1.0f,
        1.0f, 0.97f, 0.92f,
        25000.0f,
        true
    );

    // Subtle fill — lifts shadows
    add_directional_light(
        0.5f, 0.3f, -0.6f,
        0.85f, 0.88f, 1.0f,
        4000.0f,
        false
    );

    // If no IBL loaded yet, use SH-based fallback from cmgen studio HDR
    if (!indirect_light_) {
        auto bands = std::array<filament::math::float3, 9>{
            filament::math::float3{0.875f, 0.853f, 0.855f},
            filament::math::float3{0.082f, 0.082f, 0.088f},
            filament::math::float3{0.216f, 0.242f, 0.264f},
            filament::math::float3{0.106f, 0.106f, 0.100f},
            filament::math::float3{-0.248f, -0.239f, -0.241f},
            filament::math::float3{0.146f, 0.160f, 0.176f},
            filament::math::float3{0.069f, 0.068f, 0.069f},
            filament::math::float3{-0.220f, -0.246f, -0.282f},
            filament::math::float3{0.071f, 0.071f, 0.073f},
        };

        indirect_light_ = filament::IndirectLight::Builder()
            .irradiance(3, bands.data())
            .intensity(8000.0f)
            .build(*engine_);

        scene_->setIndirectLight(indirect_light_);
    }
}

void LightManager::load_ibl(const std::string& ibl_path, const std::string& skybox_path) {
    // --- Load IBL reflections cubemap ---
    {
        std::ifstream file(ibl_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return;
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();

        auto* bundle = new image::Ktx1Bundle(data.data(), data.size());
        ibl_texture_ = ktxreader::Ktx1Reader::createTexture(engine_, bundle, false);

        if (ibl_texture_) {
            // Read SH from the KTX bundle metadata
            filament::math::float3 sh[9];
            if (bundle->getSphericalHarmonics(sh)) {
                // Clean up old IBL
                if (indirect_light_) {
                    engine_->destroy(indirect_light_);
                }
                indirect_light_ = filament::IndirectLight::Builder()
                    .reflections(ibl_texture_)
                    .irradiance(3, sh)
                    .intensity(18000.0f)
                    .build(*engine_);
                scene_->setIndirectLight(indirect_light_);
            } else {
                // Fallback: use reflections cubemap without SH
                if (indirect_light_) {
                    engine_->destroy(indirect_light_);
                }
                // Use the hardcoded SH from cmgen
                auto bands = std::array<filament::math::float3, 9>{
                    filament::math::float3{0.875f, 0.853f, 0.855f},
                    filament::math::float3{0.082f, 0.082f, 0.088f},
                    filament::math::float3{0.216f, 0.242f, 0.264f},
                    filament::math::float3{0.106f, 0.106f, 0.100f},
                    filament::math::float3{-0.248f, -0.239f, -0.241f},
                    filament::math::float3{0.146f, 0.160f, 0.176f},
                    filament::math::float3{0.069f, 0.068f, 0.069f},
                    filament::math::float3{-0.220f, -0.246f, -0.282f},
                    filament::math::float3{0.071f, 0.071f, 0.073f},
                };
                indirect_light_ = filament::IndirectLight::Builder()
                    .reflections(ibl_texture_)
                    .irradiance(3, bands.data())
                    .intensity(18000.0f)
                    .build(*engine_);
                scene_->setIndirectLight(indirect_light_);
            }
        }
    }

    // --- Load skybox cubemap ---
    {
        std::ifstream file(skybox_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return;
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();

        auto* bundle = new image::Ktx1Bundle(data.data(), data.size());
        skybox_texture_ = ktxreader::Ktx1Reader::createTexture(engine_, bundle, false);

        if (skybox_texture_) {
            if (skybox_) {
                engine_->destroy(skybox_);
            }
            skybox_ = filament::Skybox::Builder()
                .environment(skybox_texture_)
                .showSun(true)
                .build(*engine_);
            scene_->setSkybox(skybox_);
        }
    }
}

utils::Entity LightManager::add_directional_light(
    float dir_x, float dir_y, float dir_z,
    float r, float g, float b,
    float intensity, bool cast_shadows)
{
    auto entity = utils::EntityManager::get().create();

    auto builder = filament::LightManager::Builder(
        filament::LightManager::Type::SUN)
        .direction({dir_x, dir_y, dir_z})
        .color({r, g, b})
        .intensity(intensity)
        .castShadows(cast_shadows);

    if (cast_shadows) {
        filament::LightManager::ShadowOptions shadowOpts;
        shadowOpts.mapSize = 2048;
        shadowOpts.shadowCascades = 3;
        shadowOpts.constantBias = 0.001f;
        shadowOpts.normalBias = 1.0f;
        builder.shadowOptions(shadowOpts);
    }

    builder.build(*engine_, entity);

    scene_->addEntity(entity);
    light_entities_.push_back(entity);
    return entity;
}

utils::Entity LightManager::add_point_light(
    float pos_x, float pos_y, float pos_z,
    float r, float g, float b,
    float intensity, float falloff_radius,
    bool cast_shadows)
{
    auto entity = utils::EntityManager::get().create();

    filament::LightManager::Builder(filament::LightManager::Type::POINT)
        .position({pos_x, pos_y, pos_z})
        .color({r, g, b})
        .intensity(intensity)
        .falloff(falloff_radius)
        .castShadows(cast_shadows)
        .build(*engine_, entity);

    scene_->addEntity(entity);
    light_entities_.push_back(entity);
    return entity;
}

utils::Entity LightManager::add_spot_light(
    float pos_x, float pos_y, float pos_z,
    float dir_x, float dir_y, float dir_z,
    float r, float g, float b,
    float intensity, float falloff_radius,
    float inner_cone_deg, float outer_cone_deg,
    bool cast_shadows, bool focused)
{
    auto entity = utils::EntityManager::get().create();

    constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;

    // FOCUSED_SPOT concentrates a fixed luminous power into the cone (narrow cone
    // => bright saturated hotspot). SPOT decouples illumination from the cone
    // width so the light reads as an even wash across the whole cone — which is
    // what we want for broad, soft, ceiling-mounted warehouse fixtures.
    auto builder = filament::LightManager::Builder(
        focused ? filament::LightManager::Type::FOCUSED_SPOT
                : filament::LightManager::Type::SPOT)
        .position({pos_x, pos_y, pos_z})
        .direction({dir_x, dir_y, dir_z})
        .color({r, g, b})
        .intensity(intensity)
        .falloff(falloff_radius)
        .spotLightCone(inner_cone_deg * DEG2RAD, outer_cone_deg * DEG2RAD)
        .castShadows(cast_shadows);

    if (cast_shadows) {
        filament::LightManager::ShadowOptions shadowOpts;
        shadowOpts.mapSize = 1024;
        builder.shadowOptions(shadowOpts);
    }

    builder.build(*engine_, entity);

    scene_->addEntity(entity);
    light_entities_.push_back(entity);
    return entity;
}

void LightManager::clear_dynamic_lights() {
    for (auto entity : light_entities_) {
        scene_->remove(entity);
        engine_->destroy(entity);
    }
    light_entities_.clear();
}

void LightManager::set_indirect_light_intensity(float intensity) {
    auto* ibl = scene_->getIndirectLight();
    if (ibl) {
        ibl->setIntensity(intensity);
    }
}

void LightManager::clear() {
    for (auto entity : light_entities_) {
        scene_->remove(entity);
        engine_->destroy(entity);
    }
    light_entities_.clear();

    if (skybox_) {
        scene_->setSkybox(nullptr);
        engine_->destroy(skybox_);
        skybox_ = nullptr;
    }
    if (indirect_light_) {
        scene_->setIndirectLight(nullptr);
        engine_->destroy(indirect_light_);
        indirect_light_ = nullptr;
    }
    if (ibl_texture_) {
        engine_->destroy(ibl_texture_);
        ibl_texture_ = nullptr;
    }
    if (skybox_texture_) {
        engine_->destroy(skybox_texture_);
        skybox_texture_ = nullptr;
    }
}

} // namespace vf_mujoco
