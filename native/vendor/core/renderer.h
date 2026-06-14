// mujofil-warp's own vf_mujoco::Renderer — a shared-device Filament renderer
// that renders into an EXPORTABLE Vulkan swapchain and exposes the pixels to
// CUDA with no CPU round-trip.
//
// It presents the SAME accessor interface that mujofil's SceneBridge/
// MaterialManager/LightManager expect (engine/scene/view/camera/width/height),
// so those source files compile UNCHANGED against this header. All Vulkan/bluevk/
// CUDA state is hidden in a PIMPL so SceneBridge's translation unit never pulls
// in bluevk (which forbids including <vulkan/vulkan.h>).
//
// mujofil's PyPI package is NOT modified or rebuilt by any of this.
#pragma once

#include <cstdint>
#include <memory>

#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/SwapChain.h>
#include <filament/ColorGrading.h>
#include <utils/Entity.h>

namespace vf_mujoco {

struct RendererConfig {
    uint32_t width = 256;
    uint32_t height = 256;
    bool enable_ssao = true;
    // SSAO quality: 0=LOW, 1=MEDIUM, 2=HIGH, 3=ULTRA. SSAO is the dominant GPU
    // cost; lower quality is much faster with little visual change. ssao_ssct
    // toggles screen-space cone tracing (extra contact shadows, expensive).
    uint8_t ssao_quality = 3;
    bool ssao_ssct = true;
    bool enable_bloom = false;
    bool enable_fxaa = false;
    bool enable_msaa = true;
    uint8_t msaa_samples = 4;
    bool enable_shadows = true;
    float exposure = 1.0f;
    bool tone_mapping = true;
    bool dithering = true;
    uint32_t batch_size = 1;   // # of swapchain images for batched rendering
};

class Renderer {
public:
    explicit Renderer(const RendererConfig& config = {});
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool initialize();
    void destroy();

    // --- accessors used by SceneBridge / MaterialManager / LightManager ---
    filament::Engine* engine() const { return engine_; }
    filament::Scene* scene() const { return scene_; }
    filament::View* view() const { return view_; }
    filament::Camera* camera() const { return camera_; }
    utils::Entity camera_entity() const { return camera_entity_; }
    uint32_t width() const { return config_.width; }
    uint32_t height() const { return config_.height; }
    const RendererConfig& config() const { return config_; }

    // Batch CPU-readback primitives referenced by SceneBridge::render_batch_rgb.
    // Not used on the zero-copy path; provided so the vendored source links.
    bool render_readback_async(uint8_t* out_rgba);
    void finish();

    // --- zero-copy path ---
    // Render the current scene into the exportable swapchain, copy the pixels
    // into the persistent CUDA-imported buffer, and return its device pointer.
    // The buffer is (height, width, 4) uint8, row-major. Owned by the renderer.
    void* render_to_cuda();
    int cuda_device() const;

    // --- batched zero-copy path ---
    // begin_batch() resets the swapchain to image 0; call render_frame_no_sync()
    // once per world (after syncing that world's transforms), then
    // finish_batch_to_cuda(n) does ONE GPU sync and copies all n images into a
    // single (n, height, width, 4) uint8 CUDA buffer, returning its device ptr.
    void begin_batch();
    void render_frame_no_sync();
    void flush_wait();   // engine flushAndWait (used to bound frames-in-flight)
    void* finish_batch_to_cuda(uint32_t n);
    uint32_t batch_size() const { return config_.batch_size; }

    // --- profiling (ns accumulators; reset_profile() zeros them) ---
    void reset_profile();
    double prof_render_ms() const;   // beginFrame+render+endFrame
    double prof_flush_ms() const;    // per-frame engine flushAndWait
    double prof_copy_ms() const;     // image->buffer cmd record+submit+wait
    int prof_frames() const;

private:
    void setup_view();
    void setup_color_grading();

    RendererConfig config_;
    bool initialized_ = false;

    filament::Engine* engine_ = nullptr;
    filament::Renderer* renderer_ = nullptr;
    filament::Scene* scene_ = nullptr;
    filament::View* view_ = nullptr;
    filament::Camera* camera_ = nullptr;
    utils::Entity camera_entity_;
    filament::SwapChain* swapchain_ = nullptr;
    filament::ColorGrading* color_grading_ = nullptr;

    struct VulkanState;            // PIMPL: Vulkan + bluevk + CUDA
    std::unique_ptr<VulkanState> vk_;
};

}  // namespace vf_mujoco
