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

namespace filament { class RenderTarget; }

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
    bool enable_ssr = true;     // screen-space reflections (glossy surfaces)
    float exposure = 0.0f;     // Exposure compensation (EV); 0 = neutral
    bool tone_mapping = true;
    bool dithering = true;
    uint32_t batch_size = 1;   // # of swapchain images for batched rendering
    // Parallel layered batch: render all batch_size worlds in ONE instanced draw
    // into a (W,H,batch_size) array texture via the forked Filament gl_Layer
    // path (NVIDIA GL only, needs the layered-batch Filament fork). Off = the
    // existing per-world render loop.
    bool layered = false;
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

    // True when the whole batch MUST render inside ONE beginFrame/endFrame (the
    // atlas/megatexture path): all N tiles share one render target, so a
    // mid-batch endFrame+beginFrame would re-clear and lose earlier tiles. The
    // driver loop must therefore skip the periodic flush_wait() when this is set.
    bool single_sync() const { return single_sync_; }

    // --- LAYERED parallel-batch path (forked Filament gl_Layer routing) ---
    // When the renderer is built with config_.layered (env MUJOFIL_WARP_LAYERED),
    // it renders into ONE (W, H, batch_size) array-texture render target. The
    // SceneBridge builds each geom instanced batch_size times, so a SINGLE
    // render() draws all N worlds at once, each routed to its own array layer by
    // the forked vertex shader. render_layered_to_cuda() does that one render and
    // slices the N layers into the persistent (N, H, W, 4) CUDA buffer.
    bool layered() const { return layered_; }
    filament::RenderTarget* layered_render_target() const;  // for SceneBridge/View
    // Max worlds renderable in ONE instanced pass (the Filament UBO instance cap).
    // For batches larger than this, render in chunks of this size.
    uint32_t layered_max_per_pass() const;
    // PASS 1: render the shared static environment into the 2D backdrop texture +
    // copy it to the backdrop CUDA buffer. Call once per batch.
    void render_layered_backdrop();
    // PASS 2 (one chunk): render `count` (<= layered_max_per_pass) per-world
    // instanced objects into the array, then slice the `count` layers into the
    // output buffer starting at world `out_offset`. The SceneBridge must have
    // filled the InstanceBuffer with this chunk's `count` worlds first.
    void render_layered_objects(uint32_t out_offset, uint32_t count);
    // Single-pass convenience (N <= layered_max_per_pass): backdrop + one chunk.
    void* render_layered_to_cuda();
    // Device pointer to the (height, width, 4) uint8 static-backdrop buffer,
    // filled by render_layered_backdrop(). Composited under the per-world objects
    // (which carry alpha) on the Python side.
    void* layered_backdrop_ptr() const;
    void* layered_output_ptr() const;   // the persistent (N, H, W, 4) buffer

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
    bool single_sync_ = false;   // atlas/megatexture path renders all N in one frame
    bool layered_ = false;       // forked-Filament gl_Layer array-RT parallel path

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
