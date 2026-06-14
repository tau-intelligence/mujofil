// mujofil-warp OpenGL (EGL) renderer: the TRUE single-sync zero-copy path,
// HEADLESS (no X server required).
//
// Filament's GL backend uses an in-order command queue with NO 2-frame-in-flight
// cap (unlike Vulkan), so we can render N worlds into N distinct GL textures and
// issue a SINGLE flushAndWait — exactly the batching win the old fast OpenGL
// render_batch_rgb had — while keeping pixels on the GPU via GL<->CUDA interop.
//
// Pipeline: create a SURFACELESS EGL desktop-GL context we own -> share it with
// Filament (built with FILAMENT_SUPPORTS_EGL_ON_LINUX -> PlatformEGLHeadless) ->
// create N GL textures and Texture::import each as a RenderTarget color
// attachment -> render world i into texture i (no flush) -> ONE flushAndWait ->
// cudaGraphicsGLRegister + map + memcpy2DFromArray each texture into a single
// (N,H,W,4) CUDA buffer -> DLPack -> torch.cuda. No CPU round-trip, no display.
//
// Our EGL context mirrors Filament's PlatformEGL display selection (default
// display first, EGL_PLATFORM_DEVICE_EXT fallback for true headless) and binds
// EGL_OPENGL_API so it shares object namespace with Filament's context.
//
// Implements the SAME vf_mujoco::Renderer interface as renderer_warp.cpp; compiled
// INSTEAD of it (mutually exclusive), so Renderer::VulkanState here holds GL state.
#include "core/renderer.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>

#include <cstring>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <filament/Renderer.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <filament/Options.h>
#include <filament/Texture.h>
#include <filament/RenderTarget.h>
#include <utils/EntityManager.h>

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace filament;

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif

namespace vf_mujoco {

namespace {
using clk = std::chrono::high_resolution_clock;
inline long long ns(clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

// Open an EGL display the same way Filament's PlatformEGL::createDriver does:
// the default display first, then the EGL_PLATFORM_DEVICE_EXT NVIDIA device
// (true headless, no X). Returns an INITIALIZED display or EGL_NO_DISPLAY.
EGLDisplay openEglDisplay() {
    EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;
    if (d != EGL_NO_DISPLAY && eglInitialize(d, &major, &minor)) return d;

    auto eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) return EGL_NO_DISPLAY;
    EGLDeviceEXT devs[16]; EGLint nd = 0;
    if (!eglQueryDevicesEXT(16, devs, &nd)) return EGL_NO_DISPLAY;
    for (EGLint i = 0; i < nd; ++i) {
        EGLDisplay c = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devs[i], nullptr);
        if (c != EGL_NO_DISPLAY && eglInitialize(c, &major, &minor)) {
            const char* vendor = eglQueryString(c, EGL_VENDOR);
            if (vendor && std::strstr(vendor, "NVIDIA")) return c;
            d = c;  // remember a working non-NVIDIA display as last resort
        }
    }
    return d;
}
}  // namespace

// PIMPL holds all EGL/GL/CUDA state (kept out of the header so the vendored
// SceneBridge TU never sees GL/CUDA).
struct Renderer::VulkanState {
    EGLDisplay edpy = EGL_NO_DISPLAY;
    EGLContext ectx = EGL_NO_CONTEXT;
    EGLSurface esurf = EGL_NO_SURFACE;

    uint32_t n = 1;                       // # batch slots (= max(batch_size,1))
    std::vector<GLuint> glTex;            // N color textures we own
    std::vector<Texture*> filColor;       // N imported Filament textures
    std::vector<RenderTarget*> rt;        // N render targets
    Texture* depth = nullptr;             // shared depth (renders are sequential)
    uint32_t cur = 0;                     // next slot to render into

    std::vector<cudaGraphicsResource*> cudaRes;  // registered once per texture
    uint8_t* cudaBuf = nullptr;           // device linear (N,H,W,4)
    size_t cudaBytes = 0;
    int cudaDevice = 0;
    bool frameOpen = false;               // a beginFrame is in progress
    bool perFrame = false;                // per-frame beginFrame/endFrame model

    long long t_render = 0, t_flush = 0, t_copy = 0;
    int n_frames = 0;
};

Renderer::Renderer(const RendererConfig& config)
    : config_(config), vk_(std::make_unique<VulkanState>()) {}

Renderer::~Renderer() { destroy(); }

bool Renderer::initialize() {
    if (initialized_) return true;
    auto& v = *vk_;
    v.n = config_.batch_size > 0 ? config_.batch_size : 1;
    if (const char* e = std::getenv("MUJOFIL_WARP_GL_PERFRAME")) v.perFrame = atoi(e) != 0;
    const uint32_t W = config_.width, H = config_.height;

    // --- 1. our surfaceless EGL desktop-GL context (no X; headless-capable) -
    // Bind desktop GL (EGL_OPENGL_API) to match Filament's PlatformEGLHeadless,
    // so our context shares its object namespace with Filament's.
    const bool dbg = std::getenv("MUJOFIL_WARP_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "[gl] opening EGL display...\n");
    v.edpy = openEglDisplay();
    if (v.edpy == EGL_NO_DISPLAY)
        throw std::runtime_error("GL renderer: no usable EGL display (need NVIDIA EGL)");
    if (dbg) fprintf(stderr, "[gl] EGL display=%p vendor=%s\n", (void*)v.edpy, eglQueryString(v.edpy, EGL_VENDOR));
    if (!eglBindAPI(EGL_OPENGL_API))
        throw std::runtime_error("GL renderer: eglBindAPI(EGL_OPENGL_API) failed");
    const EGLint cfgAttr[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    EGLConfig cfg; EGLint ncfg = 0;
    if (!eglChooseConfig(v.edpy, cfgAttr, &cfg, 1, &ncfg) || ncfg == 0)
        throw std::runtime_error("GL renderer: eglChooseConfig failed");
    const EGLint ctxAttr[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE
    };
    // Create the context with EGL_NO_CONFIG_KHR when available so it is
    // compatible with Filament's own EGL_NO_CONFIG contexts (it creates shared
    // contexts for its shader-compiler thread pool; a config-bearing context
    // here would make those shares fail with EGL_BAD_MATCH, breaking shader
    // linking). The pbuffer SURFACE below still uses a concrete config.
    EGLConfig ctxCfg = cfg;
    {
        const char* exts = eglQueryString(v.edpy, EGL_EXTENSIONS);
        if (exts && std::strstr(exts, "EGL_KHR_no_config_context"))
            ctxCfg = (EGLConfig)EGL_NO_CONFIG_KHR;
    }
    v.ectx = eglCreateContext(v.edpy, ctxCfg, EGL_NO_CONTEXT, ctxAttr);
    if (v.ectx == EGL_NO_CONTEXT)
        throw std::runtime_error("GL renderer: eglCreateContext failed");
    // A tiny pbuffer surface (more portable than relying on surfaceless support).
    const EGLint pbAttr[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    v.esurf = eglCreatePbufferSurface(v.edpy, cfg, pbAttr);
    if (!eglMakeCurrent(v.edpy, v.esurf, v.esurf, v.ectx))
        throw std::runtime_error("GL renderer: eglMakeCurrent failed");
    if (dbg) fprintf(stderr, "[gl] ctx current; GL_VERSION=%s\n", (const char*)glGetString(GL_VERSION));

    // --- 2. N GL textures we own (Filament renders into them; CUDA reads) ----
    if (dbg) fprintf(stderr, "[gl] creating %u textures %ux%u...\n", v.n, W, H);
    v.glTex.resize(v.n);
    glGenTextures(v.n, v.glTex.data());
    for (uint32_t i = 0; i < v.n; ++i) {
        glBindTexture(GL_TEXTURE_2D, v.glTex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();

    // --- 3. Filament engine sharing OUR EGL context -------------------------
    if (dbg) fprintf(stderr, "[gl] building Filament engine (shared ctx=%p)...\n", (void*)v.ectx);
    // Force SYNCHRONOUS shader compilation. Filament's parallel shader-compiler
    // threads don't bind the EGL desktop-GL API per thread (it's per-thread
    // state), so on the EGL desktop-GL path they fail to create shared contexts
    // (EGL_BAD_MATCH) and shader linking breaks. Synchronous compile runs on the
    // main driver thread (which has the API bound) and is a one-time startup cost.
    Engine::Config ecfg{};
    ecfg.disableParallelShaderCompile = true;
    engine_ = Engine::Builder()
        .backend(Engine::Backend::OPENGL)
        .sharedContext((void*)v.ectx)
        .config(&ecfg)
        .build();
    if (!engine_) throw std::runtime_error("GL renderer: Filament engine build failed");
    if (dbg) fprintf(stderr, "[gl] engine built OK\n");

    renderer_ = engine_->createRenderer();
    scene_ = engine_->createScene();
    camera_entity_ = utils::EntityManager::get().create();
    camera_ = engine_->createCamera(camera_entity_);
    camera_->setProjection(70.0, double(W) / double(H), 0.05, 200.0);
    camera_->lookAt({3, 3, 3}, {0, 0, 0}, {0, 0, 1});

    // Shared depth (sequential, in-order renders) + N imported color RTs.
    v.depth = Texture::Builder().width(W).height(H).levels(1)
        .format(Texture::InternalFormat::DEPTH24)
        .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine_);
    v.filColor.resize(v.n);
    v.rt.resize(v.n);
    for (uint32_t i = 0; i < v.n; ++i) {
        v.filColor[i] = Texture::Builder().width(W).height(H).levels(1)
            .format(Texture::InternalFormat::RGBA8)
            .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
            .import((intptr_t)v.glTex[i]).build(*engine_);
        v.rt[i] = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, v.filColor[i])
            .texture(RenderTarget::AttachmentPoint::DEPTH, v.depth)
            .build(*engine_);
    }

    // Swapchain is still required for beginFrame() even with an offscreen RT.
    swapchain_ = engine_->createSwapChain(W, H, SwapChain::CONFIG_READABLE);

    setup_view();
    setup_color_grading();
    view_->setRenderTarget(v.rt[0]);

    // --- 4. CUDA: register each GL texture once; allocate the linear buffer ---
    unsigned int cudaCount = 0;
    int cudaDev = 0;
    if (cudaGLGetDevices(&cudaCount, &cudaDev, 1, cudaGLDeviceListAll) == cudaSuccess && cudaCount > 0)
        v.cudaDevice = cudaDev;
    cudaSetDevice(v.cudaDevice);
    v.cudaRes.resize(v.n, nullptr);
    for (uint32_t i = 0; i < v.n; ++i) {
        cudaError_t e = cudaGraphicsGLRegisterImage(
            &v.cudaRes[i], v.glTex[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("GL renderer: cudaGraphicsGLRegisterImage: ")
                                     + cudaGetErrorString(e));
    }
    v.cudaBytes = size_t(v.n) * W * H * 4;
    if (cudaMalloc(&v.cudaBuf, v.cudaBytes) != cudaSuccess)
        throw std::runtime_error("GL renderer: cudaMalloc failed");

    initialized_ = true;
    return true;
}

void Renderer::setup_view() {
    view_ = engine_->createView();
    view_->setScene(scene_);
    view_->setCamera(camera_);
    view_->setViewport({0, 0, config_.width, config_.height});

    filament::Renderer::ClearOptions co;
    co.clearColor = {0.35f, 0.42f, 0.55f, 1.0f};
    co.clear = true;
    renderer_->setClearOptions(co);

    view_->setAntiAliasing(config_.enable_fxaa
        ? View::AntiAliasing::FXAA : View::AntiAliasing::NONE);
    if (config_.enable_msaa)
        view_->setMultiSampleAntiAliasingOptions({.enabled = true, .sampleCount = config_.msaa_samples});
    if (config_.enable_ssao) {
        View::AmbientOcclusionOptions ao;
        ao.radius = 0.3f; ao.power = 1.0f; ao.intensity = 1.0f;
        switch (config_.ssao_quality) {
            case 0:  ao.quality = View::QualityLevel::LOW; break;
            case 1:  ao.quality = View::QualityLevel::MEDIUM; break;
            case 2:  ao.quality = View::QualityLevel::HIGH; break;
            default: ao.quality = View::QualityLevel::ULTRA; break;
        }
        ao.enabled = true; ao.ssct.enabled = config_.ssao_ssct;
        view_->setAmbientOcclusionOptions(ao);
    }
    if (config_.enable_bloom)
        view_->setBloomOptions({.strength = 0.1f, .enabled = true});
    view_->setShadowingEnabled(config_.enable_shadows);
    view_->setDithering(config_.dithering ? View::Dithering::TEMPORAL : View::Dithering::NONE);
}

void Renderer::setup_color_grading() {
    auto b = ColorGrading::Builder();
    b.toneMapping(config_.tone_mapping
        ? ColorGrading::ToneMapping::FILMIC : ColorGrading::ToneMapping::LINEAR);
    b.exposure(config_.exposure);
    color_grading_ = b.build(*engine_);
    view_->setColorGrading(color_grading_);
}

void Renderer::begin_batch() { vk_->cur = 0; vk_->frameOpen = false; }

void Renderer::render_frame_no_sync() {
    auto& v = *vk_;
    const uint32_t slot = v.cur % v.n;
    auto t0 = clk::now();
    view_->setRenderTarget(v.rt[slot]);
    if (v.perFrame) {
        // Per-frame model: one beginFrame/endFrame per world (matches the old
        // fast render_batch_rgb). On the GL backend this can pipeline better than
        // a single mega-frame; ONE flushAndWait still syncs the whole batch.
        renderer_->beginFrame(swapchain_);
        renderer_->render(view_);
        renderer_->endFrame();
    } else {
        // Single-frame multi-view: ONE beginFrame brackets N render() calls, each
        // into a distinct imported RenderTarget. Avoids the frames-in-flight cap
        // that would gate N beginFrame/endFrame pairs without a sync.
        if (!v.frameOpen) {
            renderer_->beginFrame(swapchain_);
            v.frameOpen = true;
        }
        renderer_->render(view_);
    }
    auto t1 = clk::now();
    v.t_render += ns(t0, t1);
    v.cur += 1;
    v.n_frames += 1;
}

void Renderer::flush_wait() {
    auto& v = *vk_;
    auto t0 = clk::now();
    if (v.frameOpen) {
        renderer_->endFrame();
        v.frameOpen = false;
    }
    engine_->flushAndWait();
    v.t_flush += ns(t0, clk::now());
}

void* Renderer::finish_batch_to_cuda(uint32_t n) {
    auto& v = *vk_;
    if (n == 0) return v.cudaBuf;
    if (n > v.n) n = v.n;
    const uint32_t W = config_.width, H = config_.height;
    const size_t rowBytes = size_t(W) * 4;
    const size_t slice = rowBytes * H;
    auto t0 = clk::now();
    eglMakeCurrent(v.edpy, v.esurf, v.esurf, v.ectx);
    cudaGraphicsMapResources(n, v.cudaRes.data(), 0);
    for (uint32_t i = 0; i < n; ++i) {
        cudaArray_t arr = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&arr, v.cudaRes[i], 0, 0);
        cudaMemcpy2DFromArray(v.cudaBuf + size_t(i) * slice, rowBytes,
                              arr, 0, 0, rowBytes, H, cudaMemcpyDeviceToDevice);
    }
    cudaGraphicsUnmapResources(n, v.cudaRes.data(), 0);
    cudaDeviceSynchronize();
    v.t_copy += ns(t0, clk::now());
    return v.cudaBuf;
}

void* Renderer::render_to_cuda() {
    begin_batch();
    render_frame_no_sync();
    flush_wait();
    return finish_batch_to_cuda(1);
}

int Renderer::cuda_device() const { return vk_->cudaDevice; }

// CPU-readback primitives are only referenced by SceneBridge::render_batch_rgb,
// which the GL warp path never calls. Provide trivial impls so it links.
bool Renderer::render_readback_async(uint8_t*) { return false; }
void Renderer::finish() { if (engine_) engine_->flushAndWait(); }

void Renderer::reset_profile() {
    vk_->t_render = vk_->t_flush = vk_->t_copy = 0;
    vk_->n_frames = 0;
}
double Renderer::prof_render_ms() const { return vk_->t_render / 1e6; }
double Renderer::prof_flush_ms() const { return vk_->t_flush / 1e6; }
double Renderer::prof_copy_ms() const { return vk_->t_copy / 1e6; }
int Renderer::prof_frames() const { return vk_->n_frames; }

void Renderer::destroy() {
    if (!vk_) return;
    auto& v = *vk_;
    for (auto* r : v.cudaRes) if (r) cudaGraphicsUnregisterResource(r);
    v.cudaRes.clear();
    if (v.cudaBuf) { cudaFree(v.cudaBuf); v.cudaBuf = nullptr; }
    if (engine_) {
        for (auto* rt : v.rt) if (rt) engine_->destroy(rt);
        for (auto* t : v.filColor) if (t) engine_->destroy(t);
        if (v.depth) engine_->destroy(v.depth);
        if (color_grading_) engine_->destroy(color_grading_);
        if (view_) engine_->destroy(view_);
        if (scene_) engine_->destroy(scene_);
        if (renderer_) engine_->destroy(renderer_);
        if (swapchain_) engine_->destroy(swapchain_);
        if (camera_entity_) engine_->destroyCameraComponent(camera_entity_);
        Engine::destroy(&engine_);
        engine_ = nullptr;
    }
    v.rt.clear(); v.filColor.clear(); v.depth = nullptr;
    if (!v.glTex.empty()) { glDeleteTextures(v.glTex.size(), v.glTex.data()); v.glTex.clear(); }
    if (v.ectx != EGL_NO_CONTEXT) {
        eglMakeCurrent(v.edpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (v.esurf != EGL_NO_SURFACE) eglDestroySurface(v.edpy, v.esurf);
        eglDestroyContext(v.edpy, v.ectx);
        v.ectx = EGL_NO_CONTEXT;
    }
    if (v.edpy != EGL_NO_DISPLAY) { eglTerminate(v.edpy); v.edpy = EGL_NO_DISPLAY; }
    initialized_ = false;
}

}  // namespace vf_mujoco
