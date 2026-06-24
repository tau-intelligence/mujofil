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
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace filament;

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_SRGB8_ALPHA8
#define GL_SRGB8_ALPHA8 0x8C43
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_HALF_FLOAT
#define GL_HALF_FLOAT 0x140B
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

    // --- atlas/megatexture mode (MUJOFIL_WARP_ATLAS=1) ---
    // Instead of N distinct render targets (one per world), pack N tiles into a
    // SINGLE atlas texture (cols x rows grid of WxH tiles). The render target
    // never changes between tiles — only the viewport — so Filament keeps its
    // aux buffers (SSAO/MSAA/post) configured once. glTex/filColor/rt/cudaRes
    // hold exactly ONE element (the atlas) in this mode.
    bool atlas = false;
    uint32_t cols = 1, rows = 1;          // tile grid
    uint32_t atlasW = 0, atlasH = 0;      // cols*W, rows*H

    // --- LAYERED parallel-batch mode (config.layered, MUJOFIL_WARP_LAYERED) ---
    // ONE (W,H,N) GL array texture, imported into Filament, attached as a LAYERED
    // render target. SceneBridge instances each geom N times so a single render()
    // draws all N worlds, the forked vertex shader routing world w -> array layer
    // w via gl_Layer. The array is registered with CUDA once; each layer slices
    // into the (N,H,W,4) buffer.
    bool layered = false;
    GLuint arrayTex = 0;                  // our GL_TEXTURE_2D_ARRAY (W,H,L) RGBA8
    Texture* arrayColor = nullptr;        // Filament import of arrayTex
    Texture* arrayDepth = nullptr;        // Filament-created (W,H,L) depth array
    RenderTarget* arrayRT = nullptr;      // layered RT (whole array)
    cudaGraphicsResource* arrayCudaRes = nullptr;
    uint32_t layers = 1;                  // array layers = min(n, MAX_PER_PASS)
    // Static-backdrop: a SEPARATE (W,H) 2D texture (depth==1 -> always single-layer
    // attach, never the layered path) for the shared environment, rendered once
    // then broadcast (glCopyImageSubData) into every array layer.
    GLuint backdropTex = 0;
    Texture* backdropColor = nullptr;
    Texture* backdropDepth = nullptr;
    RenderTarget* backdropRT = nullptr;
    cudaGraphicsResource* backdropCudaRes = nullptr;  // registered once
    uint8_t* backdropBuf = nullptr;                   // device (H,W,4), composited in torch

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

    // --- 1b. decide atlas/megatexture mode --------------------------------
    // Pack the N WxH tiles into ONE cols x rows atlas texture so the render
    // target never changes per tile (only the viewport). Needs the GL context
    // current to honor GL_MAX_TEXTURE_SIZE; falls back to multi-RT if the atlas
    // would exceed it. n==1 and forced perFrame keep the simple per-RT path.
    if (const char* e = std::getenv("MUJOFIL_WARP_ATLAS"))
        v.atlas = (atoi(e) != 0) && v.n > 1 && !v.perFrame;
    if (v.atlas) {
        v.cols = (uint32_t)std::ceil(std::sqrt((double)v.n));
        v.rows = (v.n + v.cols - 1) / v.cols;
        v.atlasW = v.cols * W;
        v.atlasH = v.rows * H;
        GLint maxTex = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
        if (maxTex > 0 && (v.atlasW > (uint32_t)maxTex || v.atlasH > (uint32_t)maxTex)) {
            if (dbg) fprintf(stderr, "[gl] atlas %ux%u exceeds GL_MAX_TEXTURE_SIZE %d; using multi-RT\n",
                             v.atlasW, v.atlasH, maxTex);
            v.atlas = false;
        }
    }
    single_sync_ = v.atlas;  // atlas renders all N inside one beginFrame/endFrame

    // --- 1c. decide LAYERED parallel-batch mode ---------------------------
    // config.layered (or env MUJOFIL_WARP_LAYERED) renders all N worlds in ONE
    // instanced draw into a (W,H,N) array texture via the forked gl_Layer path.
    // Mutually exclusive with atlas. Requires the layered Filament fork + the
    // -g (gl_Layer) material set; the caller points VF_MUJOCO_MATERIALS_DIR at it.
    v.layered = config_.layered || (std::getenv("MUJOFIL_WARP_LAYERED") &&
                                    atoi(std::getenv("MUJOFIL_WARP_LAYERED")) != 0);
    if (v.layered) { v.atlas = false; single_sync_ = true; layered_ = true; }
    // Enable the forked layered-batch FBO attach NOW, before the array render
    // target is ever created/configured. The backend's framebufferTexture reads
    // this env to decide LAYERED (whole array) vs SINGLE-LAYER attach, and it
    // CACHES that decision per render target. It runs deferred on the driver
    // thread, so setting the env only inside the per-pass render() (as before)
    // races: a heavy GLB backdrop can trigger the array FBO to be configured
    // early -- before the first render's setenv -- caching a SINGLE-LAYER attach
    // that collapses every world into layer 0 (observed on neon_street/scifi).
    // setenv is process-global + permanent, and a depth-1 backdrop target still
    // attaches single-layer regardless, so enabling it here unconditionally for
    // a layered renderer is safe and removes the timing dependency.
    if (v.layered) setenv("FILAMENT_LAYERED_BATCH", "1", 1);
    // The array texture / instanced draw is capped at MAX_PER_PASS worlds (the
    // Filament UBO instance limit). Larger batches render in chunks of this size
    // into the SAME array, each chunk copied to its slice of the (N,H,W,4) output.
    v.layers = v.layered ? std::min(v.n, layered_max_per_pass()) : v.n;

    // --- 2. GL color texture(s) we own (Filament renders into them; CUDA reads).
    // Layered mode: ONE (W,H,N) GL_TEXTURE_2D_ARRAY. Atlas: ONE (cols*W)x(rows*H)
    // 2D texture. Multi-RT: N WxH 2D textures.
    const uint32_t nTex = (v.atlas || v.layered) ? 1u : v.n;
    const uint32_t texW = v.atlas ? v.atlasW : W;
    const uint32_t texH = v.atlas ? v.atlasH : H;
    if (v.layered) {
        if (dbg) fprintf(stderr, "[gl] creating (W,H,L)=(%u,%u,%u) array tex (n=%u, chunked if >L)...\n",
                         W, H, v.layers, v.n);
        glGenTextures(1, &v.arrayTex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, v.arrayTex);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        // FLOAT16 HDR: the objects pass runs post-processing OFF and writes LINEAR
        // shaded colour. A half-float RT stores it WITHOUT 8-bit quantization, so
        // the Python auto-exposure + FILMIC tone map can lift dark indoor regions
        // with no posterisation/banding (an 8-bit linear or sRGB RT bands under the
        // exposure gain). 8 bytes/texel; Python reads it back as float16.
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA16F, W, H, v.layers, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        // Backdrop scratch: a single (W,H) 2D texture for the shared environment.
        glGenTextures(1, &v.backdropTex);
        glBindTexture(GL_TEXTURE_2D, v.backdropTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
    } else {
        if (dbg) fprintf(stderr, "[gl] creating %u texture(s) %ux%u (atlas=%d grid %ux%u)...\n",
                         nTex, texW, texH, (int)v.atlas, v.cols, v.rows);
        v.glTex.resize(nTex);
        glGenTextures(nTex, v.glTex.data());
        for (uint32_t i = 0; i < nTex; ++i) {
            glBindTexture(GL_TEXTURE_2D, v.glTex[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texW, texH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
    }

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

    // Depth + imported color render target(s). Layered: ONE (W,H,N) array color
    // (imported) + (W,H,N) array depth + ONE layered RT (whole array). Atlas: ONE
    // (atlasW x atlasH) depth + ONE RT. Multi-RT: ONE shared WxH depth + N RTs.
    if (v.layered) {
        v.arrayColor = Texture::Builder().width(W).height(H).depth(v.layers).levels(1)
            .sampler(Texture::Sampler::SAMPLER_2D_ARRAY)
            .format(Texture::InternalFormat::RGBA16F)
            .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
            .import((intptr_t)v.arrayTex).build(*engine_);
        v.arrayDepth = Texture::Builder().width(W).height(H).depth(v.layers).levels(1)
            .sampler(Texture::Sampler::SAMPLER_2D_ARRAY)
            .format(Texture::InternalFormat::DEPTH32F)
            .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine_);
        v.arrayRT = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, v.arrayColor)
            .texture(RenderTarget::AttachmentPoint::DEPTH, v.arrayDepth)
            .build(*engine_);
        // Backdrop RT: a single (W,H) 2D color+depth target for the shared
        // environment pass. depth==1 so the GL backend always does a plain
        // single-layer attach (the layered path needs depth>1).
        v.backdropColor = Texture::Builder().width(W).height(H).levels(1)
            .sampler(Texture::Sampler::SAMPLER_2D)
            .format(Texture::InternalFormat::RGBA8)
            .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
            .import((intptr_t)v.backdropTex).build(*engine_);
        v.backdropDepth = Texture::Builder().width(W).height(H).levels(1)
            .sampler(Texture::Sampler::SAMPLER_2D)
            .format(Texture::InternalFormat::DEPTH32F)
            .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine_);
        v.backdropRT = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, v.backdropColor)
            .texture(RenderTarget::AttachmentPoint::DEPTH, v.backdropDepth)
            .build(*engine_);
    } else {
        const uint32_t dW = v.atlas ? v.atlasW : W;
        const uint32_t dH = v.atlas ? v.atlasH : H;
        v.depth = Texture::Builder().width(dW).height(dH).levels(1)
            .format(Texture::InternalFormat::DEPTH24)
            .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine_);
        v.filColor.resize(nTex);
        v.rt.resize(nTex);
        for (uint32_t i = 0; i < nTex; ++i) {
            v.filColor[i] = Texture::Builder().width(texW).height(texH).levels(1)
                .format(Texture::InternalFormat::RGBA8)
                .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
                .import((intptr_t)v.glTex[i]).build(*engine_);
            v.rt[i] = RenderTarget::Builder()
                .texture(RenderTarget::AttachmentPoint::COLOR0, v.filColor[i])
                .texture(RenderTarget::AttachmentPoint::DEPTH, v.depth)
                .build(*engine_);
        }
    }

    // Swapchain is still required for beginFrame() even with an offscreen RT.
    swapchain_ = engine_->createSwapChain(W, H, SwapChain::CONFIG_READABLE);

    setup_view();
    setup_color_grading();
    view_->setRenderTarget(v.layered ? v.arrayRT : v.rt[0]);

    // --- 4. CUDA: register the GL texture(s) once; allocate the linear buffer ---
    unsigned int cudaCount = 0;
    int cudaDev = 0;
    if (cudaGLGetDevices(&cudaCount, &cudaDev, 1, cudaGLDeviceListAll) == cudaSuccess && cudaCount > 0)
        v.cudaDevice = cudaDev;
    cudaSetDevice(v.cudaDevice);
    if (v.layered) {
        cudaError_t e = cudaGraphicsGLRegisterImage(
            &v.arrayCudaRes, v.arrayTex, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsReadOnly);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("GL renderer: cudaGraphicsGLRegisterImage(array): ")
                                     + cudaGetErrorString(e));
        // Backdrop 2D texture -> its own CUDA buffer (composited under objects).
        e = cudaGraphicsGLRegisterImage(
            &v.backdropCudaRes, v.backdropTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("GL renderer: cudaGraphicsGLRegisterImage(backdrop): ")
                                     + cudaGetErrorString(e));
        if (cudaMalloc(&v.backdropBuf, size_t(W) * H * 4) != cudaSuccess)
            throw std::runtime_error("GL renderer: cudaMalloc(backdrop) failed");
    } else {
        v.cudaRes.resize(nTex, nullptr);
        for (uint32_t i = 0; i < nTex; ++i) {
            cudaError_t e = cudaGraphicsGLRegisterImage(
                &v.cudaRes[i], v.glTex[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("GL renderer: cudaGraphicsGLRegisterImage: ")
                                         + cudaGetErrorString(e));
        }
    }
    // Layered objects buffer is RGBA16F (8 bytes/texel); all other paths are
    // RGBA8 (4 bytes/texel).
    const size_t bppOut = v.layered ? 8u : 4u;
    v.cudaBytes = size_t(v.n) * W * H * bppOut;
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
        // Params matched to mujofil's tuned SSAO: a moderate radius that grounds
        // objects, with minHorizonAngleRad rejecting coplanar floor/wall samples
        // (avoids the broad "gloom" band a wide radius paints at frame edges).
        ao.radius = 0.30f; ao.bias = 0.0005f; ao.power = 1.0f;
        ao.intensity = 1.5f; ao.bilateralThreshold = 0.05f;
        ao.minHorizonAngleRad = 0.10f;
        switch (config_.ssao_quality) {
            case 0:  ao.quality = View::QualityLevel::LOW; break;
            case 1:  ao.quality = View::QualityLevel::MEDIUM; break;
            case 2:  ao.quality = View::QualityLevel::HIGH; break;
            default: ao.quality = View::QualityLevel::ULTRA; break;
        }
        ao.lowPassFilter = View::QualityLevel::HIGH;
        ao.upsampling = View::QualityLevel::HIGH;
        ao.enabled = true; ao.ssct.enabled = config_.ssao_ssct;
        view_->setAmbientOcclusionOptions(ao);
    }
    if (config_.enable_bloom)
        view_->setBloomOptions({.strength = 0.1f, .enabled = true});
    if (config_.enable_ssr) {
        filament::ScreenSpaceReflectionsOptions ssr;
        ssr.enabled = true; ssr.thickness = 0.1f; ssr.bias = 0.01f;
        ssr.maxDistance = 6.0f; ssr.stride = 2.0f;
        view_->setScreenSpaceReflectionsOptions(ssr);
    }
    view_->setShadowingEnabled(config_.enable_shadows);
    view_->setDithering(config_.dithering ? View::Dithering::TEMPORAL : View::Dithering::NONE);

    // LAYERED parallel-batch: the scene MUST render directly into our array
    // render target so the vertex shader's gl_Layer routes each world to its own
    // layer. With post-processing enabled Filament renders into its OWN
    // single-layer intermediate and blits the result (collapsing all worlds into
    // layer 0). So disable post-processing in layered mode (PBR + IBL still run
    // in the material; only screen-space post/tone-map is skipped — consistent
    // with the effects-off fast RL path). Also forces no SSAO/SSR/bloom passes.
    if (layered_) {
        view_->setPostProcessingEnabled(false);
        view_->setShadowingEnabled(false);
    }
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
    if (v.atlas) {
        // Atlas: ONE render target (set once at init), ONE beginFrame brackets
        // all N renders. Each tile renders into its grid cell via the viewport;
        // Filament's per-frame clear fills the whole atlas once, then each
        // viewport's render writes its cell. NEVER flush mid-batch (would
        // re-clear and erase earlier tiles) -> single_sync() gates the loop.
        const uint32_t col = slot % v.cols, row = slot / v.cols;
        view_->setViewport({(int32_t)(col * config_.width), (int32_t)(row * config_.height),
                            config_.width, config_.height});
        if (!v.frameOpen) {
            renderer_->beginFrame(swapchain_);
            v.frameOpen = true;
        }
        renderer_->render(view_);
    } else {
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
    if (v.atlas) {
        // ONE atlas image: extract each tile's WxH region into the contiguous
        // (N,H,W,4) buffer via offset copies (wOffset in bytes, hOffset in rows).
        cudaGraphicsMapResources(1, &v.cudaRes[0], 0);
        cudaArray_t arr = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&arr, v.cudaRes[0], 0, 0);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t col = i % v.cols, row = i / v.cols;
            cudaMemcpy2DFromArray(v.cudaBuf + size_t(i) * slice, rowBytes,
                                  arr, size_t(col) * W * 4, size_t(row) * H,
                                  rowBytes, H, cudaMemcpyDeviceToDevice);
        }
        cudaGraphicsUnmapResources(1, &v.cudaRes[0], 0);
    } else {
        cudaGraphicsMapResources(n, v.cudaRes.data(), 0);
        for (uint32_t i = 0; i < n; ++i) {
            cudaArray_t arr = nullptr;
            cudaGraphicsSubResourceGetMappedArray(&arr, v.cudaRes[i], 0, 0);
            cudaMemcpy2DFromArray(v.cudaBuf + size_t(i) * slice, rowBytes,
                                  arr, 0, 0, rowBytes, H, cudaMemcpyDeviceToDevice);
        }
        cudaGraphicsUnmapResources(n, v.cudaRes.data(), 0);
    }
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

filament::RenderTarget* Renderer::layered_render_target() const {
    return vk_->arrayRT;
}

uint32_t Renderer::layered_max_per_pass() const { return 256u; }

void Renderer::render_layered_backdrop() {
    // PASS 1: render the shared static environment (visibility layer bit 0) ONCE
    // into the (W,H) 2D backdrop texture; copy it to the backdrop CUDA buffer.
    // The 2D texture (depth 1) always does a plain single-layer attach, so it is
    // never confused with the layered array attach.
    auto& v = *vk_;
    const uint32_t W = config_.width, H = config_.height;
    setenv("FILAMENT_LAYERED_BATCH", "1", 1);
    auto t0 = clk::now();
    // The backdrop renders into a single-layer 2D texture, so gl_Layer routing is
    // irrelevant here and we can run the FULL post-processing pipeline. This is
    // REQUIRED for screen-space refraction: gltfio scenes using
    // KHR_materials_transmission (e.g. glass in the chess/gallery GLBs) allocate a
    // refraction pass that only exists when post-processing is enabled; with it
    // off Filament hits an invalid handle ("corrupted heap Handle") and aborts.
    // Post-processing is re-disabled for the object array pass (render_layered_
    // objects) where it WOULD collapse the gl_Layer routing.
    view_->setPostProcessingEnabled(true);
    view_->setVisibleLayers(0xFF, 0x01);
    view_->setRenderTarget(v.backdropRT);
    view_->setViewport({0, 0, W, H});
    {
        filament::Renderer::ClearOptions co;
        co.clearColor = {0.05f, 0.06f, 0.09f, 1.0f};
        co.clear = true;
        renderer_->setClearOptions(co);
    }
    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_);
        renderer_->endFrame();
    }
    engine_->flushAndWait();
    v.t_render += ns(t0, clk::now());

    const size_t rowBytes = size_t(W) * 4;
    eglMakeCurrent(v.edpy, v.esurf, v.esurf, v.ectx);
    cudaGraphicsMapResources(1, &v.backdropCudaRes, 0);
    cudaArray_t arr = nullptr;
    cudaGraphicsSubResourceGetMappedArray(&arr, v.backdropCudaRes, 0, 0);
    cudaMemcpy2DFromArray(v.backdropBuf, rowBytes, arr, 0, 0, rowBytes, H,
                          cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &v.backdropCudaRes, 0);
}

void Renderer::render_layered_objects(uint32_t out_offset, uint32_t count) {
    // PASS 2 (one chunk): the SceneBridge has filled the InstanceBuffer with this
    // chunk's `count` worlds. ONE instanced draw routes world w -> array layer w
    // via gl_Layer, on a TRANSPARENT background (alpha 0). Slice the `count`
    // layers into the output buffer starting at world `out_offset`.
    auto& v = *vk_;
    const uint32_t W = config_.width, H = config_.height;
    if (count > v.layers) count = v.layers;
    setenv("FILAMENT_LAYERED_BATCH", "1", 1);
    auto t0 = clk::now();
    // Post-processing OFF for the array pass: with it on, Filament renders into
    // its own single-layer intermediate and blits, collapsing every world into
    // layer 0 (the backdrop pass re-enables it for refraction support).
    view_->setPostProcessingEnabled(false);
    view_->setVisibleLayers(0xFF, 0x02);
    view_->setRenderTarget(v.arrayRT);
    view_->setViewport({0, 0, W, H});
    {
        filament::Renderer::ClearOptions co;
        co.clearColor = {0.0f, 0.0f, 0.0f, 0.0f};   // transparent: alpha 0 = no object
        co.clear = true;
        renderer_->setClearOptions(co);
    }
    if (renderer_->beginFrame(swapchain_)) {
        renderer_->render(view_);
        renderer_->endFrame();
    }
    engine_->flushAndWait();
    auto t1 = clk::now();
    v.t_render += ns(t0, t1);

    const size_t rowBytes = size_t(W) * 8;   // RGBA16F = 8 bytes/texel
    const size_t slice = rowBytes * H;
    eglMakeCurrent(v.edpy, v.esurf, v.esurf, v.ectx);
    cudaGraphicsMapResources(1, &v.arrayCudaRes, 0);
    for (uint32_t i = 0; i < count; ++i) {
        cudaArray_t arr = nullptr;
        cudaGraphicsSubResourceGetMappedArray(&arr, v.arrayCudaRes, i, 0);
        cudaMemcpy2DFromArray(v.cudaBuf + size_t(out_offset + i) * slice, rowBytes,
                              arr, 0, 0, rowBytes, H, cudaMemcpyDeviceToDevice);
    }
    cudaGraphicsUnmapResources(1, &v.arrayCudaRes, 0);
    cudaDeviceSynchronize();
    v.t_copy += ns(t1, clk::now());
    v.n_frames += count;
}

void* Renderer::render_layered_to_cuda() {
    // Single-pass convenience for N <= layered_max_per_pass(): backdrop + 1 chunk.
    render_layered_backdrop();
    render_layered_objects(0, vk_->n);
    return vk_->cudaBuf;
}

void* Renderer::layered_backdrop_ptr() const { return vk_->backdropBuf; }
void* Renderer::layered_output_ptr() const { return vk_->cudaBuf; }

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
    if (v.arrayCudaRes) { cudaGraphicsUnregisterResource(v.arrayCudaRes); v.arrayCudaRes = nullptr; }
    if (v.backdropCudaRes) { cudaGraphicsUnregisterResource(v.backdropCudaRes); v.backdropCudaRes = nullptr; }
    if (v.backdropBuf) { cudaFree(v.backdropBuf); v.backdropBuf = nullptr; }
    if (v.cudaBuf) { cudaFree(v.cudaBuf); v.cudaBuf = nullptr; }
    if (engine_) {
        for (auto* rt : v.rt) if (rt) engine_->destroy(rt);
        for (auto* t : v.filColor) if (t) engine_->destroy(t);
        if (v.depth) engine_->destroy(v.depth);
        if (v.arrayRT) engine_->destroy(v.arrayRT);
        if (v.backdropRT) engine_->destroy(v.backdropRT);
        if (v.arrayColor) engine_->destroy(v.arrayColor);
        if (v.arrayDepth) engine_->destroy(v.arrayDepth);
        if (v.backdropColor) engine_->destroy(v.backdropColor);
        if (v.backdropDepth) engine_->destroy(v.backdropDepth);
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
    v.arrayRT = nullptr; v.arrayColor = nullptr; v.arrayDepth = nullptr;
    v.backdropRT = nullptr; v.backdropColor = nullptr; v.backdropDepth = nullptr;
    if (!v.glTex.empty()) { glDeleteTextures(v.glTex.size(), v.glTex.data()); v.glTex.clear(); }
    if (v.arrayTex) { glDeleteTextures(1, &v.arrayTex); v.arrayTex = 0; }
    if (v.backdropTex) { glDeleteTextures(1, &v.backdropTex); v.backdropTex = 0; }
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
