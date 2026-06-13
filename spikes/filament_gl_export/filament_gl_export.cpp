// Spike 2 (GLX): Filament's OpenGL backend on Linux uses the GLX platform (needs
// an X display). So we create a GLX context, share it with Filament, let Filament
// render into a GL texture WE own, then CUDA reads it back zero-copy.
#include <X11/Xlib.h>
#include <GL/glx.h>
#include <GL/gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/SwapChain.h>
#include <filament/Texture.h>
#include <filament/RenderTarget.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace filament;

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); \
    return 1;} } while(0)

typedef GLXContext (*PFNGLXCREATECTXATTRIBS)(Display*, GLXFBConfig, GLXContext, Bool, const int*);

int main() {
    const uint32_t W = 128, H = 96;

    // --- our GLX context (same windowing system Filament's PlatformGLX uses) --
    Display* xdpy = XOpenDisplay(nullptr);
    if (!xdpy) { fprintf(stderr, "XOpenDisplay failed\n"); return 1; }
    int screen = DefaultScreen(xdpy);
    int fbAttr[] = {
        GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8,
        None
    };
    int nfb = 0;
    GLXFBConfig* fbc = glXChooseFBConfig(xdpy, screen, fbAttr, &nfb);
    if (!fbc || nfb == 0) { fprintf(stderr, "glXChooseFBConfig failed\n"); return 1; }

    auto glXCreateContextAttribsARB =
        (PFNGLXCREATECTXATTRIBS)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
    GLXContext ctx;
    if (glXCreateContextAttribsARB) {
        int ctxAttr[] = { GLX_CONTEXT_MAJOR_VERSION_ARB, 4, GLX_CONTEXT_MINOR_VERSION_ARB, 1, None };
        ctx = glXCreateContextAttribsARB(xdpy, fbc[0], nullptr, True, ctxAttr);
    } else {
        ctx = glXCreateNewContext(xdpy, fbc[0], GLX_RGBA_TYPE, nullptr, True);
    }
    if (!ctx) { fprintf(stderr, "glXCreateContext failed\n"); return 1; }
    int pbAttr[] = { GLX_PBUFFER_WIDTH, 16, GLX_PBUFFER_HEIGHT, 16, None };
    GLXPbuffer pb = glXCreatePbuffer(xdpy, fbc[0], pbAttr);
    if (!glXMakeContextCurrent(xdpy, pb, pb, ctx)) {
        fprintf(stderr, "glXMakeContextCurrent failed\n"); return 1;
    }
    fprintf(stderr, "our ctx GL_RENDERER=%s\n", glGetString(GL_RENDERER));

    // --- GL texture we own --------------------------------------------------
    GLuint glTex = 0;
    glGenTextures(1, &glTex);
    glBindTexture(GL_TEXTURE_2D, glTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
    fprintf(stderr, "created GL texture id=%u\n", glTex);

    // --- Filament engine sharing OUR GLX context ----------------------------
    Engine* engine = Engine::Builder()
        .backend(Engine::Backend::OPENGL)
        .sharedContext((void*)ctx)
        .build();
    if (!engine) { fprintf(stderr, "Engine build failed\n"); return 1; }
    fprintf(stderr, "Filament engine built (shared GLX context)\n");

    Renderer* renderer = engine->createRenderer();
    Scene* scene = engine->createScene();
    auto camEnt = utils::EntityManager::get().create();
    Camera* camera = engine->createCamera(camEnt);
    camera->setProjection(60.0, (double)W / H, 0.1, 100.0);
    camera->lookAt({0, 0, 3}, {0, 0, 0}, {0, 1, 0});

    Texture* color = Texture::Builder()
        .width(W).height(H).levels(1)
        .format(Texture::InternalFormat::RGBA8)
        .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
        .import((intptr_t)glTex)
        .build(*engine);
    Texture* depth = Texture::Builder()
        .width(W).height(H).levels(1)
        .format(Texture::InternalFormat::DEPTH24)
        .usage(Texture::Usage::DEPTH_ATTACHMENT)
        .build(*engine);
    RenderTarget* rt = RenderTarget::Builder()
        .texture(RenderTarget::AttachmentPoint::COLOR0, color)
        .texture(RenderTarget::AttachmentPoint::DEPTH, depth)
        .build(*engine);

    View* view = engine->createView();
    view->setScene(scene);
    view->setCamera(camera);
    view->setViewport({0, 0, W, H});
    view->setRenderTarget(rt);
    view->setPostProcessingEnabled(false);

    Renderer::ClearOptions co;
    co.clearColor = {0.90f, 0.40f, 0.20f, 1.0f};
    co.clear = true;
    renderer->setClearOptions(co);

    SwapChain* sc = engine->createSwapChain(W, H, SwapChain::CONFIG_READABLE);
    if (renderer->beginFrame(sc)) {
        renderer->render(view);
        renderer->endFrame();
    }
    engine->flushAndWait();
    fprintf(stderr, "Filament rendered + flushed\n");

    // --- CUDA reads the GL texture Filament rendered into -------------------
    glXMakeContextCurrent(xdpy, pb, pb, ctx);
    cudaGraphicsResource* res = nullptr;
    CK(cudaGraphicsGLRegisterImage(&res, glTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    CK(cudaGraphicsMapResources(1, &res, 0));
    cudaArray_t array = nullptr;
    CK(cudaGraphicsSubResourceGetMappedArray(&array, res, 0, 0));
    uint8_t* dBuf = nullptr;
    const size_t bytes = (size_t)W * H * 4;
    CK(cudaMalloc(&dBuf, bytes));
    CK(cudaMemcpy2DFromArray(dBuf, (size_t)W * 4, array, 0, 0, (size_t)W * 4, H,
                             cudaMemcpyDeviceToDevice));
    CK(cudaGraphicsUnmapResources(1, &res, 0));

    uint8_t* host = (uint8_t*)malloc(bytes);
    CK(cudaMemcpy(host, dBuf, bytes, cudaMemcpyDeviceToHost));

    const uint8_t* c = host + (size_t)(H / 2 * W + W / 2) * 4;
    int nonuniform = 0;
    for (int p = 0; p < (int)(W * H); ++p) {
        const uint8_t* px = host + (size_t)p * 4;
        if (abs(px[0]-c[0])>2 || abs(px[1]-c[1])>2 || abs(px[2]-c[2])>2) ++nonuniform;
    }
    printf("center pixel = (%d,%d,%d,%d)\n", c[0], c[1], c[2], c[3]);
    bool ok = (c[0] > 5 || c[1] > 5 || c[2] > 5) && nonuniform == 0;
    if (ok) printf("FILAMENT-GL -> CUDA OK: Filament rendered into our GL texture, "
                   "CUDA read it back (uniform, non-zero), zero CPU bounce.\n");
    else printf("FAIL: center=(%d,%d,%d) nonuniform=%d/%d\n", c[0], c[1], c[2], nonuniform, W*H);

    cudaGraphicsUnregisterResource(res);
    cudaFree(dBuf); free(host);
    engine->destroy(rt); engine->destroy(color); engine->destroy(depth);
    engine->destroy(view); engine->destroy(sc);
    engine->destroyCameraComponent(camEnt);
    engine->destroy(scene); engine->destroy(renderer);
    Engine::destroy(&engine);
    glDeleteTextures(1, &glTex);
    glXMakeContextCurrent(xdpy, None, None, nullptr);
    glXDestroyContext(xdpy, ctx);
    XCloseDisplay(xdpy);
    return ok ? 0 : 1;
}
