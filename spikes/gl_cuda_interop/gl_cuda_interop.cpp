// Spike: prove EGL headless GL <-> CUDA zero-copy interop on this RTX4060/driver.
// No Filament. Steps:
//   1. Create a surfaceless EGL context (GL ES 3) on the NVIDIA device.
//   2. Create a GL texture, attach to an FBO, glClear it to a known color.
//   3. cudaGraphicsGLRegisterImage(tex) -> map -> cudaMemcpy2DFromArray to a
//      CUDA linear device buffer (this is the on-GPU copy, no CPU bounce).
//   4. cudaMemcpy D2H just to VERIFY the bytes match the clear color.
// If the verify passes, the GL->CUDA path works and we can build the Filament
// OpenGL single-sync zero-copy renderer on top of it.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); \
    return 1;} } while(0)

static void glchk(const char* where) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) fprintf(stderr, "GL error 0x%x at %s\n", e, where);
}

int main() {
    const int W = 64, H = 48;
    const float CR = 0.20f, CG = 0.60f, CB = 0.90f;  // expect (51,153,230,255)

    // --- 1. surfaceless EGL context on NVIDIA -------------------------------
    // Use the device-platform extension to pick the NVIDIA EGLDevice explicitly.
    PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    EGLDisplay dpy = EGL_NO_DISPLAY;
    if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
        EGLDeviceEXT devs[8]; EGLint nd = 0;
        eglQueryDevicesEXT(8, devs, &nd);
        fprintf(stderr, "EGL devices: %d\n", nd);
        for (EGLint i = 0; i < nd; ++i) {
            EGLDisplay d = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devs[i], nullptr);
            if (d == EGL_NO_DISPLAY) continue;
            EGLint major, minor;
            if (eglInitialize(d, &major, &minor)) {
                const char* vendor = eglQueryString(d, EGL_VENDOR);
                fprintf(stderr, "  dev %d vendor=%s egl=%d.%d\n", i, vendor ? vendor : "?", major, minor);
                dpy = d;
                if (vendor && strstr(vendor, "NVIDIA")) break;  // prefer NVIDIA
            }
        }
    }
    if (dpy == EGL_NO_DISPLAY) {
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLint major, minor;
        if (!eglInitialize(dpy, &major, &minor)) { fprintf(stderr, "eglInitialize failed\n"); return 1; }
    }

    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint cfgAttr[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    EGLConfig cfg; EGLint ncfg = 0;
    if (!eglChooseConfig(dpy, cfgAttr, &cfg, 1, &ncfg) || ncfg == 0) {
        fprintf(stderr, "eglChooseConfig failed\n"); return 1;
    }
    EGLint ctxAttr[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 0, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
    if (ctx == EGL_NO_CONTEXT) { fprintf(stderr, "eglCreateContext failed\n"); return 1; }
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        fprintf(stderr, "eglMakeCurrent (surfaceless) failed 0x%x\n", eglGetError()); return 1;
    }
    fprintf(stderr, "GL_VERSION=%s\nGL_RENDERER=%s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));

    // --- 2. GL texture + FBO, clear to known color --------------------------
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glchk("texImage");

    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete\n"); return 1;
    }
    glViewport(0, 0, W, H);
    glClearColor(CR, CG, CB, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glchk("clear");

    // --- 3. register GL texture with CUDA, map, copy to linear CUDA buffer ---
    cudaGraphicsResource* res = nullptr;
    CK(cudaGraphicsGLRegisterImage(&res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    CK(cudaGraphicsMapResources(1, &res, 0));
    cudaArray_t array = nullptr;
    CK(cudaGraphicsSubResourceGetMappedArray(&array, res, 0, 0));

    uint8_t* dBuf = nullptr;
    const size_t bytes = (size_t)W * H * 4;
    CK(cudaMalloc(&dBuf, bytes));
    CK(cudaMemcpy2DFromArray(dBuf, (size_t)W * 4, array, 0, 0, (size_t)W * 4, H,
                             cudaMemcpyDeviceToDevice));
    CK(cudaGraphicsUnmapResources(1, &res, 0));

    // --- 4. verify (D2H only for the test) ----------------------------------
    uint8_t* host = (uint8_t*)malloc(bytes);
    CK(cudaMemcpy(host, dBuf, bytes, cudaMemcpyDeviceToHost));
    uint8_t er = (uint8_t)(CR * 255 + 0.5f), eg = (uint8_t)(CG * 255 + 0.5f), eb = (uint8_t)(CB * 255 + 0.5f);
    int bad = 0;
    for (int p = 0; p < W * H; ++p) {
        const uint8_t* px = host + p * 4;
        if (abs(px[0]-er) > 1 || abs(px[1]-eg) > 1 || abs(px[2]-eb) > 1 || px[3] != 255) {
            if (bad < 3) fprintf(stderr, "  px %d = (%d,%d,%d,%d) expected (%d,%d,%d,255)\n",
                                 p, px[0], px[1], px[2], px[3], er, eg, eb);
            ++bad;
        }
    }
    printf("center pixel = (%d,%d,%d,%d) expected (%d,%d,%d,255)\n",
           host[(H/2*W+W/2)*4+0], host[(H/2*W+W/2)*4+1], host[(H/2*W+W/2)*4+2], host[(H/2*W+W/2)*4+3],
           er, eg, eb);
    if (bad == 0) printf("GL->CUDA INTEROP OK (all %d pixels match, zero CPU bounce in render->CUDA)\n", W*H);
    else printf("FAIL: %d/%d pixels wrong\n", bad, W*H);

    cudaGraphicsUnregisterResource(res);
    cudaFree(dBuf); free(host);
    glDeleteFramebuffers(1, &fbo); glDeleteTextures(1, &tex);
    eglDestroyContext(dpy, ctx); eglTerminate(dpy);
    return bad == 0 ? 0 : 1;
}
