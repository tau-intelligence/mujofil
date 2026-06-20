// Spike: can we render N different worlds in ONE GPU pass instead of N serial
// draws? This is the core feasibility test for a parallel batch rasterizer.
//
// Pure EGL + desktop GL 4.1 (NVIDIA, headless) — NO Filament, NO CUDA. We render
// a shared "scene" mesh into an (W x H x N) GL_TEXTURE_2D_ARRAY, giving each batch
// world its own per-instance transform + color (a UBO indexed by gl_InstanceID).
//
// Two paths, same total pixel + vertex work, compared head to head:
//   SERIAL   : N draws, each bound to one array layer (today's "render() x N").
//   LAYERED  : ONE glDrawArraysInstanced(.., N); the vertex shader writes
//              gl_Layer = gl_InstanceID (via GL_ARB_shader_viewport_layer_array)
//              so all N worlds rasterize in a single draw call.
// Also tests VIEWPORT routing (gl_ViewportIndex, capped at GL_MAX_VIEWPORTS=16)
// for completeness.
//
// If LAYERED >> SERIAL, a Filament GL-backend fork to emit gl_Layer routing is
// justified: it's the only way to break the per-world draw-call serialization.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void glchk(const char* w) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) fprintf(stderr, "GL error 0x%x at %s\n", e, w);
}

// ---- EGL headless desktop-GL 4.1 context on the NVIDIA device ----------------
static EGLDisplay g_dpy = EGL_NO_DISPLAY;
static bool init_gl() {
    auto qd = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    auto gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    EGLDisplay dpy = EGL_NO_DISPLAY;
    if (qd && gpd) {
        EGLDeviceEXT devs[8]; EGLint nd = 0; qd(8, devs, &nd);
        for (EGLint i = 0; i < nd; ++i) {
            EGLDisplay d = gpd(EGL_PLATFORM_DEVICE_EXT, devs[i], nullptr);
            EGLint a, b;
            if (d != EGL_NO_DISPLAY && eglInitialize(d, &a, &b)) {
                const char* v = eglQueryString(d, EGL_VENDOR);
                if (v && strstr(v, "NVIDIA")) { dpy = d; break; }
                if (dpy == EGL_NO_DISPLAY) dpy = d;
            }
        }
    }
    if (dpy == EGL_NO_DISPLAY) {
        dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLint a, b;
        if (!eglInitialize(dpy, &a, &b)) { fprintf(stderr, "eglInitialize failed\n"); return false; }
    }
    g_dpy = dpy;
    eglBindAPI(EGL_OPENGL_API);
    EGLint ca[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE };
    EGLConfig c; EGLint n = 0;
    if (!eglChooseConfig(dpy, ca, &c, 1, &n) || n == 0) { fprintf(stderr, "chooseConfig failed\n"); return false; }
    EGLint cx[] = { EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, c, EGL_NO_CONTEXT, cx);
    if (ctx == EGL_NO_CONTEXT) { fprintf(stderr, "createContext failed\n"); return false; }
    EGLint pb[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    EGLSurface s = eglCreatePbufferSurface(dpy, c, pb);
    if (!eglMakeCurrent(dpy, s, s, ctx)) { fprintf(stderr, "makeCurrent failed 0x%x\n", eglGetError()); return false; }
    glewExperimental = GL_TRUE;
    GLenum ge = glewInit();
    // GLEW (classic) returns GLEW_ERROR_NO_GLX_DISPLAY / "Unknown error" on a
    // core-profile context because it probes glGetString(GL_EXTENSIONS) (NULL on
    // core). The function pointers ARE loaded with glewExperimental — so proceed.
    if (ge != GLEW_OK)
        fprintf(stderr, "glewInit warning (proceeding): %s\n", glewGetErrorString(ge));
    glGetError();  // swallow the benign GL_INVALID_ENUM glewInit leaves behind
    return true;
}

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "shader compile failed:\n%s\n", log); return 0; }
    return s;
}
static GLuint link(const char* vs, const char* fs) {
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "link failed:\n%s\n", log); return 0; }
    return p;
}

// Per-world data (std140): a 2D rotation+scale (as mat4) and a color.
struct World { float m[16]; float color[4]; };

int main(int argc, char** argv) {
    int W = 256, H = 256, N = 256, TRIS = 4096, ITERS = 50;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--res")) W = H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tris")) TRIS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) ITERS = atoi(argv[++i]);
    }
    if (!init_gl()) return 1;
    fprintf(stderr, "GL=%s\n", glGetString(GL_VERSION));
    bool haveVPLayer = glewIsSupported("GL_ARB_shader_viewport_layer_array");
    int maxLayers = 0, maxVP = 0;
    glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &maxLayers);
    glGetIntegerv(GL_MAX_VIEWPORTS, &maxVP);
    fprintf(stderr, "ARB_shader_viewport_layer_array=%d  MAX_ARRAY_LAYERS=%d  MAX_VIEWPORTS=%d\n",
            (int)haveVPLayer, maxLayers, maxVP);
    if (N > maxLayers) { fprintf(stderr, "N=%d > MAX_ARRAY_LAYERS=%d\n", N, maxLayers); return 1; }
    // The per-world data lives in a UBO sized [512] (40KB, within the 64KB block
    // limit). Production would hold transforms in a texture-buffer (samplerBuffer,
    // GL 3.1) to scale past the UBO cap; the spike only needs N<=512 to prove the
    // single-pass-vs-serial throughput question.
    if (N > 512) { fprintf(stderr, "spike UBO caps N at 512 (use a TBO in production)\n"); return 1; }

    // --- shared "scene" geometry: TRIS triangles in a unit disk (a stand-in for
    // real per-world geometry; same VBO drawn for every world) -----------------
    std::vector<float> verts;
    verts.reserve(TRIS * 3 * 2);
    for (int t = 0; t < TRIS; ++t) {
        float a0 = 6.2831853f * t / TRIS, a1 = 6.2831853f * (t + 1) / TRIS;
        float r = 0.85f;
        verts.insert(verts.end(), { 0.f, 0.f, r * cosf(a0), r * sinf(a0), r * cosf(a1), r * sinf(a1) });
    }
    const int VTX = TRIS * 3;
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * 4, verts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // --- per-world UBO (distinct rotation + color so layers differ) -----------
    std::vector<World> worlds(N);
    for (int i = 0; i < N; ++i) {
        float ang = 6.2831853f * i / N, c = cosf(ang), s = sinf(ang), sc = 0.6f;
        float m[16] = { c*sc,-s*sc,0,0,  s*sc,c*sc,0,0,  0,0,1,0,  0,0,0,1 };
        memcpy(worlds[i].m, m, sizeof m);
        worlds[i].color[0] = 0.5f + 0.5f * cosf(ang);
        worlds[i].color[1] = 0.5f + 0.5f * cosf(ang + 2.094f);
        worlds[i].color[2] = 0.5f + 0.5f * cosf(ang + 4.188f);
        worlds[i].color[3] = 1.f;
    }
    GLuint ubo; glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, N * sizeof(World), worlds.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

    // --- (W x H x N) texture array target + layered FBO -----------------------
    GLuint texArr; glGenTextures(1, &texArr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texArr);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, W, H, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint fbo; glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glchk("setup");

    // shaders: per-instance transform+color from UBO; LAYERED writes gl_Layer.
    const char* fs =
        "#version 410 core\n"
        "in vec4 vCol; out vec4 o; void main(){ o = vCol; }\n";
    const char* vs_layered =
        "#version 410 core\n"
        "#extension GL_ARB_shader_viewport_layer_array : require\n"
        "layout(location=0) in vec2 p;\n"
        "struct Wd { mat4 m; vec4 color; };\n"
        "layout(std140) uniform U { Wd w[512]; };\n"
        "out vec4 vCol;\n"
        "void main(){ gl_Layer = gl_InstanceID; vCol = w[gl_InstanceID].color;\n"
        "  gl_Position = w[gl_InstanceID].m * vec4(p,0,1); }\n";
    const char* vs_serial =
        "#version 410 core\n"
        "layout(location=0) in vec2 p;\n"
        "struct Wd { mat4 m; vec4 color; };\n"
        "layout(std140) uniform U { Wd w[512]; };\n"
        "uniform int uWorld;\n"
        "out vec4 vCol;\n"
        "void main(){ vCol = w[uWorld].color;\n"
        "  gl_Position = w[uWorld].m * vec4(p,0,1); }\n";

    GLuint progSerial = link(vs_serial, fs);
    GLuint progLayered = haveVPLayer ? link(vs_layered, fs) : 0;
    if (!progSerial) return 1;
    for (GLuint pr : {progSerial, progLayered}) {
        if (!pr) continue;
        GLuint bi = glGetUniformBlockIndex(pr, "U");
        if (bi != GL_INVALID_INDEX) glUniformBlockBinding(pr, bi, 0);
    }
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);

    auto bench = [&](const char* name, auto&& draw_once) -> double {
        for (int w = 0; w < 5; ++w) draw_once();   // warmup
        glFinish();
        auto t0 = clk::now();
        for (int it = 0; it < ITERS; ++it) draw_once();
        glFinish();
        double dt = ms(t0, clk::now()) / ITERS;
        double cams = N / (dt / 1000.0);
        fprintf(stdout, "  %-9s %7.3f ms/batch  %9.0f cam/s\n", name, dt, cams);
        return dt;
    };

    // SERIAL: bind each layer, draw the scene once per world (today's model).
    int uWorld = glGetUniformLocation(progSerial, "uWorld");
    auto serial = [&]() {
        glUseProgram(progSerial);
        glBindVertexArray(vao);
        for (int i = 0; i < N; ++i) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texArr, 0, i);
            glClear(GL_COLOR_BUFFER_BIT);
            glUniform1i(uWorld, i);
            glDrawArrays(GL_TRIANGLES, 0, VTX);
        }
    };
    // LAYERED: attach the WHOLE array (layered), ONE instanced draw -> all worlds.
    auto layered = [&]() {
        glUseProgram(progLayered);
        glBindVertexArray(vao);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texArr, 0);  // layered
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArraysInstanced(GL_TRIANGLES, 0, VTX, N);
    };

    printf("=== parallel-batch render spike: N=%d  %dx%d  tris/world=%d  iters=%d ===\n",
           N, W, H, TRIS, ITERS);
    double tS = bench("SERIAL", serial);
    double tL = progLayered ? bench("LAYERED", layered) : 0;

    // correctness: read a few layers back, confirm distinct colors.
    bool ok = true;
    if (progLayered) {
        layered(); glFinish();
        std::vector<unsigned char> px(W * H * 4);
        auto centerPixel = [&](int layer, unsigned char out[4]) {
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texArr, 0, layer);
            glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
        };
        unsigned char a[4], b[4];
        centerPixel(0, a); centerPixel(N / 2, b);
        int d = abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2]);
        printf("  correctness: layer0 center=(%d,%d,%d) layer%d center=(%d,%d,%d) diff=%d -> %s\n",
               a[0],a[1],a[2], N/2, b[0],b[1],b[2], d, d > 10 ? "DISTINCT (ok)" : "IDENTICAL (BAD)");
        ok = d > 10;
    }
    if (progLayered && tL > 0)
        printf("  => LAYERED speedup vs SERIAL: %.2fx  (%s)\n", tS / tL, ok ? "correct" : "INCORRECT");
    else
        printf("  => layered path unavailable (no GL_ARB_shader_viewport_layer_array)\n");
    return 0;
}
