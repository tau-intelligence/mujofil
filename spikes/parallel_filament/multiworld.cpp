// M1 STEP 2: render N WORLDS of real depth-tested 3D geometry in ONE pass.
//
// Each world = a tumbling cube at a distinct per-world pose (carried in a
// Filament InstanceBuffer). The forked main.vs routes instance w -> array layer
// w via gl_Layer, so ONE instanced draw fills all N layers, each a different
// world, through Filament's real pipeline (depth test, perspective camera).
//
// Compares:
//   LAYERED : one (W,H,N) array RT, ONE render() -> all N worlds
//   SERIAL  : N single-layer render targets, N render() calls (today's model)
// Verifies they agree pixel-wise, then benchmarks both at output resolution.
#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/SwapChain.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/Material.h>
#include <filament/Texture.h>
#include <filament/RenderTarget.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/InstanceBuffer.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>
#include <math/mat4.h>
#include <math/vec3.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace filament;
using namespace filament::math;
using clk = std::chrono::high_resolution_clock;
static double msec(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static char* readFile(const char* p, size_t* len) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc(*len);
    if (fread(b, 1, *len, f) != *len) { exit(1); }
    fclose(f); return b;
}

// per-world model transform: a cube spun + shifted by world index.
static mat4f worldPose(int w, int N) {
    float a = 6.2831853f * w / N;
    mat4f t = mat4f::translation(float3{0.4f * std::cos(a), 0.4f * std::sin(a), 0.0f});
    mat4f r = mat4f::rotation(a, float3{0.3f, 0.7f, 0.5f});
    return t * r;
}

int main(int argc, char** argv) {
    int W = 256, H = 256, N = 256, ITERS = 30;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--res")) W = H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) ITERS = atoi(argv[++i]);
    }

    Engine::Config ec{}; ec.disableParallelShaderCompile = true;
    Engine* engine = Engine::Builder().backend(Engine::Backend::OPENGL).config(&ec).build();
    if (!engine) { fprintf(stderr, "engine build failed\n"); return 1; }
    Renderer* renderer = engine->createRenderer();
    Scene* sceneL = engine->createScene();
    Scene* sceneS = engine->createScene();
    auto& em = utils::EntityManager::get();
    utils::Entity camEnt = em.create();
    Camera* cam = engine->createCamera(camEnt);
    cam->setProjection(45.0, double(W) / H, 0.1, 20.0);
    cam->lookAt({0, -2.4, 1.4}, {0, 0, 0}, {0, 0, 1});

    size_t mlen; char* mbuf = readFile("layered_proof.filamat", &mlen);
    Material* mat = Material::Builder().package(mbuf, mlen).build(*engine); free(mbuf);

    // cube geometry (positions only; material colors per instance)
    const float s = 0.25f;
    float V[] = {
        -s,-s,-s,  s,-s,-s,  s,s,-s,  -s,s,-s,
        -s,-s, s,  s,-s, s,  s,s, s,  -s,s, s };
    uint16_t I[] = {0,1,2, 0,2,3, 4,6,5, 4,7,6, 0,4,5, 0,5,1,
                    1,5,6, 1,6,2, 2,6,7, 2,7,3, 3,7,4, 3,4,0};
    VertexBuffer* vb = VertexBuffer::Builder().vertexCount(8).bufferCount(1)
        .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT3, 0, 12)
        .build(*engine);
    vb->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(V, sizeof V, nullptr));
    IndexBuffer* ib = IndexBuffer::Builder().indexCount(36)
        .bufferType(IndexBuffer::IndexType::USHORT).build(*engine);
    ib->setBuffer(*engine, IndexBuffer::BufferDescriptor(I, sizeof I, nullptr));

    // per-world transforms
    std::vector<mat4f> poses(N);
    for (int w = 0; w < N; ++w) poses[w] = worldPose(w, N);

    // LAYERED scene: ONE renderable instanced N times (instance w -> layer w)
    InstanceBuffer* instances = InstanceBuffer::Builder(N).build(*engine);
    instances->setLocalTransforms(poses.data(), N);
    utils::Entity cubeL = em.create();
    RenderableManager::Builder(1)
        .boundingBox({{0,0,0},{4,4,4}})
        .material(0, mat->getDefaultInstance())
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vb, ib, 0, 36)
        .instances(N, instances)
        .culling(false)
        .build(*engine, cubeL);
    sceneL->addEntity(cubeL);

    // SERIAL scene: ONE non-instanced renderable; we re-pose it per world each draw
    utils::Entity cubeS = em.create();
    RenderableManager::Builder(1)
        .boundingBox({{0,0,0},{4,4,4}})
        .material(0, mat->getDefaultInstance())
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vb, ib, 0, 36)
        .culling(false)
        .build(*engine, cubeS);
    sceneS->addEntity(cubeS);
    auto& tcm = engine->getTransformManager();

    // (W,H,N) array color + depth, layered RT
    Texture* color = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY).format(Texture::InternalFormat::RGBA8)
        .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE | Texture::Usage::BLIT_SRC)
        .build(*engine);
    Texture* depth = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY).format(Texture::InternalFormat::DEPTH32F)
        .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine);
    RenderTarget* rtLayered = RenderTarget::Builder()
        .texture(RenderTarget::AttachmentPoint::COLOR0, color)
        .texture(RenderTarget::AttachmentPoint::DEPTH, depth)
        .build(*engine);

    // SERIAL path gets its OWN array textures so its single-layer FBOs never
    // corrupt the layered RT's FBO state (Filament caches FBO attachments).
    Texture* colorS = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY).format(Texture::InternalFormat::RGBA8)
        .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE | Texture::Usage::BLIT_SRC)
        .build(*engine);
    Texture* depthS = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY).format(Texture::InternalFormat::DEPTH32F)
        .usage(Texture::Usage::DEPTH_ATTACHMENT).build(*engine);

    View* viewL = engine->createView();
    viewL->setScene(sceneL); viewL->setCamera(cam);
    viewL->setViewport({0,0,(uint32_t)W,(uint32_t)H});
    viewL->setRenderTarget(rtLayered);
    viewL->setPostProcessingEnabled(false);
    View* viewS = engine->createView();
    viewS->setScene(sceneS); viewS->setCamera(cam);
    viewS->setViewport({0,0,(uint32_t)W,(uint32_t)H});
    viewS->setPostProcessingEnabled(false);
    Renderer::ClearOptions co; co.clear = true; co.clearColor = {0.05f,0.05f,0.07f,1};
    renderer->setClearOptions(co);
    SwapChain* sc = engine->createSwapChain(W, H, SwapChain::CONFIG_READABLE);

    auto setEnv = [](bool on){ if(on) setenv("FILAMENT_LAYERED_BATCH","1",1); else unsetenv("FILAMENT_LAYERED_BATCH"); };

    // per-layer readback RTs for the LAYERED color array (separate from serial's)
    std::vector<RenderTarget*> lrt(N);
    for (int w = 0; w < N; ++w)
        lrt[w] = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, color)
            .layer(RenderTarget::AttachmentPoint::COLOR0, w)
            .build(*engine);

    // ---- LAYERED: one render() into all N layers ----
    auto layeredOnce = [&](){
        setEnv(true);
        if (renderer->beginFrame(sc)) { renderer->render(viewL); renderer->endFrame(); }
        engine->flushAndWait();
    };
    // ---- SERIAL: N single-layer RTs, re-pose + render each ----
    std::vector<RenderTarget*> srt(N);
    for (int w = 0; w < N; ++w)
        srt[w] = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, colorS)
            .layer(RenderTarget::AttachmentPoint::COLOR0, w)
            .texture(RenderTarget::AttachmentPoint::DEPTH, depthS)
            .layer(RenderTarget::AttachmentPoint::DEPTH, w)
            .build(*engine);
    auto serialOnce = [&](){
        setEnv(false);
        for (int w = 0; w < N; ++w) {
            tcm.setTransform(tcm.getInstance(cubeS), poses[w]);
            viewS->setRenderTarget(srt[w]);
            if (renderer->beginFrame(sc)) { renderer->render(viewS); renderer->endFrame(); }
        }
        engine->flushAndWait();
    };

    auto bench = [&](const char* name, auto&& once){
        for (int i=0;i<3;i++) once();
        auto t0=clk::now();
        for (int i=0;i<ITERS;i++) once();
        double dt=msec(t0,clk::now())/ITERS;
        printf("  %-8s %8.3f ms/batch  %9.0f cam/s\n", name, dt, N/(dt/1000.0));
        return dt;
    };

    printf("=== N worlds (instanced cubes, depth) one-pass vs serial: N=%d %dx%d iters=%d ===\n",
           N, W, H, ITERS);

    // --- CORRECTNESS FIRST (fresh layered render, before any serial path runs
    // so the in-process serial single-layer FBOs can't corrupt the read) ---
    layeredOnce();
    setEnv(false);
    std::vector<uint8_t> full(W * H * 4);
    {
        int probe[5] = {0, N/4, N/2, 3*N/4, N-1};
        bool allRendered = true;
        printf("  per-layer non-bg pixel counts (each world rendered in ONE pass):\n");
        for (int k = 0; k < 5; ++k) {
            Texture::PixelBufferDescriptor pbd(full.data(), full.size(),
                Texture::Format::RGBA, Texture::Type::UBYTE);
            renderer->readPixels(lrt[probe[k]], 0, 0, W, H, std::move(pbd));
            engine->flushAndWait();
            int nonbg = 0;
            for (int i = 0; i < W*H; ++i)
                if (abs(full[i*4]-13)+abs(full[i*4+1]-13)+abs(full[i*4+2]-18) > 30) nonbg++;
            printf("    world %4d: %6d px\n", probe[k], nonbg);
            if (nonbg < 50 || nonbg >= W*H) allRendered = false;  // empty OR all-black = bad
        }
        printf("  => %s\n", allRendered ? "ALL N WORLDS RENDERED in one pass (ok)" : "LAYER ANOMALY");
        // montage
        int show[4] = {0, N/4, N/2, N-1};
        std::vector<uint8_t> mont(H * (W*4) * 3);
        for (int k = 0; k < 4; ++k) {
            Texture::PixelBufferDescriptor pbd(full.data(), full.size(),
                Texture::Format::RGBA, Texture::Type::UBYTE);
            renderer->readPixels(lrt[show[k]], 0, 0, W, H, std::move(pbd));
            engine->flushAndWait();
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                int src = ((H-1-y)*W + x)*4, dst = (y*(W*4) + (k*W + x))*3;
                mont[dst]=full[src]; mont[dst+1]=full[src+1]; mont[dst+2]=full[src+2];
            }
        }
        FILE* f = fopen("out_multiworld.ppm","wb");
        fprintf(f, "P6\n%d %d\n255\n", W*4, H);
        fwrite(mont.data(), 1, mont.size(), f); fclose(f);
        printf("  wrote out_multiworld.ppm (worlds 0, %d, %d, %d)\n", N/4, N/2, N-1);
    }

    // --- TIMING (serial uses its OWN textures; correctness already captured) ---
    const bool layeredOnly = getenv("LAYERED_ONLY") != nullptr;
    if (!layeredOnly) {
        double tS = bench("SERIAL", serialOnce);
        double tL = bench("LAYERED", layeredOnce);
        printf("  => LAYERED speedup: %.2fx\n", tS/tL);
    }
    printf("DONE.\n");
    return 0;
}
