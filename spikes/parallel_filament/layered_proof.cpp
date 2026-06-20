// M1 PROOF: does the FORKED Filament route gl_Layer so ONE instanced draw fills
// N layers of an array-texture render target, each layer = one batch world?
//
// Fullscreen triangle, instanced N times. The forked main.vs writes
// gl_Layer = instance_index; the forked GL backend (env FILAMENT_LAYERED_BATCH)
// binds the whole (W,H,N) array texture as a LAYERED FBO. The unlit material
// colors each instance by getInstanceIndex(). If layer i reads back color(i),
// the fork correctly parallelized N worlds into one pass through Filament's PBR
// pipeline. No CUDA; pure Filament + GL readback.
#include <filament/Engine.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/View.h>
#include <filament/Camera.h>
#include <filament/SwapChain.h>
#include <filament/RenderableManager.h>
#include <filament/TransformManager.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/Texture.h>
#include <filament/RenderTarget.h>
#include <filament/VertexBuffer.h>
#include <filament/IndexBuffer.h>
#include <filament/InstanceBuffer.h>
#include <filament/Viewport.h>
#include <utils/EntityManager.h>
#include <math/mat4.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace filament;
using namespace filament::math;

static char* readFile(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(*len);
    if (fread(buf, 1, *len, f) != *len) { fprintf(stderr, "read fail\n"); exit(1); }
    fclose(f); return buf;
}

int main(int argc, char** argv) {
    const int W = 64, H = 64, N = 8;

    // --- forked-Filament GL engine (Filament creates its own headless EGL ctx) ---
    Engine::Config ec{}; ec.disableParallelShaderCompile = true;
    Engine* engine = Engine::Builder().backend(Engine::Backend::OPENGL)
                        .config(&ec).build();
    if (!engine) { fprintf(stderr, "engine build failed\n"); return 1; }
    Renderer* renderer = engine->createRenderer();
    Scene* scene = engine->createScene();
    View* view = engine->createView();
    auto& em = utils::EntityManager::get();
    utils::Entity camEnt = em.create();
    Camera* cam = engine->createCamera(camEnt);
    cam->setProjection(Camera::Projection::ORTHO, -1, 1, -1, 1, 0, 10);
    cam->lookAt({0, 0, 1}, {0, 0, 0}, {0, 1, 0});

    // --- material (forked: writes gl_Layer = instance) ---
    size_t mlen; char* mbuf = readFile("layered_proof.filamat", &mlen);
    Material* mat = Material::Builder().package(mbuf, mlen).build(*engine);
    free(mbuf);

    // --- (W,H,N) array color + depth render target ---
    Texture* color = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY)
        .format(Texture::InternalFormat::RGBA8)
        .usage(Texture::Usage::COLOR_ATTACHMENT | Texture::Usage::SAMPLEABLE)
        .build(*engine);
    Texture* depth = Texture::Builder().width(W).height(H).depth(N)
        .sampler(Texture::Sampler::SAMPLER_2D_ARRAY)
        .format(Texture::InternalFormat::DEPTH32F)
        .usage(Texture::Usage::DEPTH_ATTACHMENT)
        .build(*engine);
    RenderTarget* rt = RenderTarget::Builder()
        .texture(RenderTarget::AttachmentPoint::COLOR0, color)
        .texture(RenderTarget::AttachmentPoint::DEPTH, depth)
        .build(*engine);

    // --- fullscreen triangle, instanced N times ---
    static const float verts[] = { -1,-1,  3,-1,  -1,3 };
    VertexBuffer* vb = VertexBuffer::Builder().vertexCount(3).bufferCount(1)
        .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT2, 0, 8)
        .build(*engine);
    vb->setBufferAt(*engine, 0, VertexBuffer::BufferDescriptor(verts, sizeof verts, nullptr));
    static const uint16_t idx[] = { 0, 1, 2 };
    IndexBuffer* ib = IndexBuffer::Builder().indexCount(3)
        .bufferType(IndexBuffer::IndexType::USHORT).build(*engine);
    ib->setBuffer(*engine, IndexBuffer::BufferDescriptor(idx, sizeof idx, nullptr));

    std::vector<mat4f> xforms(N, mat4f());  // identity per world (layer routing only)
    InstanceBuffer* instances = InstanceBuffer::Builder(N).build(*engine);
    instances->setLocalTransforms(xforms.data(), N);

    utils::Entity tri = em.create();
    RenderableManager::Builder(1)
        .boundingBox({{0, 0, 0}, {2, 2, 2}})
        .material(0, mat->getDefaultInstance())
        .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vb, ib, 0, 3)
        .instances(N, instances)
        .culling(false)
        .build(*engine, tri);
    scene->addEntity(tri);

    view->setScene(scene);
    view->setCamera(cam);
    view->setViewport({0, 0, W, H});
    view->setRenderTarget(rt);
    view->setPostProcessingEnabled(false);
    Renderer::ClearOptions co; co.clear = true; co.clearColor = {0, 0, 0, 1};
    renderer->setClearOptions(co);

    SwapChain* sc = engine->createSwapChain(W, H, SwapChain::CONFIG_READABLE);

    // --- render ONE frame (all N layers in one instanced draw) ---
    if (renderer->beginFrame(sc)) {
        renderer->render(view);
        renderer->endFrame();
    }
    engine->flushAndWait();

    // --- read back each layer via a per-layer single-layer RenderTarget +
    // Filament readPixels (env disabled so the backend binds the single layer) ---
    printf("RENDERED N=%d layers in one instanced draw via forked Filament (no crash).\n", N);
    for (int layer = 0; layer < N; ++layer) {
        RenderTarget* lrt = RenderTarget::Builder()
            .texture(RenderTarget::AttachmentPoint::COLOR0, color)
            .layer(RenderTarget::AttachmentPoint::COLOR0, layer)
            .build(*engine);
        unsetenv("FILAMENT_LAYERED_BATCH");
        uint8_t px[4] = {0,0,0,0};
        Texture::PixelBufferDescriptor pbd(px, 4, Texture::Format::RGBA,
                                           Texture::Type::UBYTE);
        renderer->readPixels(lrt, W/2, H/2, 1, 1, std::move(pbd));
        engine->flushAndWait();
        printf("  layer %d center = (%3d,%3d,%3d)\n", layer, px[0], px[1], px[2]);
        engine->destroy(lrt);
        setenv("FILAMENT_LAYERED_BATCH", "1", 1);
    }
    printf("DONE.\n");
    return 0;
}
