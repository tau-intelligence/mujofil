// mujofil-warp native pybind module: a Filament PBR renderer that renders real
// MuJoCo geometry (via mujofil's SceneBridge) and returns the frame as a
// torch.cuda tensor (DLPack), with no CPU round-trip.
//
// Reuses mujofil's SceneBridge/MaterialManager/LightManager source UNCHANGED and
// supplies mujofil-warp's own shared-device Renderer. mujofil's PyPI release is
// not modified.
#include "core/renderer.h"
#include "core/scene_bridge.h"

#include <mujoco/mujoco.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utils/Panic.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace py = pybind11;
using vf_mujoco::Renderer;
using vf_mujoco::RendererConfig;
using vf_mujoco::SceneBridge;

namespace {

const mjModel* MODEL(uintptr_t a) { return reinterpret_cast<const mjModel*>(a); }
const mjData*  DATA (uintptr_t a) { return reinterpret_cast<const mjData*>(a); }

// Frames-in-flight wave size. Filament caps in-flight frames; rendering more
// than this before a sync drops frames. Measured safe max = 2 on this build
// (K>=3 silently drops frames). Overridable via MUJOFIL_WARP_FLUSH_EVERY.
int flush_every_init() {
    if (const char* e = std::getenv("MUJOFIL_WARP_FLUSH_EVERY")) {
        int v = atoi(e);
        if (v >= 1) return v;
    }
    return 2;
}
const int FLUSH_EVERY = flush_every_init();

// DLPack: wrap an externally-owned CUDA pointer (the renderer keeps ownership).
struct DLCtx { int64_t shape[4]; };

void dl_deleter(DLManagedTensor* self) {
    delete static_cast<DLCtx*>(self->manager_ctx);
    delete self;  // NOTE: does NOT free the CUDA buffer (renderer owns it)
}
void capsule_dtor(PyObject* cap) {
    if (PyCapsule_IsValid(cap, "dltensor")) {
        auto* mt = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (mt && mt->deleter) mt->deleter(mt);
    }
}

class WarpRenderer {
public:
    explicit WarpRenderer(const RendererConfig& cfg)
        : renderer_(std::make_unique<Renderer>(cfg)) {
        renderer_->initialize();
        bridge_ = std::make_unique<SceneBridge>(*renderer_);
        // Layered parallel-batch: tell the bridge to build each geom instanced
        // batch_size times BEFORE the user calls load_model.
        if (renderer_->layered() && cfg.batch_size > 1)
            bridge_->enable_layered((int)cfg.batch_size);
    }

    // --- scene setup (delegates to SceneBridge) ---
    void load_model(uintptr_t m) { bridge_->load_model(MODEL(m)); }
    void sync_transforms(uintptr_t m, uintptr_t d) { bridge_->sync_transforms(MODEL(m), DATA(d)); }
    void sync_camera(uintptr_t m, uintptr_t d, int cam) { bridge_->sync_camera(MODEL(m), DATA(d), cam); }
    void set_free_camera(float ex, float ey, float ez, float tx, float ty, float tz) {
        bridge_->set_free_camera(ex, ey, ez, tx, ty, tz);
    }
    void load_glb(const std::string& p) { bridge_->load_glb(p); }
    void load_glb_xform(const std::string& p, const std::vector<float>& m) {
        if (m.size() != 16) throw std::runtime_error("load_glb_xform needs 16 floats (column-major 4x4)");
        bridge_->load_glb_xform(p, m.data());
    }
    // Ingest one GLB-environment mesh (already in WORLD space) as an instanced
    // layered renderable so a photoreal environment rides the single parallel draw
    // AND per-world egocentric cameras. Maps passed as raw RGBA bytes (or empty);
    // tangents (4/vertex, UV-aligned) enable correct normal mapping.
    void add_layered_env_mesh(
            const std::vector<float>& positions, const std::vector<float>& normals,
            const std::vector<float>& uvs, const std::vector<float>& tangents4,
            const std::vector<uint32_t>& indices,
            float r, float g, float b, float a, float roughness, float metallic,
            float em_r, float em_g, float em_b,
            const std::string& albedo_rgba, int albedo_w, int albedo_h,
            const std::string& normal_rgba, int normal_w, int normal_h,
            const std::string& mr_rgba, int mr_w, int mr_h,
            const std::string& emissive_rgba, int emissive_w, int emissive_h) {
        const int vc = (int)(positions.size() / 3);
        const float* nrm = (normals.size() == positions.size()) ? normals.data() : nullptr;
        const float* uv = (uvs.size() == (size_t)vc * 2) ? uvs.data() : nullptr;
        const float* tan = (tangents4.size() == (size_t)vc * 4) ? tangents4.data() : nullptr;
        auto bytes = [](const std::string& s) {
            return s.empty() ? nullptr : reinterpret_cast<const uint8_t*>(s.data()); };
        bridge_->add_layered_env_mesh(positions.data(), nrm, vc, uv, tan,
                                      indices.data(), (int)indices.size(),
                                      r, g, b, a, roughness, metallic,
                                      em_r, em_g, em_b,
                                      bytes(albedo_rgba), albedo_w, albedo_h,
                                      bytes(normal_rgba), normal_w, normal_h,
                                      bytes(mr_rgba), mr_w, mr_h,
                                      bytes(emissive_rgba), emissive_w, emissive_h);
    }
    void load_ibl(const std::string& ibl, const std::string& sky, bool with_skybox) {
        bridge_->load_ibl(ibl, sky, with_skybox);
    }
    void set_ambient_intensity(float i) { bridge_->set_ambient_intensity(i); }
    void clear_dynamic_lights() { bridge_->clear_dynamic_lights(); }
    void add_directional_light(float dx, float dy, float dz, float r, float g, float b,
                               float intensity, bool shadows) {
        bridge_->add_directional_light(dx, dy, dz, r, g, b, intensity, shadows);
    }
    void add_point_light(float x, float y, float z, float r, float g, float b,
                         float intensity, float falloff) {
        bridge_->add_point_light(x, y, z, r, g, b, intensity, falloff);
    }
    void add_spot_light(float x, float y, float z, float dx, float dy, float dz,
                        float r, float g, float b, float intensity, float falloff,
                        float inner_deg, float outer_deg, bool focused) {
        bridge_->add_spot_light(x, y, z, dx, dy, dz, r, g, b, intensity, falloff,
                                inner_deg, outer_deg, focused);
    }
    size_t geom_count() const { return bridge_->geom_count(); }

    // --- zero-copy render -> torch ---
    py::capsule render_dlpack() {
        void* dptr;
        {
            py::gil_scoped_release rel;
            dptr = renderer_->render_to_cuda();
        }
        return wrap(dptr, 3, renderer_->height(), renderer_->width());
    }

    // --- batched zero-copy render -> torch (N,H,W,4) ---
    py::capsule render_batch_dlpack(uintptr_t model, const std::vector<uintptr_t>& datas, int cam) {
        const uint32_t n = (uint32_t)datas.size();
        {
            py::gil_scoped_release rel;
            renderer_->begin_batch();
            // Filament caps frames-in-flight (~FLUSH_EVERY). Rendering more than
            // that without a sync silently drops frames. So we render in waves of
            // FLUSH_EVERY and flushAndWait once per wave — far cheaper than a sync
            // per frame, while keeping every world's image distinct.
            // EXCEPTION: the atlas/megatexture path (single_sync) shares ONE render
            // target across all tiles, so a mid-batch flush would re-clear the
            // atlas and lose earlier tiles — render the whole batch in one frame.
            const bool single = renderer_->single_sync();
            for (uint32_t i = 0; i < n; ++i) {
                bridge_->sync_transforms(MODEL(model), DATA(datas[i]));
                if (cam >= 0) bridge_->sync_camera(MODEL(model), DATA(datas[i]), cam);
                renderer_->render_frame_no_sync();
                if (!single && (i + 1) % FLUSH_EVERY == 0) renderer_->flush_wait();
            }
            renderer_->flush_wait();  // ensure the final partial wave completes
        }
        void* dptr;
        {
            py::gil_scoped_release rel;
            dptr = renderer_->finish_batch_to_cuda(n);
        }
        return wrap(dptr, 4, renderer_->height(), renderer_->width(), n);
    }

    // --- LAYERED parallel-batch render -> torch (N,H,W,4), chunked if N>cap ---
    py::capsule render_batch_layered_dlpack(uintptr_t model,
            const std::vector<uintptr_t>& datas, int cam) {
        const uint32_t n = (uint32_t)datas.size();
        void* dptr;
        {
            py::gil_scoped_release rel;
            // Render the shared static backdrop ONCE, then the per-world objects.
            // For N larger than the per-pass cap, render in chunks: each chunk
            // fills the InstanceBuffer with its worlds and writes its slice of the
            // (N,H,W,4) output. Per-world transforms live GPU-side (InstanceBuffer).
            //
            // EGOCENTRIC (cam >= 0): each world renders from its OWN camera. This is
            // done by VIEW-FOLDING -- sync_cameras_layered binds a shared projection-
            // only camera and computes each world's view matrix V_w, which
            // sync_transforms_layered folds into that world's instance transform
            // (localTransform_w = V_w * geomPose_w). So position = P * V_w * pose * v
            // renders world w from camera w, using only proven per-instance transform
            // + frame-uniform projection (no per-instance clip matrix). cam < 0 keeps
            // the shared-camera path byte-identical.
            renderer_->render_layered_backdrop();
            const uint32_t cap = renderer_->layered_max_per_pass();
            for (uint32_t start = 0; start < n; start += cap) {
                const uint32_t cn = std::min(cap, n - start);
                std::vector<const mjData*> chunk(cn);
                for (uint32_t i = 0; i < cn; ++i) chunk[i] = DATA(datas[start + i]);
                bridge_->sync_cameras_layered(MODEL(model), chunk, cam);
                bridge_->sync_transforms_layered(MODEL(model), chunk);
                renderer_->render_layered_objects(start, cn);
            }
            dptr = renderer_->layered_output_ptr();
        }
        return wrap(dptr, 4, renderer_->height(), renderer_->width(), n, /*half*/true);
    }

    // The (H,W,4) static backdrop from the most recent layered render. Python
    // composites the per-world objects (which carry alpha) over this.
    py::capsule layered_backdrop_dlpack() {
        return wrap(renderer_->layered_backdrop_ptr(), 3,
                    renderer_->height(), renderer_->width());
    }

    bool layered() const { return renderer_->layered(); }
    uint32_t width() const { return renderer_->width(); }
    uint32_t height() const { return renderer_->height(); }

    void reset_profile() { renderer_->reset_profile(); }
    py::dict profile() {
        py::dict d;
        d["render_ms"] = renderer_->prof_render_ms();
        d["flush_ms"] = renderer_->prof_flush_ms();
        d["copy_ms"] = renderer_->prof_copy_ms();
        d["frames"] = renderer_->prof_frames();
        return d;
    }

private:
    // Build a DLPack capsule for an externally-owned CUDA buffer. ndim is 3 (H,W,4)
    // or 4 (N,H,W,4); pass the dims in order. half=true wraps RGBA16F (float16).
    py::capsule wrap(void* dptr, int ndim, int64_t d0, int64_t d1, int64_t d2 = 0,
                     bool half = false) {
        auto* ctx = new DLCtx();
        if (ndim == 4) { ctx->shape[0] = d2; ctx->shape[1] = d0; ctx->shape[2] = d1; ctx->shape[3] = 4; }
        else           { ctx->shape[0] = d0; ctx->shape[1] = d1; ctx->shape[2] = 4; }
        auto* mt = new DLManagedTensor();
        mt->manager_ctx = ctx;
        mt->deleter = dl_deleter;
        DLTensor& t = mt->dl_tensor;
        t.data = dptr;
        t.device = DLDevice{kDLCUDA, renderer_->cuda_device()};
        t.ndim = ndim;
        t.dtype = half ? DLDataType{kDLFloat, 16, 1} : DLDataType{kDLUInt, 8, 1};
        t.shape = ctx->shape;
        t.strides = nullptr;
        t.byte_offset = 0;
        return py::capsule(mt, "dltensor", capsule_dtor);
    }

    std::unique_ptr<Renderer> renderer_;
    std::unique_ptr<SceneBridge> bridge_;
};

}  // namespace

#ifndef MUJOFIL_WARP_MODULE
#define MUJOFIL_WARP_MODULE _mujofil_warp
#endif

PYBIND11_MODULE(MUJOFIL_WARP_MODULE, m) {
    m.doc() = "mujofil-warp native: Filament PBR render of MuJoCo -> torch CUDA (zero-copy)";

    // Surface Filament's real error text. Filament throws utils::Panic, which is
    // NOT derived from std::exception, so pybind11's default handler can only
    // report the useless "Caught an unknown exception!". Catch it here and pass
    // its .what() through (e.g. the actual reason Engine::create() failed on an
    // unsupported GPU/driver). Non-Panic exceptions fall through to pybind's
    // default translator unchanged.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const utils::Panic& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    py::class_<RendererConfig>(m, "RendererConfig")
        .def(py::init<>())
        .def_readwrite("width", &RendererConfig::width)
        .def_readwrite("height", &RendererConfig::height)
        .def_readwrite("enable_ssao", &RendererConfig::enable_ssao)
        .def_readwrite("ssao_quality", &RendererConfig::ssao_quality)
        .def_readwrite("ssao_ssct", &RendererConfig::ssao_ssct)
        .def_readwrite("enable_bloom", &RendererConfig::enable_bloom)
        .def_readwrite("enable_fxaa", &RendererConfig::enable_fxaa)
        .def_readwrite("enable_msaa", &RendererConfig::enable_msaa)
        .def_readwrite("msaa_samples", &RendererConfig::msaa_samples)
        .def_readwrite("enable_shadows", &RendererConfig::enable_shadows)
        .def_readwrite("enable_ssr", &RendererConfig::enable_ssr)
        .def_readwrite("exposure", &RendererConfig::exposure)
        .def_readwrite("tone_mapping", &RendererConfig::tone_mapping)
        .def_readwrite("dithering", &RendererConfig::dithering)
        .def_readwrite("batch_size", &RendererConfig::batch_size)
        .def_readwrite("layered", &RendererConfig::layered);

    py::class_<WarpRenderer>(m, "WarpRenderer")
        .def(py::init<const RendererConfig&>(), py::arg("config"))
        .def("load_model", &WarpRenderer::load_model)
        .def("sync_transforms", &WarpRenderer::sync_transforms)
        .def("sync_camera", &WarpRenderer::sync_camera,
             py::arg("model"), py::arg("data"), py::arg("cam_id") = -1)
        .def("set_free_camera", &WarpRenderer::set_free_camera)
        .def("load_glb", &WarpRenderer::load_glb)
        .def("load_glb_xform", &WarpRenderer::load_glb_xform)
        .def("add_layered_env_mesh", &WarpRenderer::add_layered_env_mesh)
        .def("load_ibl", &WarpRenderer::load_ibl,
             py::arg("ibl"), py::arg("sky"), py::arg("with_skybox") = false)
        .def("set_ambient_intensity", &WarpRenderer::set_ambient_intensity)
        .def("clear_dynamic_lights", &WarpRenderer::clear_dynamic_lights)
        .def("add_directional_light", &WarpRenderer::add_directional_light,
             py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("r"), py::arg("g"), py::arg("b"),
             py::arg("intensity"), py::arg("cast_shadows") = false)
        .def("add_point_light", &WarpRenderer::add_point_light)
        .def("add_spot_light", &WarpRenderer::add_spot_light,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("dx"), py::arg("dy"), py::arg("dz"),
             py::arg("r"), py::arg("g"), py::arg("b"), py::arg("intensity"), py::arg("falloff"),
             py::arg("inner_deg"), py::arg("outer_deg"), py::arg("focused") = true)
        .def_property_readonly("geom_count", &WarpRenderer::geom_count)
        .def("render_dlpack", &WarpRenderer::render_dlpack,
             "Render the scene; return a DLPack capsule (H,W,4 uint8 cuda) for torch.from_dlpack.")
        .def("render_batch_dlpack", &WarpRenderer::render_batch_dlpack,
             py::arg("model"), py::arg("datas"), py::arg("cam_id") = -1,
             "Render N worlds (one MjData each), one GPU sync; return (N,H,W,4) uint8 cuda DLPack.")
        .def("render_batch_layered_dlpack", &WarpRenderer::render_batch_layered_dlpack,
             py::arg("model"), py::arg("datas"), py::arg("cam_id") = -1,
             "LAYERED: render N worlds in ONE instanced pass (forked gl_Layer); (N,H,W,4) cuda DLPack.")
        .def("layered_backdrop_dlpack", &WarpRenderer::layered_backdrop_dlpack,
             "The (H,W,4) static backdrop from the last layered render (compose under objects).")
        .def_property_readonly("layered", &WarpRenderer::layered)
        .def("reset_profile", &WarpRenderer::reset_profile)
        .def("profile", &WarpRenderer::profile)
        .def_property_readonly("width", &WarpRenderer::width)
        .def_property_readonly("height", &WarpRenderer::height);
}
