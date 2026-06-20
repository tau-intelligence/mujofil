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
    }

    // --- scene setup (delegates to SceneBridge) ---
    void load_model(uintptr_t m) { bridge_->load_model(MODEL(m)); }
    void sync_transforms(uintptr_t m, uintptr_t d) { bridge_->sync_transforms(MODEL(m), DATA(d)); }
    void sync_camera(uintptr_t m, uintptr_t d, int cam) { bridge_->sync_camera(MODEL(m), DATA(d), cam); }
    void set_free_camera(float ex, float ey, float ez, float tx, float ty, float tz) {
        bridge_->set_free_camera(ex, ey, ez, tx, ty, tz);
    }
    void load_glb(const std::string& p) { bridge_->load_glb(p); }
    void load_ibl(const std::string& ibl, const std::string& sky) { bridge_->load_ibl(ibl, sky); }
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
    // or 4 (N,H,W,4); pass the dims in order.
    py::capsule wrap(void* dptr, int ndim, int64_t d0, int64_t d1, int64_t d2 = 0) {
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
        t.dtype = DLDataType{kDLUInt, 8, 1};
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
        .def_readwrite("batch_size", &RendererConfig::batch_size);

    py::class_<WarpRenderer>(m, "WarpRenderer")
        .def(py::init<const RendererConfig&>(), py::arg("config"))
        .def("load_model", &WarpRenderer::load_model)
        .def("sync_transforms", &WarpRenderer::sync_transforms)
        .def("sync_camera", &WarpRenderer::sync_camera,
             py::arg("model"), py::arg("data"), py::arg("cam_id") = -1)
        .def("set_free_camera", &WarpRenderer::set_free_camera)
        .def("load_glb", &WarpRenderer::load_glb)
        .def("load_ibl", &WarpRenderer::load_ibl)
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
        .def("reset_profile", &WarpRenderer::reset_profile)
        .def("profile", &WarpRenderer::profile)
        .def_property_readonly("width", &WarpRenderer::width)
        .def_property_readonly("height", &WarpRenderer::height);
}
