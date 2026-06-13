// Phase 2, milestone 4: prove externally-managed CUDA memory -> torch tensor.
//
// The zero-copy chain ends by handing a CUDA device pointer to PyTorch. The
// standard, framework-agnostic way is DLPack: we wrap the pointer in a
// DLManagedTensor and expose it as a PyCapsule; Python calls
// torch.from_dlpack(capsule) to get a tensor that ALIASES the same VRAM.
//
// This module mimics the end of the real pipeline: it cudaMalloc's a buffer,
// fills it with a known pattern via cudaMemset/cudaMemcpy, and returns a torch
// tensor view of it. If torch sees the pattern (on cuda), the torch end is
// proven and this exact DLPack-export helper drops into the real renderer.

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <dlpack/dlpack.h>

namespace py = pybind11;

// Context kept alive for the lifetime of the DLPack tensor.
struct CudaBlob {
    void* dptr = nullptr;
    int64_t shape[3];
    int device = 0;
};

static void dlpack_deleter(DLManagedTensor* self) {
    auto* blob = static_cast<CudaBlob*>(self->manager_ctx);
    if (blob) {
        if (blob->dptr) cudaFree(blob->dptr);
        delete blob;
    }
    delete self;
}

static void capsule_destructor(PyObject* capsule) {
    // Only free if torch never consumed it (name still "dltensor").
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* mt = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        if (mt && mt->deleter) mt->deleter(mt);
    }
}

// Build an (H,W,C) uint8 CUDA tensor with a known fill, return as DLPack capsule.
py::capsule make_cuda_uint8(int h, int w, int c, int fill) {
    CudaBlob* blob = new CudaBlob();
    size_t bytes = (size_t)h * w * c;
    cudaError_t e = cudaMalloc(&blob->dptr, bytes);
    if (e != cudaSuccess) { delete blob; throw std::runtime_error("cudaMalloc"); }
    cudaMemset(blob->dptr, fill & 0xFF, bytes);
    cudaDeviceSynchronize();
    blob->shape[0] = h; blob->shape[1] = w; blob->shape[2] = c;
    cudaGetDevice(&blob->device);

    auto* mt = new DLManagedTensor();
    mt->manager_ctx = blob;
    mt->deleter = dlpack_deleter;
    DLTensor& t = mt->dl_tensor;
    t.data = blob->dptr;
    t.device = DLDevice{kDLCUDA, blob->device};
    t.ndim = 3;
    t.dtype = DLDataType{kDLUInt, 8, 1};
    t.shape = blob->shape;
    t.strides = nullptr;   // row-major contiguous
    t.byte_offset = 0;

    return py::capsule(mt, "dltensor", capsule_destructor);
}

PYBIND11_MODULE(dlpack_cuda, m) {
    m.doc() = "DLPack export of CUDA memory for torch.from_dlpack (Phase 2 M4)";
    m.def("make_cuda_uint8", &make_cuda_uint8,
          py::arg("h"), py::arg("w"), py::arg("c"), py::arg("fill"),
          "Allocate an (h,w,c) uint8 CUDA buffer filled with `fill`, "
          "return a DLPack capsule for torch.from_dlpack.");
}
