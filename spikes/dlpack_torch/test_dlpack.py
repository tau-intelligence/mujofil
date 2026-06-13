"""Test: externally-managed CUDA memory -> torch tensor via DLPack (Phase 2 M4).

Proves the Python end of the zero-copy chain: a C++-allocated CUDA buffer (the
analog of Filament's exported render buffer) becomes a torch.cuda tensor with
NO copy, and PyTorch can run ops on it directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import dlpack_cuda


def main():
    H, W, C = 8, 8, 4
    FILL = 0xC3  # 195
    cap = dlpack_cuda.make_cuda_uint8(H, W, C, FILL)
    t = torch.from_dlpack(cap)

    print(f"tensor: shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
    assert t.is_cuda, "tensor must be on CUDA (zero-copy from device memory)"
    assert tuple(t.shape) == (H, W, C)
    assert t.dtype == torch.uint8

    # value check — torch reads the bytes the C++ side wrote, on-device
    uniq = torch.unique(t)
    print(f"unique values on device: {uniq.tolist()} (expected [{FILL}])")
    assert uniq.numel() == 1 and int(uniq[0]) == FILL

    # run a real op on-device to confirm it's a usable cuda tensor
    f = t.float().mean()
    torch.cuda.synchronize()
    print(f"mean (on cuda) = {f.item():.1f} (expected {FILL}.0)")
    assert abs(f.item() - FILL) < 1e-3

    print("\nDLPACK->TORCH OK: external CUDA memory is a usable torch.cuda "
          "tensor with no copy. The Phase 2 chain now reaches PyTorch.")


if __name__ == "__main__":
    main()
