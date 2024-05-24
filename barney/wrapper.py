import ctypes as ct

import torch

def ptr(A: torch.Tensor) -> ct.c_void_p:
    assert A is not None
    return ct.c_void_p(A.data.data_ptr())

lib = ct.cdll.LoadLibrary('/home/fernand/nanoGPT/barney/lib.so')