import ctypes as ct

import torch
import torch.nn.functional as F

def ptr(A: torch.Tensor) -> ct.c_void_p:
    assert A is not None
    return ct.c_void_p(A.data.data_ptr())

lib = ct.cdll.LoadLibrary('/home/fernand/nanoGPT/barney/lib.so')

X = torch.rand((768,))
E1 = torch.rand((64, 768))
E2 = torch.rand((768, 64))
Xo = torch.zeros((768,))

lib.expert_forward(ptr(X), ptr(E1), ptr(E2), ptr(Xo))

Xo_ref = E2.matmul(F.relu(E1.matmul(X)))
print(Xo[0], Xo_ref[0])