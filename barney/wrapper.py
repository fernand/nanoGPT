import time
import ctypes as ct

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)

B = 16
T = 1024

def timeit(f, multiplier=1):
    n = 10
    t1 = time.perf_counter_ns()
    for _ in range(n):
        f()
    t2 = time.perf_counter_ns()
    print('exec time', (t2 - t1) / (1000 * n * multiplier), 'us')

def ptr(A: torch.Tensor) -> ct.c_void_p:
    assert A is not None
    return ct.c_void_p(A.data.data_ptr())

def test():
    X = torch.rand((768,))
    E1 = torch.rand((64, 768))
    E2 = torch.rand((768, 64))
    Xo = torch.zeros((768,))
    print((E1 @ X).shape)
    lib.expert_forward(ptr(X), ptr(E1), ptr(E2), ptr(Xo))
    Xo = Xo.numpy()
    Xo_ref = (E2 @ F.relu(E1 @ X)).numpy()
    assert all([abs((xo - xo_ref) / xo_ref) < 1e-5 for (xo, xo_ref) in zip(Xo, Xo_ref)])

def bench():
    X = torch.rand((768,))
    E1 = torch.rand((64, 768))
    E2 = torch.rand((768, 64))
    Xo = torch.zeros((768,))
    print('C lib speed: ', end='')
    # Do an inner loop in the C code to avoid benchmarking ctypes overhead.
    timeit(lambda: lib.bench_expert_forward(ptr(X), ptr(E1), ptr(E2), ptr(Xo)), multiplier=1000)

def bench_pytorch():
    X = torch.rand((768,))
    E1 = torch.rand((64, 768))
    E2 = torch.rand((768, 64))
    Xo = torch.zeros((768,))
    print('Pytorch speed: ', end='')
    timeit(lambda: E2 @ F.relu(E1 @ X))

def bench_ff_cuda():
    X = torch.rand((B, T, 768)).cuda()
    E1 = nn.Linear(768, 4 * 768).cuda()
    E2 = nn.Linear(4 * 768, 768).cuda()
    print('Pytorch MLP FF forward speed: ', end='')
    def l():
        E2(F.relu(E1(X)))
    timeit(l)

if __name__ == '__main__':
    lib = ct.cdll.LoadLibrary('/home/fernand/nanoGPT/barney/lib.so')
    test()
    bench()
    # bench_pytorch()
    # bench_ff_cuda()
