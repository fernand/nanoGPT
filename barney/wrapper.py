import time
import ctypes as ct

import torch
import torch.nn.functional as F
torch.set_num_threads(1)

def timeit(f):
    n = 1000
    t1 = time.perf_counter_ns()
    for _ in range(n):
        f()
    t2 = time.perf_counter_ns()
    print('exec time', (t2 - t1) / (1000 * n), 'us')

def ptr(A: torch.Tensor) -> ct.c_void_p:
    assert A is not None
    return ct.c_void_p(A.data.data_ptr())


def test():
    X = torch.rand((768,))
    E1 = torch.rand((64, 768))
    E2 = torch.rand((768, 64))
    Xo = torch.zeros((768,))
    lib.expert_forward(ptr(X), ptr(E1), ptr(E2), ptr(Xo))
    Xo = Xo.numpy()
    Xo_ref = E2.matmul(F.relu(E1.matmul(X))).numpy()
    assert all([abs((xo - xo_ref) / xo_ref) < 1e-5 for (xo, xo_ref) in zip(Xo, Xo_ref)])

def bench():
    X = torch.rand((768,))
    E1 = torch.rand((64, 768))
    E2 = torch.rand((768, 64))
    Xo = torch.zeros((768,))
    print('Pytorch speed: ', end='')
    timeit(lambda: E2.matmul(F.relu(E1.matmul(X))))
    print('C lib speed: ', end='')
    timeit(lambda: lib.expert_forward(ptr(X), ptr(E1), ptr(E2), ptr(Xo)))

if __name__ == '__main__':
    lib = ct.cdll.LoadLibrary('/home/fernand/nanoGPT/barney/lib.so')
    test()
    bench()

