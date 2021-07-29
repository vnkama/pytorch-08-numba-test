import numpy as np
from numba import cuda          # Библиотека Nvidia для работы с GPU
import time


def init():
    A = np.array(
            [
                [
                    [1, 2, -3],
                    [4, -5, 6],
                    [-7, 8, 9],
                ],
                [
                    [-11, 21, 31],
                    [41, -51, 61],
                    [71, 81, -91],
                ]
            ],
            dtype=np.double
    )

    B = np.array(
            [
                [
                    [10, -20, 0.30],
                    [-40, 0.50, -60],
                    [0.70, -80, 90],
                ],
                [
                    [-0.10, -20, -30],
                    [0.40, 50, 60],
                    [-0.70, -80, -90],
                ]
            ],
            dtype=np.double
    )
    return A, B


def init_1d(size):
    return np.zeros(shape=size)

########################################################

@cuda.jit
def test_1d_1block__cuda_jit(A):
    # thread index in block
    tx = cuda.threadIdx.x

    # Block id in a 1D grid
    # block
    ty = cuda.blockIdx.x
    array_size = A.shape[0]


    # Block width, i.e. number of threads per block
    threads_per_block = cuda.blockDim.x


    # Compute flattened index inside the array
    i = tx + ty * threads_per_block

    if i < array_size:  # Check array boundaries
        A[i] += 1



#
# массив размером 1d
# строго один одномерный блок
# максимальный разме 1-мерного блока = 1024 thread's
#
def test_1d_1block():
    array_size = 1000000
    A = np.zeros(shape=array_size)

    t1 = time.perf_counter()

    #d_A = cuda.to_device(A)

    if array_size <= 256:
        blocks_per_grid = 1
        threads_per_block = array_size
    else:
        threads_per_block = 256
        blocks_per_grid = ((array_size-1) // threads_per_block)+1

    print(f'blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}')

    for _ in range(10000):
        test_1d_1block__cuda_jit[blocks_per_grid, threads_per_block](A)


    #A = d_A.copy_to_host()
    t2 = time.perf_counter()
    print(t2-t1)

    print('sum A:', np.sum(A))

########################################################
########################################################

@cuda.jit
def test_1d_Nblock__cuda_jit(A):
    # thread index in block
    tx = cuda.threadIdx.x

    # block index in grid
    ty = cuda.blockIdx.x
    array_size = A.shape[0]


    # Block width, i.e. number of threads per block
    threads_per_block = cuda.blockDim.x
    #print('tx:', tx, 'ty:', ty, 'threads_per_block:', threads_per_block)


    # Compute flattened index inside the array
    i = tx + ty * threads_per_block

    if i < array_size:  # Check array boundaries
        A[i] += 1

#
# входной массив: 1d, максимально тестил на размере массива 1.000.000
# в сетке - несколько блоков
# максимальный размер 1-мерного блока = 1024 thread's
#
def test_1d_Nblock():
    array_size = 100000
    A  = init_1d(array_size)

    if array_size <= 256:
        blocks_per_grid = 1
        threads_per_block = array_size
    else:
        threads_per_block = 1024
        blocks_per_grid = ((array_size-1) // threads_per_block)+1

    print(f'blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}')

    loop_count = 10000

    t1 = time.perf_counter()

    for _ in range(loop_count):
        test_1d_Nblock__cuda_jit[blocks_per_grid, threads_per_block](A)

    t2 = time.perf_counter()


    print(t2-t1)
    print('sum A:', np.sum(A) / loop_count)


########################################################
########################################################

@cuda.jit
def test_2d_1block__cuda_jit(A):
    # thread index in block
    tx = cuda.threadIdx.x

    # block index in grid
    ty = cuda.blockIdx.x
    array_size = A.shape[0]


    # Block width, i.e. number of threads per block
    threads_per_block = cuda.blockDim.x
    #print('tx:', tx, 'ty:', ty, 'threads_per_block:', threads_per_block)


    # Compute flattened index inside the array
    i = tx + ty * threads_per_block

    if i < array_size:  # Check array boundaries
        A[i] += 1

#
# входной массив: 2d,
# в сетке - 1 блок
# максимальный размер 2-мерного блока = 1024  x 1024 thread's
#
def test_2d_1block__cuda_jit():
    array_size = 100000
    A  = init_1d(array_size)

    if array_size <= 256:
        blocks_per_grid = 1
        threads_per_block = array_size
    else:
        threads_per_block = 1024
        blocks_per_grid = ((array_size-1) // threads_per_block)+1

    print(f'blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}')

    loop_count = 10000

    t1 = time.perf_counter()

    for _ in range(loop_count):
        test_1d_Nblock__cuda_jit[blocks_per_grid, threads_per_block](A)

    t2 = time.perf_counter()


    print(t2-t1)
    print('sum A:', np.sum(A) / loop_count)

########################################################
########################################################

def main():
    test_1d_1block()



if __name__ == '__main__':
    main()
