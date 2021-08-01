import numpy as np
from numba import cuda          # Библиотека Nvidia для работы с GPU
import numba as nb
import time

#
#
#
def init_const_arrays():
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


    # результат перемножния первой пары матриц
    #
    #   [[-72.1, 221, -389.7],
    #   [244.2, -562.5, 841.2],
    #   [-383.7,	-576, 327.9]]


    # результат перемножния второй пары матриц
    #  [[-12.2	-1210	-1200],
    # [-67.2	-8250	-9780],
    # [89	9910	10920]]


    return A, B


def init_1d(size):
    return np.zeros(shape=size)

########################################################
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
    array_size = 100000
    loop_count = 100
    A = np.zeros(shape=array_size)


    t1 = time.perf_counter()
    d_A = cuda.to_device(A)

    if array_size <= 256:
        blocks_per_grid = 1
        threads_per_block = array_size
    else:
        threads_per_block = 256
        blocks_per_grid = ((array_size-1) // threads_per_block) + 1

    print(f'blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}')

    for _ in range(loop_count):
        test_1d_1block__cuda_jit[blocks_per_grid, threads_per_block](d_A)


    A = d_A.copy_to_host()

    t2 = time.perf_counter()
    print((t2-t1)/loop_count/array_size*1e6)

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
def matmul_f1__cuda_jit(A, B, C):

    tpb = (4, 4)
    bpg = cuda.gridDim.x

    # thread index in block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    #x, y = cuda.grid(2)


    # индекс матрицы 3x3
    bx = cuda.blockIdx.x
    array_size = A.shape[0]


    if tx >= 3 or ty >= 3:
        return

    # sA = cuda.shared.array(shape=tpb, dtype=nb.float32)
    # sB = cuda.shared.array(shape=tpb, dtype=nb.float32)
    # sA[tx, ty] = A[bx, tx, ty]
    # sB[tx, ty] = B[bx, tx, ty]
    # C[bx, tx, ty] = sA[tx, 0] * sB[0, ty] + sA[tx, 1] * sB[1, ty] + sA[tx, 2] * sB[2, ty]

    A_mx33 = A[bx]
    B_mx33 = B[bx]
    C[bx, tx, ty] = A_mx33[tx, 0] * B_mx33[0, ty] + A_mx33[tx, 1] * B_mx33[1, ty] + A_mx33[tx, 2] * B_mx33[2, ty]




#
# тестируем умножение группы матриц
# матрицы 3x3, их несколько, потому входной массив (N,3,3)
# в этой функции 1 матрица 3x3 соответствует 1 блоку
#
# в сетке - N блоков размером минимум 3x3, но по факту делаем 4x4 или
# максимальный размер 2-мерного блока = 1024  x 1024 thread's
#
def test_matmul_f1():
    array_size = 10000
    loop_count = 1000

    # 1000 10000, 0.026мкс на операцию

    A = np.random.uniform(low=-100, high=100, size=(array_size, 3, 3))
    B = np.random.uniform(low=-100, high=100, size=(array_size, 3, 3))
    A = np.float32(A)
    B = np.float32(B)


    #A, B = init_const_arrays()

    C = np.zeros_like(A)
    #array_size = A.shape[0]

    t1 = time.perf_counter()

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    blocks_per_grid = array_size
    threads_per_block = (4, 4)

    #print(f'blocks_per_grid: {blocks_per_grid}, threads_per_block: {threads_per_block}')


    for _ in range(loop_count):
        matmul_f1__cuda_jit[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    A = d_A.copy_to_host()
    B = d_B.copy_to_host()
    C = d_C.copy_to_host()

    t2 = time.perf_counter()

    print(t2-t1)
    print('мкс:', (t2-t1) / loop_count / array_size * 1e6)

    # print('A:\n', A)
    # print('B:\n', B)
    # print('C:\n', C)

########################################################
########################################################

def main():
    test_matmul_f1()



if __name__ == '__main__':
    main()
