import numpy as np


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


@cuda.jit
def increment_by_one(an_array):
    pass




def main():
    A, B = init()

    threadsperblock = 32
    blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock
    increment_by_one[blockspergrid, threadsperblock](an_array)


main()