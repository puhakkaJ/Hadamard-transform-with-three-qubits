import numpy as np
from math import sqrt


def kronecker_product(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    result = [[0] * (cols_A * cols_B) for _ in range(rows_A * rows_B)]

    for i in range(rows_A):
        for j in range(cols_A):
            for k in range(rows_B):
                for l in range(cols_B):
                    result[i * rows_B + k][j * cols_B + l] = A[i][j] * B[k][l]

    return result



# 1. a) Show that H ⊗ H ⊗ H = H^[3|2]H^[3|1]H^[3|0].

## Definitions

H = (1/(sqrt(2)))*np.array([[1, 1],
                            [1, -1]])

H32 = (1/(sqrt(2)))*np.array([   [1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 1, 0, 0, 0, -1, 0, 0],
                        [0, 0, 1, 0, 0, 0, -1, 0],
                        [0, 0, 0, 1, 0, 0, 0, -1]])

H31 = (1/(sqrt(2)))*np.array([   [1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, -1, 0, 0, 0, 0, 0],
                        [0, 1, 0, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0, -1, 0],
                        [0, 0, 0, 0, 0, 1, 0, -1]])

H30 = (1/(sqrt(2)))*np.array([   [1, 1, 0, 0, 0, 0, 0, 0],
                        [1, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, -1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, -1]])



## multiplication H^[3|2]H^[3|1]H^[3|0]

first_multiplication = np.matmul(H31, H30)
multiplication_result = np.matmul(H32, first_multiplication)

multiplication_result_common = multiplication_result / (1/(2*sqrt(2))) # common factor 1/2 out

print("**RIGHT HAND SIDE:\n")
print("H^[3|2]H^[3|1]H^[3|0] =", multiplication_result, "\n")
print("= 1/(2*sqrt(2))*", multiplication_result_common, "\n")

## Computing the Kronecker product H ⊗ H ⊗ H

# Using numpy
first_kronecker_product = np.kron(H, H)
kronecker_product_result = np.kron(first_kronecker_product,H)

# Using direct defenitions in the sake of learning
first_kronecker_product_definitions = kronecker_product(H, H)
kronecker_product_result_definitions  = kronecker_product(H, first_kronecker_product_definitions)

kronecker_product_result_common = kronecker_product_result / (1/(2*sqrt(2))) # common factor 1/2 out

print("**LEFT HAND SIDE:\n")
print("H ⊗ H ⊗ H =",kronecker_product_result, "\n")
print("Using the definitions - H ⊗ H ⊗ H =",kronecker_product_result_definitions, "\n")
print("= 1/(2*sqrt(2))*", kronecker_product_result_common)

## Equality between the right and left hand side
print("\nEquality between the right and left hand side = ", kronecker_product_result == multiplication_result)
