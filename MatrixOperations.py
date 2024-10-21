import time
import numpy as np

# Matrix Operations, implemented for greater understanding. Includes Example arrays

def scalar_mul_row(scalar, row):
    ret = np.zeros(len(row))
    for i in range(len(row)):
        ret[i] = row[i] * scalar
    return ret
        
def subtract_rows(row_1, row_2):
    ret = np.zeros(len(row_1))
    for i in range(len(row_1)):
        ret[i] = row_1[i] - row_2[i]
    return ret

def sort_mat(mat):
    row_len_list = []
    for row in mat:
        row_len_list.append((find_first_num(row), row))
    row_len_list.sort(key=lambda a:a[0])
    ret = np.zeros(mat.shape)
    for i in range(len(mat)):
        ret[i] = row_len_list[i][1]
    return ret
    
        
def find_first_num(row):
    for i in range(len(row)):
        if row[i] != 0:
            return i
    #all 0's
    return -1

def mul_matrix(mat_1, mat_2):
    mat_1_rows, mat_1_columns = mat_1.shape
    mat_2_rows, mat_2_columns = mat_2.shape
    if mat_1_columns != mat_2_rows:
        raise Exception("Invalid Matrices Multiplied")
    ret = np.zeros((mat_1_rows, mat_2_columns))
    for i in range(mat_1_rows):
        for j in range(mat_2_columns):
            for k in range(mat_1_columns):
                ret[i][j] += mat_1[i][k] * mat_2[k][j]
    return ret

def add_matrix(mat_1, mat_2):
    mat_1_rows, mat_1_columns = mat_1.shape
    mat_2_rows, mat_2_columns = mat_2.shape
    ret = np.zeros((mat_1_rows, mat_1_columns))
    if mat_1_rows != mat_2_rows and mat_1_columns != mat_2_columns:
        raise Exception("Invalid Matrices Added")
    for i in range(mat_1_rows):
        for j in range(mat_1_columns):
            ret[i][j] = mat_1[i][j] + mat_2[i][j]
    return ret

def solve_matrix_gaussian(mat):
    mat_rows, mat_columns = mat.shape
    ret = np.copy(mat)  # Work on a copy of the matrix
    
    #Forward pass
    for i in range(mat_rows):
        first_num_index = find_first_num(ret[i])
        first_num = ret[i][first_num_index]
        
        if(first_num_index != -1):
            norm_num = 1.0 / first_num
            ret[i] = scalar_mul_row(norm_num, ret[i])
            for j in range(mat_rows - i - 1):
                if(ret[i+j+1][first_num_index] != -1):
                    sub_by = scalar_mul_row(ret[i+j+1][first_num_index] / ret[i][first_num_index], ret[i])
                    ret[i + j + 1] = subtract_rows(ret[i+j+1], sub_by) 
        ret = sort_mat(ret)
        
    #Back substitution
    for i in range(mat_rows):
        index = mat_rows - i - 1
        first_num_index = find_first_num(ret[index])
        if(first_num_index != -1):
            for j in range(mat_rows - i - 1):
                ret[j] = subtract_rows(ret[j], scalar_mul_row(ret[j][first_num_index], ret[index]))

    return ret [:, -1]      
    
"""
Takes and solves a tridiagonal matrix, Ax=d
a: Subdiagonal
b: Main diagonal
c: Superdiagonal
"""
def solve_matrix_thomas(a, b, c, d):    
    n = len(d)
    
    # Forward elimination
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-12:
            raise ValueError("Numerical instability: Main diagonal element is too small.")
        w = a[i] / b[i - 1]  # Scale factor
        b[i] = b[i] - w * c[i - 1]  # Modify the main diagonal
        d[i] = d[i] - w * d[i - 1]  # Modify the result vector

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]  # Solve last unknown 

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]  # Solve for the other unknowns
    
    return x

"""
More for testing. Solving the 1d diffusion equation will not need it
"""
def not_inplace_solve_matrix_thomas(a, b, c, d):
    return solve_matrix_thomas(a, b.copy(), c.copy(), d)
    
  
def extract_diagonals(tridag_mat):
    n = len(tridag_mat)

    a = np.zeros(n, dtype="float64")  # Subdiagonal a[0] == 0
    b = np.zeros(n, dtype="float64")  # Main diagonal
    c = np.zeros(n, dtype="float64")  # Superdiagonal c[-1] == 0
    d = np.zeros(n, dtype="float64")  # Right-hand side vector, solutions

    for i in range(n):
        b[i] = tridag_mat[i][i]          # Main diagonal
        d[i] = tridag_mat[i][-1]         # Right-hand side
        if i > 0:
            a[i] = tridag_mat[i][i-1]    # Subdiagonal
        if i < n - 1:
            c[i] = tridag_mat[i][i+1]    # Superdiagonal

    return a, b, c, d

def main():
    matrix_1 = np.array([[1, 2, 4 , 2],
                     [2, 4, 2, 2],
                     [2, 5, 1, 3]])

    matrix_2 = np.array([[3, 3, 6 , 4],
                        [1, 7, 2, 1],
                        [20, 9, 1, 4]])

    matrix_3 = np.array([[5, 2, 3],
                        [2, 0, 2],
                        [2, 5, 1],
                        [7, 5, 2]])

    tri_matrix = np.array([[2, -1, 0, 0, 1],
                        [-1, 2, -1, 0, 0],
                        [0, -1, 2, -1, 0],
                        [0, 0, -1, 2, 1]])
    
    a = np.array([0, 2, 3, 4], dtype="float64")  # Subdiagonal 
    b = np.array([5, 6, 7, 8], dtype="float64")  # Main diagonal 
    c = np.array([1, 2, 3, 0], dtype="float64")  # Superdiagonal 
    d = np.array([5, 6, 7, 8], dtype="float64")  # Right-hand side 

    # Solve the system
    x = solve_matrix_thomas(a, b, c, d)

    print("Solution:", x)

    verif_with_gauss = np.array([[4, 1, 0, 0, 15],
                        [1, 4, 1, 0, 8],
                        [0, 1, 4, 1, 8],
                        [0, 0, 1, 3, 15]], dtype="float64")

    a = np.array([0, 1, 1, 1], dtype="float64")  # Subdiagonal 
    b = np.array([4, 4, 4, 3], dtype="float64")  # Main diagonal 
    c = np.array([1, 1, 1, 0], dtype="float64")  # Superdiagonal 
    d = np.array([15, 8, 8, 15], dtype="float64")  # Right-hand side 

    # Solve the system
    x = solve_matrix_thomas(a, b, c, d)

    print("Solution:", x)

    print("Solution 2: ", solve_matrix_gaussian(verif_with_gauss))

    #print(f"Thomas time: {thomas_time:.5f} seconds")

        
if __name__ == "__main__":
    main()