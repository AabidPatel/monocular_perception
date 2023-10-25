""" TEST 1

import numpy as np

A = np.array([[1, 0], [0, 1]])
B = np.array([[5, 6], [7, 8]])

C = np.matmul(A, B)

print(C)
"""

""" TEST 2 """

import numpy as np

identity_array = np.identity(4)
num_times = 3
appended_list = []

for i in range(num_times):
    appended_list.append(identity_array)

print("Appended Array: ", appended_list)
print("shape of array : ", len(appended_list))

for i, j in enumerate(appended_list):
    print("i = ", i)
    print("gt_pose: ")
    print("x = ", j[0, 3])
    print("y = ", j[2, 3])