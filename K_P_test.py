import os
import numpy as np
import cv2

def load_calib(filepath):

    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P


data_dir = "KITTI_sequence_2"

K, P = load_calib(os.path.join(data_dir, 'calib.txt'))

#P = np.array([[92, 0, 160, 0], [0, 92, 120, 0], [0, 0, 1, 0]])
#K = np.array([[92, 0, 160], [0, 92, 120], [0, 0, 1]])

print("K = ", K)
print("P = ", P)
