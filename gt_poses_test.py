import os
import numpy as np
import cv2
from tqdm import tqdm


def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

data_dir = "KITTI_sequence_2"
gt_poses = _load_poses(os.path.join(data_dir,"poses.txt"))

for i, gt_pose in enumerate(gt_poses):
    print("i = ", i)
    print("shape of gt_poses = ", len(gt_poses))