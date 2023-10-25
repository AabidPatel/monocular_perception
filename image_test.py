import os
import numpy as np
import cv2

def _load_images(filepath):
    # image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath), key=lambda x: int(x.split('.')[0]))]
    return image_paths, [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

data_dir = "KITTI_sequence_2"
image_paths, images = _load_images(os.path.join(data_dir,"images"))

print(image_paths)
print(len(images))