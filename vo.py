import os
import numpy as np
import cv2
from lib.visualization import plotting

i = 0


class VisualOdometry:
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib()
        # self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "images_4"))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params
        )

    @staticmethod
    def _load_calib():
        P = np.array([[92, 0, 160, 0], [0, 92, 120, 0], [0, 0, 1, 0]])
        K = np.array([[92, 0, 160], [0, 92, 120], [0, 0, 1]])
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=" ")
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [
            os.path.join(filepath, file)
            for file in sorted(os.listdir(filepath), key=lambda x: int(x.split(".")[0]))
        ]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_target_match():
        pass

    def get_matches(self, i):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(
            matchColor=-1,  # draw matches in green color
            singlePointColor=None,
            matchesMask=None,  # draw only inliers
            flags=2,
        )

        img3 = cv2.drawMatches(
            self.images[i], kp1, self.images[i - 1], kp2, good, None, **draw_params
        )
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def sum_z_cal_relative_scale(self, R, t, q1, q2):
        # Get the transformation matrix
        T = self._form_transf(R, t)
        # Make the projection matrix

        P2 = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(self.P, P2, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Form point pairs and calculate the relative scale
        a = np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
        b = np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)

        relative_scale = np.mean(a / b)

        if np.isnan(relative_scale):
            relative_scale = 1
        elif np.isinf(relative_scale):
            relative_scale = 1

        return relative_scale

    def decomp_essential_mat(self, E, q1, q2):
        global j
        # Decompose the essential matrix
        # R1, R2, t = cv2.decomposeEssentialMat(E)

        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)

        t = np.squeeze(t)

        T = self._form_transf(R, t)
        # Make the projection matrix

        P2 = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(self.P, P2, q1.T, q2.T)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Form point pairs and calculate the relative scale
        a = np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
        b = np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)

        relative_scale = np.mean(a / b)

        if np.isnan(relative_scale):
            relative_scale = 0.999999999
        elif np.isinf(relative_scale):
            relative_scale = 0.999999999

        relative_scale = self.sum_z_cal_relative_scale(R, t, q1, q2)
        t = t * relative_scale

        return [R, t]


def main():
    data_dir = "KITTI_sequence_2"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)

    # play_trip(vo.images)  # Comment out to not play the trip
    images = vo.images
    gt_path = []
    estimated_path = []
    cur_poses = []

    identity_array = np.identity(4)
    for i in range(len(images)):
        cur_poses.append(identity_array)

    for i, init_pose in enumerate(cur_poses):
        if i == 0:
            cur_pose = init_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    print("estimated_path = ", estimated_path)

    plotting.visualize_paths(
        estimated_path,
        estimated_path,
        "Visual Odometry",
        file_out=os.path.basename(data_dir) + ".html",
    )


if __name__ == "__main__":
    main()
