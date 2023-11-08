import os
import numpy as np
import cv2
from lib.visualization import plotting

i = 0

gt_list = [[0, []], [1, []], [2, []], [3, []], [4, []], [5, []], [6, []], [7, ['F']], [8, ['F']], [9, ['F']], [10, ['F']], [11, ['F']], [12, ['F']], [13, ['F']], [14, ['F']], [15, ['F']], [16, ['F']], [17, ['F']], [18, ['F']], [19, ['F']], [20, ['F']], [21, ['F']], [22, ['F']], [23, ['F']], [24, ['F']], [25, ['F']], [26, ['F']], [27, ['F']], [28, ['F']], [29, ['F']], [30, ['F']], [31, ['F']], [32, ['F']], [33, ['F']], [34, ['F']], [35, ['F']], [36, ['F']], [37, ['F']], [38, ['F']], [39, ['F']], [40, ['F']], [41, ['F']], [42, ['F']], [43, ['F']], [44, ['F']], [45, ['F']], [46, ['F']], [47, ['F']], [48, ['F']], [49, ['F']], [50, ['F']], [51, ['F']], [52, ['F']], [53, ['F']], [54, ['F']], [55, ['F']], [56, ['F']], [57, ['F']], [58, ['F']], [59, ['F']], [60, ['F']], [61, ['F']], [62, ['F']], [63, ['F']], [64, ['F']], [65, ['F']], [66, ['F']], [67, ['F']], [68, ['F']], [69, ['F']], [70, ['F']], [71, ['F']], [72, ['F']], [73, ['F']], [74, ['F']], [75, ['F']], [76, ['F']], [77, ['F']], [78, ['F']], [79, ['F']], [80, ['F']], [81, ['F']], [82, ['F']], [83, ['F']], [84, ['F']], [85, ['F']], [86, ['F']], [87, ['F']], [88, ['F']], [89, ['F']], [90, []], [91, []], [92, []], [93, []], [94, []], [95, []], [96, []], [97, ['R']], [98, ['R']], [99, ['R']], [100, ['R']], [101, ['R']], [102, ['R']], [103, ['R']], [104, ['R']], [105, ['R']], [106, ['R']], [107, ['R']], [108, ['R']], [109, ['R']], [110, ['R']], [111, ['R']], [112, ['R']], [113, ['R']], [114, ['R']], [115, ['R']], [116, ['R']], [117, ['R']], [118, ['R']], [119, ['R']], [120, ['R']], [121, ['R']], [122, ['R']], [123, ['R']], [124, ['R']], [125, []], [126, []], [127, []], [128, ['R']], [129, ['R']], [130, ['R']], [131, ['R']], [132, ['R']], [133, ['R']], [134, ['R']], [135, ['R']], [136, []], [137, []], [138, []], [139, []], [140, []], [141, []], [142, []], [143, []], [144, []], [145, []], [146, []], [147, []], [148, []], [149, []], [150, []], [151, ['R']], [152, ['R']], [153, []], [154, []], [155, []], [156, []], [157, ['F']], [158, ['F']], [159, ['F']], [160, ['F']], [161, ['F']], [162, ['F']], [163, ['F']], [164, ['F']], [165, ['F']], [166, ['F']], [167, ['F']], [168, ['F']], [169, ['F']], [170, []], [171, []], [172, []], [173, []], [174, []], [175, []], [176, []], [177, ['R']], [178, ['R']], [179, ['R']], [180, ['R']], [181, ['R']], [182, ['R']], [183, ['R']], [184, ['R']], [185, ['R']], [186, ['R']], [187, ['R']], [188, ['R']], [189, ['R']], [190, ['R']], [191, ['R']], [192, ['R']], [193, ['R']], [194, ['R']], [195, ['R']], [196, ['R']], [197, ['R']], [198, ['R']], [199, ['R']], [200, ['R']], [201, ['R']], [202, ['R']], [203, ['R']], [204, ['R']], [205, ['R']], [206, ['R']], [207, ['R']], [208, ['R']], [209, ['R']], [210, ['R']], [211, ['R']], [212, ['R']], [213, []], [214, []], [215, []], [216, []], [217, []], [218, []], [219, []], [220, ['R']], [221, ['R']], [222, ['R']], [223, []], [224, []], [225, []], [226, []], [227, []], [228, []], [229, []], [230, []], [231, []], [232, []], [233, []], [234, ['F']], [235, ['F']], [236, ['F']], [237, ['F']], [238, ['F']], [239, ['F']], [240, ['F']], [241, ['F']], [242, ['F']], [243, ['F']], [244, ['F']], [245, ['F']], [246, ['F']], [247, ['F']], [248, ['F']], [249, ['F']], [250, ['F']], [251, ['F']], [252, ['F']], [253, ['F']], [254, ['F']], [255, []], [256, []], [257, []], [258, []], [259, []], [260, []], [261, []], [262, []], [263, []], [264, []], [265, []], [266, []], [267, []], [268, []], [269, []], [270, []], [271, []], [272, []], [273, []], [274, []], [275, []]]
#gt_list = [[0, []], [1, []], [2, []], [3, []], [4, []], [5, []], [6, ['F']], [7, ['F']], [8, ['F']], [9, ['F']], [10, ['F']], [11, ['F']], [12, ['F']], [13, ['F']], [14, ['F']], [15, ['F']], [16, ['F']], [17, ['F']], [18, ['F']], [19, ['F']], [20, ['F']], [21, ['F']], [22, ['F']], [23, ['F']], [24, ['F']], [25, ['F']], [26, ['F']], [27, ['F']], [28, ['F']], [29, ['F']], [30, ['F']], [31, ['F']], [32, ['F']], [33, ['F']], [34, ['F']], [35, ['F']], [36, ['F']], [37, ['F']], [38, ['F']], [39, ['F']], [40, ['F']], [41, ['F']], [42, ['F']], [43, ['F']], [44, ['F']], [45, ['F']], [46, ['F']], [47, ['F']], [48, ['F']], [49, ['F']], [50, ['F']], [51, ['F']], [52, ['F']], [53, ['F']], [54, ['F']], [55, ['F']], [56, ['F']], [57, ['F']], [58, ['F']], [59, ['F']], [60, ['F']], [61, ['F']], [62, ['F']], [63, ['F']], [64, ['F']], [65, ['F']], [66, ['F']], [67, ['F']], [68, ['F']], [69, ['F']], [70, ['F']], [71, ['F']], [72, ['F']], [73, ['F']], [74, ['F']], [75, ['F']], [76, ['F']], [77, ['F']], [78, ['F']], [79, ['F']], [80, ['F']], [81, ['F']], [82, ['F']], [83, ['F']], [84, ['F']], [85, ['F']], [86, ['F']], [87, ['F']], [88, ['F']], [89, ['F']], [90, ['F']], [91, []], [92, []], [93, []]]

class VisualOdometry:
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib()
        # self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "images_6"))
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

    def error_correction(self, transformation_mat, prev_transformation_mat, j):
        global gt, rotated

        est_x = transformation_mat[0, 3]
        est_y = transformation_mat[2, 3]
        est_z = transformation_mat[1, 3]
        est_r = transformation_mat[0:3, 0:3]

        prev_x = prev_transformation_mat[0, 3]
        prev_y  = prev_transformation_mat[2, 3]
        prev_z  = prev_transformation_mat[1, 3]
        prev_r = prev_transformation_mat[0:3, 0:3]

        Id = gt_list[j - 1][0]
        if len(gt_list[j - 1][1]) == 0:
            gt = "S"
        else:
            gt = gt_list[j - 1][1][0]

        if gt == "S":
            corrected_r = prev_r
            est_z = prev_z
            if est_x > prev_x or est_x < prev_x:
                est_x = prev_x
            if est_y > prev_y or est_y < prev_y:
                est_y = prev_y

        elif gt == "F":
            corrected_r = prev_r
            if rotated > 0:
                """
                est_z = 0
            if est_x > prev_x:
                est_x = prev_x
            if est_y < prev_y:
                est_y = prev_y + (prev_y - est_y)
                """

        elif gt == "R":
            #corrected_r = est_r
            rotated += 1
            est_z = prev_z
            if est_x > prev_x or est_x < prev_x:
                est_x = prev_x
            if est_y > prev_y or est_y < prev_y:
                est_y = prev_y
            
            corrected_r = est_r

        elif gt == "L":
            corrected_r = est_r


        #transf = np.eye(4)
        transf = np.eye(4, dtype=np.float64)
        transf[0, 3] = est_x
        transf[1, 3] = est_z
        transf[2, 3] = est_y
        transf[0:3, 0:3] = corrected_r

        return transf

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
        # Decompose the essential matrix
        # R1, R2, t = cv2.decomposeEssentialMat(E)

        _, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)

        t = np.squeeze(t)

        # relative_scale = self.sum_z_cal_relative_scale(R, t, q1, q2)
        # t = t * relative_scale

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

    for i, init_pose in enumerate(cur_poses):
        if i == 0:
            cur_pose = identity_array
            prev_trans = cur_pose
        else:
            q1, q2 = vo.get_matches(i)
            trans = vo.get_pose(q1, q2)
            transf = vo.error_correction(trans, prev_trans, i)
            inv_transf = np.linalg.inv(transf)
            #print("C = ", cur_pose[0:3, 3])
            #print("I = ", inv_transf[0:3, 3])
            if (cur_pose[0:3, 3] != transf[0:3, 3]).all():
                cur_pose = np.matmul(cur_pose, inv_transf)
                prev_trans = cur_pose
            
            #prev_trans = cur_pose

        print("cur_pose = ", cur_pose)
        x = cur_pose[0, 3]
        y = cur_pose[2, 3]

        print("x = ", x)
        print("y = ", y)

        estimated_path.append((x, y))
    print("estimated_path = ", estimated_path)

    plotting.visualize_paths(
        estimated_path,
        estimated_path,
        "Visual Odometry",
        file_out=os.path.basename(data_dir) + ".html",
    )


if __name__ == "__main__":
    main()
