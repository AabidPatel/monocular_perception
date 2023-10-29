from vis_nav_game import Player, Action
import pygame
import cv2
import os
import datetime
import keyboard
import time

i = 0
s = [0,0,0,0]
counter = 0
key_events = []
capturing = False
# setattr(Action, "CAPTURE", 1 << 6)
#key_events = [[0, []], [1, ['F']], [2, ['F']], [3, ['F']], [4, ['F']], [5, ['F']], [6, ['F']], [7, ['F']], [8, ['F']], [9, ['F']], [10, ['F']], [11, ['F']], [12, ['F']], [13, ['F']], [14, ['F']], [15, ['F']], [16, ['F']], [17, ['F']], [18, ['F']], [19, ['F']], [20, ['F']], [21, ['F']], [22, ['F']], [23, ['F']], [24, ['F']], [25, ['F']], [26, ['F']], [27, ['F']], [28, ['F']], [29, ['F']], [30, ['F']], [31, ['F']], [32, ['F']], [33, ['F']], [34, ['F']], [35, ['F']], [36, ['F']], [37, ['F']], [38, ['F']], [39, ['F']], [40, ['F']], [41, ['F']], [42, ['F']], [43, ['F']], [44, ['F']], [45, ['F']], [46, ['F']], [47, ['F']], [48, ['F']], [49, ['F']], [50, ['F']], [51, ['F']], [52, ['F']], [53, ['F']], [54, ['F']], [55, ['F']], [56, ['F']], [57, ['F']], [58, ['F']], [59, ['F']], [60, ['F']], [61, ['F']], [62, ['F']], [63, ['F']], [64, ['F']], [65, ['F']], [66, ['F']], [67, ['F']], [68, ['F']], [69, ['F']], [70, ['F']], [71, ['F']], [72, ['F']], [73, ['F']], [74, ['F']], [75, ['F']], [76, ['F']], [77, ['F']], [78, ['F']], [79, ['F']], [80, ['F']], [81, ['F']], [82, ['F']], [83, ['F']], [84, ['F']], [85, ['F']]]
key_events = [[0, []], [1, []], [2, []], [3, []], [4, []], [5, []], [6, []], [7, ['F']], [8, ['F']], [9, ['F']], [10, ['F']], [11, ['F']], [12, ['F']], [13, ['F']], [14, ['F']], [15, ['F']], [16, ['F']], [17, ['F']], [18, ['F']], [19, ['F']], [20, ['F']], [21, ['F']], [22, ['F']], [23, ['F']], [24, ['F']], [25, ['F']], [26, ['F']], [27, ['F']], [28, ['F']], [29, ['F']], [30, ['F']], [31, ['F']], [32, ['F']], [33, ['F']], [34, ['F']], [35, ['F']], [36, ['F']], [37, ['F']], [38, ['F']], [39, ['F']], [40, ['F']], [41, ['F']], [42, ['F']], [43, ['F']], [44, ['F']], [45, ['F']], [46, ['F']], [47, ['F']], [48, ['F']], [49, ['F']], [50, ['F']], [51, ['F']], [52, ['F']], [53, ['F']], [54, ['F']], [55, ['F']], [56, ['F']], [57, ['F']], [58, ['F']], [59, ['F']], [60, ['F']], [61, ['F']], [62, ['F']], [63, ['F']], [64, ['F']], [65, ['F']], [66, ['F']], [67, ['F']], [68, ['F']], [69, ['F']], [70, ['F']], [71, ['F']], [72, ['F']], [73, ['F']], [74, ['F']], [75, ['F']], [76, ['F']], [77, ['F']], [78, ['F']], [79, ['F']], [80, ['F']], [81, ['F']], [82, ['F']], [83, ['F']], [84, ['F']], [85, ['F']], [86, ['F']], [87, ['F']], [88, ['F']], [89, ['F']], [90, []], [91, []], [92, []], [93, []], [94, []], [95, []], [96, []], [97, ['R']], [98, ['R']], [99, ['R']], [100, ['R']], [101, ['R']], [102, ['R']], [103, ['R']], [104, ['R']], [105, ['R']], [106, ['R']], [107, ['R']], [108, ['R']], [109, ['R']], [110, ['R']], [111, ['R']], [112, ['R']], [113, ['R']], [114, ['R']], [115, ['R']], [116, ['R']], [117, ['R']], [118, ['R']], [119, ['R']], [120, ['R']], [121, ['R']], [122, ['R']], [123, ['R']], [124, ['R']], [125, []], [126, []], [127, []], [128, ['R']], [129, ['R']], [130, ['R']], [131, ['R']], [132, ['R']], [133, ['R']], [134, ['R']], [135, ['R']], [136, []], [137, []], [138, []], [139, []], [140, []], [141, []], [142, []], [143, []], [144, []], [145, []], [146, []], [147, []], [148, []], [149, []], [150, []], [151, ['R']], [152, ['R']], [153, []], [154, []], [155, []], [156, []], [157, ['F']], [158, ['F']], [159, ['F']], [160, ['F']], [161, ['F']], [162, ['F']], [163, ['F']], [164, ['F']], [165, ['F']], [166, ['F']], [167, ['F']], [168, ['F']], [169, ['F']], [170, []], [171, []], [172, []], [173, []], [174, []], [175, []], [176, []], [177, ['R']], [178, ['R']], [179, ['R']], [180, ['R']], [181, ['R']], [182, ['R']], [183, ['R']], [184, ['R']], [185, ['R']], [186, ['R']], [187, ['R']], [188, ['R']], [189, ['R']], [190, ['R']], [191, ['R']], [192, ['R']], [193, ['R']], [194, ['R']], [195, ['R']], [196, ['R']], [197, ['R']], [198, ['R']], [199, ['R']], [200, ['R']], [201, ['R']], [202, ['R']], [203, ['R']], [204, ['R']], [205, ['R']], [206, ['R']], [207, ['R']], [208, ['R']], [209, ['R']], [210, ['R']], [211, ['R']], [212, ['R']], [213, []], [214, []], [215, []], [216, []], [217, []], [218, []], [219, []], [220, ['R']], [221, ['R']], [222, ['R']], [223, []], [224, []], [225, []], [226, []], [227, []], [228, []], [229, []], [230, []], [231, []], [232, []], [233, []], [234, ['F']], [235, ['F']], [236, ['F']], [237, ['F']], [238, ['F']], [239, ['F']], [240, ['F']], [241, ['F']], [242, ['F']], [243, ['F']], [244, ['F']], [245, ['F']], [246, ['F']], [247, ['F']], [248, ['F']], [249, ['F']], [250, ['F']], [251, ['F']], [252, ['F']], [253, ['F']], [254, ['F']], [255, []], [256, []], [257, []], [258, []], [259, []], [260, []], [261, []], [262, []], [263, []], [264, []], [265, []], [266, []], [267, []], [268, []], [269, []], [270, []], [271, []], [272, []], [273, []], [274, []], [275, []]]
target_path = []

target_ID = 275

for idx in range(target_ID+1):
    if(len(key_events[idx][1]) == 0):
        target_path.append('0')
    else:
        target_path.append(key_events[idx][1][0])
    if idx == target_ID:
        target_path.append('1')

print(target_path)

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.key_events = []
        global capturing

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        global key_events, j, capturing
        current_key_events = []
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if keyboard.is_pressed("r"):
                    capturing = not capturing
                    # print("r is pressed")
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        if capturing:
            for key, action in self.keymap.items():
                if keys[key]:
                    if action == Action.FORWARD:
                        current_key_events.append("F")
                    elif action == Action.BACKWARD:
                        current_key_events.append("B")
                    elif action == Action.RIGHT:
                        current_key_events.append("R")
                    elif action == Action.LEFT:
                        current_key_events.append("L")
                    elif action == Action.QUIT:
                        current_key_events.append("Q")

            key_events.append([i, current_key_events])

        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h / 2), 0), (int(h / 2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w / 2)), (h, int(w / 2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(
            concat_img,
            "Front View",
            (h_offset, w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Right View",
            (int(h / 2) + h_offset, w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Back View",
            (h_offset, int(w / 2) + w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )
        cv2.putText(
            concat_img,
            "Left View",
            (int(h / 2) + h_offset, int(w / 2) + w_offset),
            font,
            size,
            color,
            stroke,
            line,
        )

        cv2.imshow(f"KeyboardPlayer:target_images", concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def get_k_matrix(self, images):
        k = super(KeyboardPlayerPyGame, self).get_camera_intrinsic_matrix()
        print(k)
    
    def nav(self, c):
        print(c)
        print(target_path[c])
        if target_path[c] == '0':
            self.last_act = Action.IDLE
        elif(target_path[c] == 'F'):
            self.last_act = Action.FORWARD
        elif(target_path[c] == 'B'):
            self.last_act = Action.BACKWARD
        elif(target_path[c] == 'R'):
            self.last_act = Action.RIGHT
        elif(target_path[c] == 'L'):
            self.last_act = Action.LEFT
        elif(counter == len(target_path)-1):
            self.last_act = Action.CHECKIN
            #self.last_act = Action.QUIT          
            print("IDLE")

    def see(self, fpv):
        global counter
        global i, s
        if fpv is None or len(fpv.shape) < 3:
            return

        if capturing:
            if not os.path.exists("./images"):
                os.mkdir("./images")

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            # (height,width,Number of colors) -> (width, height)
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, "RGB")

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

        cv2.imshow("fpv", fpv)
        key = cv2.waitKey(1) & 0xFF

        if capturing:
            file_path = f"./images/{i}.jpg"
            cv2.imwrite(file_path, fpv)
            i += 1

        s = self.get_state()
        if(s is not None):
            print("phase = ", s[1].value)

            if(s[1].value == 2):
                if counter < len(target_path):
                    self.nav(counter)
                    counter += 1
                print("counter = ", counter )

    

        

if __name__ == "__main__":
    import vis_nav_game

    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
    print(key_events)
