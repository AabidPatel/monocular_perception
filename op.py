from vis_nav_game import Player, Action
import pygame
import cv2
import keyboard
import os

i = 0
counter = 0
target_loc = 0
capturing = False
navigating = False
key_events = []
target_images = []
frameID_des = []

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
        global target_images
        targets = self.get_target_images()

        if targets is None or len(targets) <= 0:
            return

        for target in range(len(targets)):
            #  [f_view , r_view , b_view , l_view] 
      #      file_path = f"./target/{target}.jpg"
      #      cv2.imwrite(file_path,targets[target] )
            target_images.append(targets[target])

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

        cv2.putText(concat_img,"Front View",(h_offset, w_offset),font,size,color,stroke,line,)
        cv2.putText(concat_img,"Right View",(int(h / 2) + h_offset, w_offset),font,size,color,stroke,line,)
        cv2.putText(concat_img,"Back View",(h_offset, int(w / 2) + w_offset),font,size,color,stroke,line,)
        cv2.putText(concat_img,"Left View",(int(h / 2) + h_offset, int(w / 2) + w_offset),font,size,color,stroke,line,)

        cv2.imshow(f"KeyboardPlayer:target_images", concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def get_k(self):
        k = self.get_camera_intrinsic_matrix()
        return k


    def get_target_location(self, target_imgs):
        global key_events, frameID_des
        print("Fetching Target Location ....")
        orb = cv2.ORB_create(500)
    # print(len(goal))
        t = []
        des_targets = []
        target_match = [[], [], [], []]
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params) 
        #K = self.get_k()

        for j in range(len(target_imgs)):
            print("Matching ", j+1, "target image")
            prev_count = 0
            kp_target,des_target =  orb.detectAndCompute(target_imgs[j], None)
            des_targets.append([j, des_target])

            #print("target ID = ", j)

            for k in range(len(key_events)):
                print("k = ", k)

                if(len(key_events[k][1]) == 0):
                    frameID_des.append([])
                    continue
                else:
                    kp1, des1 =  orb.detectAndCompute(key_events[k][2], None)
                    frameID_des.append([k, des1])

                    matches = flann.knnMatch(frameID_des[k][1], des_targets[j][1], k=2)
                    count = 0

                    try:
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                count +=1
                    except ValueError:
                        pass

                    if count > prev_count:
                        if len(target_match[j]) > 0:    
                            target_match[j].pop(0)
                        target_match[j].append([frameID_des[k][0], count])
                        prev_count = count

        print("target_match = ", target_match)

        for k in range(len(target_match)):
            t.append(target_match[k][0][1])

        idx = t.index(max(t))
        target_location = target_match[idx][0][0]
        return target_location


    def get_path(self, target_ID):
        target_path = []
        for idx in range(target_ID+1):
            if(len(key_events[idx][1]) == 0):
                # target_path.append('0')
                #print("NO key")
                continue
            else:
                target_path.append(key_events[idx][1][0])
            if idx == target_ID:
                target_path.append('1')

        return target_path


    def nav(self, c, target_path):
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
            self.last_act = Action.IDLE
            #self.last_act = Action.QUIT          
            print("IDLE")


    def see(self, fpv):
        global i, navigating, counter, target_loc
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

        s = self.get_state()
        
        if(s is not None):
            print("phase = ", s[1].value)

            if(s[1].value == 1):
                if capturing:
                    print("Recording...")
                    key_events[i].append(cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY))
                    i += 1

            elif(s[1].value == 2):
                if not navigating:
                    target_loc = self.get_target_location(target_images)
                    print("matches =", target_loc)
                    cv2.imshow("fpv1", key_events[target_loc][2])
                    cv2.waitKey(20000) & 0xFF
                    navigating = True
                elif navigating:
                    path = self.get_path(target_loc)
                    if counter < len(path):
                        self.nav(counter, path)
                        counter += 1
    


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
    cv2.imshow("fpv", key_events[target_loc][2])
    cv2.waitKey(20000) & 0xFF
