import numpy as np
import cv2
import mediapipe as mp

class PalmDetect:

    def __init__(self, confidence=0.5):
        self.__conf = confidence
        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_drawing_styles = mp.solutions.drawing_styles
        self.__mp_hands = mp.solutions.hands
        landmark_names = ['INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_DIP',
                          'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP', 'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP',
                          'PINKY_TIP', 'RING_FINGER_DIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_TIP', 'THUMB_CMC',
                          'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST']
        self.__landmark_pos = {i:getattr(self.__mp_hands.HandLandmark, i) for i in landmark_names}

    def __get_landmarks(self, image:np.ndarray):
        with self.__mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=self.__conf) as hands:
                results = hands.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks
            classif = [i.classification[0].label for i in results.multi_handedness]
        else:
            landmarks, classif = None, None
        return landmarks, classif

    def landmarks(self, image:np.ndarray):
        landmarks, classification = self.__get_landmarks(image)
        if landmarks:
            output = {}
            for i in range(len(landmarks)):
                landmark = landmarks[i]
                side = classification[i]
                results = {key: landmark.landmark[value] for key, value in self.__landmark_pos.items()}
                results = {key: (value.x, value.y, value.z) for key, value in results.items()}
                output[side] = results
        else:
            output = None
        return output

    def annotate(self, image:np.ndarray, bbox=False, text=False):
        annotated_image = cv2.flip(image, 1).copy()
        image_height, image_width, _ = annotated_image.shape
        hand_landmarks, classification = self.__get_landmarks(image)
        if hand_landmarks:
            for i in range(len(hand_landmarks)):
                landmark = hand_landmarks[i]
                side = classification[i]
                self.__mp_drawing.draw_landmarks(
                      annotated_image,
                      landmark,
                      self.__mp_hands.HAND_CONNECTIONS,
                      self.__mp_drawing_styles.get_default_hand_landmarks_style(),
                      self.__mp_drawing_styles.get_default_hand_connections_style())

                x = [lndmrk.x for lndmrk in landmark.landmark]
                y = [lndmrk.y for lndmrk in landmark.landmark]

                max_x = int(np.max(x)*image_width)+10
                max_y = int(np.max(y)*image_height)+10
                min_x = int(np.min(x)*image_width)-10
                min_y = int(np.min(y)*image_height)-10
                
                if bbox: 
                    cv2.rectangle(annotated_image, (min_x, max_y), (max_x, min_y), (255,0,0), 2)
                if text:
                    cv2.putText(annotated_image, side, (min_x, min_y-10), cv2.FONT_HERSHEY_COMPLEX, 1, [0,255,0], 2)
        return annotated_image

    @staticmethod
    def __dic2array(dic):
        arr = []
        for coord in dic.values():
            arr += list(coord)
        return arr
        
    def convert_coords_to_array_for_training(self, image:np.ndarray):

        landmarks = self.landmarks(image)

        left_detected = False
        right_detected = False
    
        if landmarks:

            left = landmarks.get('Left')      
            if left:

                left_detected = True
                
                left_coords = list(left.values())
                left_x = [i[0] for i in left_coords]
                left_y = [i[1] for i in left_coords]
                left_z = [i[2] for i in left_coords]       
                left_max_x = np.max(left_x)
                left_max_y = np.max(left_y)
                left_min_x = np.min(left_x)
                left_min_y = np.min(left_y)
                left_max_z = np.max(left_z)
                left_min_z = np.min(left_z)
                left_img_size = np.sqrt((left_max_x-left_min_x)**2 + (left_max_y-left_min_y)**2 + (left_max_z-left_min_z)**2)
                
                left_wrist = left['WRIST']
                del(left['WRIST'])
                for key in left.keys():
                    left_rel_pos = np.array(left_wrist) - left[key]
                    left[key] = tuple([i/left_img_size for i in left_rel_pos])
                left = self.__dic2array(left)
            else:
                left = [-1]*60
                        
            right = landmarks.get('Right')
            if right:

                right_detected = True
                
                right_coords = list(right.values())
                right_x = [i[0] for i in right_coords]
                right_y = [i[1] for i in right_coords] 
                right_z = [i[2] for i in right_coords]        
                right_max_x = np.max(right_x)
                right_max_y = np.max(right_y)
                right_min_x = np.min(right_x)
                right_min_y = np.min(right_y)
                right_max_z = np.max(right_z)
                right_min_z = np.min(right_z)
                right_img_size = np.sqrt((right_max_x-right_min_x)**2 + (right_max_y-right_min_y)**2 + (right_max_z-right_min_z)**2)

                right_wrist = right['WRIST']
                del(right['WRIST'])
                for key in right.keys():
                    right_rel_pos = np.array(right_wrist) - right[key]
                    
                    right[key] = tuple([i/right_img_size for i in right_rel_pos])
                right = self.__dic2array(right)
            else:
                right = [-1]*60
                    
        else:
            left = [-1]*60
            right = [-1]*60

        if left_detected and right_detected:
            left_right_mean_size = (left_img_size+right_img_size)/2
            dist_bet_wrists = np.array(left_wrist) - np.array(right_wrist)
            dist_bet_wrists = [i/left_right_mean_size for i in dist_bet_wrists]
        else:
            dist_bet_wrists = [-1,-1,-1]
            
        return left_detected, right_detected, left+right+dist_bet_wrists