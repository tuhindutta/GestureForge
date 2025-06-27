import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ArmDetect:

    def __init__(self, pose_model_path):
        base_options = python.BaseOptions(model_asset_path=pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def __draw_landmarks_on_image(rgb_image, detection_result):
      pose_landmarks_list = detection_result.pose_landmarks
      annotated_image = np.copy(rgb_image)
    
      for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
    
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
      return annotated_image

    def __get_landmarks(self, image:np.ndarray):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(img)
        return detection_result

    def annotate(self, image:np.ndarray):
        annotated = image.copy()
        annotated = self.__draw_landmarks_on_image(annotated, self.__get_landmarks(image))
        return annotated

    def convert_coords_to_array_for_training(self, image:np.ndarray):
        side1_detected = False
        side2_detected = False

        lndmrks = self.__get_landmarks(image)
        
        if lndmrks and (len(lndmrks.pose_landmarks)>0):
            side1, side2 = [12,14,16,18,20,22], [11,13,15,17,19,21]
            ref_idx = 2
            detected_coords = []
            for x,y,z,presence in [(i.x, i.y, i.z, i.presence) for i in lndmrks.pose_landmarks[0]]:
                if presence >= 0.9:
                    detected_coords.append((x,y,z))
                else:
                    detected_coords.append(None)
                    
            side1_coords = [detected_coords[i] for i in side1]
            side2_coords = [detected_coords[i] for i in side2]
            
            
            side1_detected = any(side1_coords) and side1_coords[ref_idx] is not None
            side2_detected = any(side2_coords) and side2_coords[ref_idx] is not None

            
            
            
            
            if side1_detected:
                side1_ref = side1_coords.pop(ref_idx)
                side1_x = [i[0] for i in side1_coords if i]
                side1_y = [i[1] for i in side1_coords if i]
                side1_z = [i[2] for i in side1_coords if i]
                side1_min_x = min(side1_x)
                side1_min_y = min(side1_y)
                side1_min_z = min(side1_z)
                side1_max_x = max(side1_x)
                side1_max_y = max(side1_y)
                side1_max_z = max(side1_z)
                side1_image_size = np.sqrt((side1_max_x - side1_min_x)**2 + (side1_max_y - side1_min_y)**2 + (side1_max_z - side1_min_z)**2)

                side1_rel_coords = [np.array(side1_ref) - np.array(i) if i else i for i in side1_coords]
                side1_rel_coords = [tuple(i/side1_image_size) if i is not None else (-1,-1,-1) for i in side1_rel_coords]
            else:
                side1_rel_coords = [-1]*15

            if side2_detected:
                side2_ref = side2_coords.pop(ref_idx)
                side2_x = [i[0] for i in side2_coords if i]
                side2_y = [i[1] for i in side2_coords if i]
                side2_z = [i[2] for i in side2_coords if i]
                side2_min_x = min(side2_x)
                side2_min_y = min(side2_y)
                side2_min_z = min(side2_z)
                side2_max_x = max(side2_x)
                side2_max_y = max(side2_y)
                side2_max_z = max(side2_z)
                side2_image_size = np.sqrt((side2_max_x - side2_min_x)**2 + (side2_max_y - side2_min_y)**2 + (side2_max_z - side2_min_z)**2)
   
                side2_rel_coords = [np.array(side2_ref) - np.array(i) if i else i for i in side2_coords]
                side2_rel_coords = [tuple(i/side2_image_size) if i is not None else (-1,-1,-1) for i in side2_rel_coords]
            else:
                side2_rel_coords = [-1]*15

            ref_diff = list(np.array(side1_ref) - np.array(side2_ref)) if side1_detected and side2_detected else [-1]*3

            coords = side1_rel_coords + side1_rel_coords
            coords = [i[0] for i in np.array(coords).reshape(-1,1)] + ref_diff

        else:
            side1_detected, side2_detected, coords = False, False, [-1]*33
        return side1_detected, side2_detected, coords