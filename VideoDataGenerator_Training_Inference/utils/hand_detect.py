import cv2
from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect

# class HandDetectCoords:

#     def __init__(self, palm_detect:PalmDetect, arm_detect:ArmDetect, consider_palm_and_arm=False):
#         self.palm_detect = palm_detect
#         self.arm_detect = arm_detect
#         self.consider_palm_and_arm = consider_palm_and_arm
#         self.total_landmarks_count = 156

#     def detect_palm(self, image, both_hands=False):
#         left_detected, right_detected, coords = self.palm_detect.convert_coords_to_array_for_training(image)
#         coords = coords + [-1]*(self.total_landmarks_count-len(coords)) if self.consider_palm_and_arm else coords
#         return left_detected, right_detected, coords

#     def detect_arm(self, image):
#         side1_detected, side2_detected, coords = self.arm_detect.convert_coords_to_array_for_training(image)
#         coords = [-1]*(self.total_landmarks_count-len(coords)) + coords if self.consider_palm_and_arm else coords
#         return side1_detected, side2_detected, coords

#     def convert_coords_to_array_for_training(self, image):
#         left_detected, right_detected, palm_ref1, palm_ref2, palm_ref1_image_size, palm_ref2_image_size, palm_coords = self.detect_palm(image)
#         palm_coords = palm_coords[:123]
#         side1_detected, side2_detected, arm_ref1, arm_ref2, arm_ref1_image_size, arm_ref2_image_size, arm_coords = self.detect_arm(image)
#         arm_coords = arm_coords[123:] if self.consider_palm_and_arm else arm_coords
#         return left_detected, right_detected, palm_ref1, palm_ref2, palm_ref1_image_size, palm_ref2_image_size, side1_detected, side2_detected, arm_ref1, arm_ref2, arm_ref1_image_size, arm_ref2_image_size, palm_coords+arm_coords

#     def annotate(self, image):
#         annotated = image.copy()
#         annotated = self.palm_detect.annotate(annotated, bbox=False, text=False)
#         annotated = self.arm_detect.annotate(annotated)
#         return annotated

class HandDetectCoords:

    def __init__(self, palm_detect:PalmDetect, arm_detect:ArmDetect, consider_palm_and_arm=False):
        self.palm_detect = palm_detect
        self.arm_detect = arm_detect

    def convert_coords_to_array_for_training(self, image):
        left_detected, right_detected, palm_ref1, palm_ref2, palm_ref1_image_size, palm_ref2_image_size, palm_coords = self.palm_detect.convert_coords_to_array_for_training(image)
        side1_detected, side2_detected, arm_ref1, arm_ref2, arm_ref1_image_size, arm_ref2_image_size, arm_coords = self.arm_detect.convert_coords_to_array_for_training(image)
        return left_detected, right_detected, palm_ref1, palm_ref2, palm_ref1_image_size, palm_ref2_image_size, side1_detected, side2_detected, arm_ref1, arm_ref2, arm_ref1_image_size, arm_ref2_image_size, palm_coords+arm_coords

    def annotate(self, image):
        annotated = image.copy()
        annotated = self.palm_detect.annotate(annotated, bbox=False, text=False)
        annotated = cv2.flip(annotated, 1)
        annotated = self.arm_detect.annotate(annotated)
        return annotated