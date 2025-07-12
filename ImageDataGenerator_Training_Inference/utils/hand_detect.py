import cv2
from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect

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