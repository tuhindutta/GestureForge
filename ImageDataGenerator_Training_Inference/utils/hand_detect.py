from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect

class HandDetectCoords:

    def __init__(self, palm_detect:PalmDetect, arm_detect:ArmDetect, consider_palm_and_arm=False):
        self.palm_detect = palm_detect
        self.arm_detect = arm_detect
        self.consider_palm_and_arm = consider_palm_and_arm
        self.total_landmarks_count = 156

    def detect_palm(self, image, both_hands=False):
        left_detected, right_detected, coords = self.palm_detect.convert_coords_to_array_for_training(image)
        coords = coords + [-1]*(self.total_landmarks_count-len(coords)) if self.consider_palm_and_arm else coords
        return left_detected, right_detected, coords

    def detect_arm(self, image):
        side1_detected, side2_detected, coords = self.arm_detect.convert_coords_to_array_for_training(image)
        coords = [-1]*(self.total_landmarks_count-len(coords)) + coords if self.consider_palm_and_arm else coords
        return side1_detected, side2_detected, coords

    def convert_coords_to_array_for_training(self, image):
        left_detected, right_detected, palm_coords = self.detect_palm(image)
        palm_coords = palm_coords[:123]
        side1_detected, side2_detected, arm_coords = self.detect_arm(image)
        arm_coords = arm_coords[123:] if self.consider_palm_and_arm else arm_coords
        coords =  arm_coords + palm_coords
        return left_detected, right_detected, side1_detected, side2_detected, palm_coords+arm_coords

    def annotate(self, image):
        annotated = image.copy()
        annotated = self.palm_detect.annotate(annotated, bbox=False, text=False)
        annotated = self.arm_detect.annotate(annotated)
        return annotated