import cv2
import numpy as np
import os
import pickle
import argparse
from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect
from utils.hand_detect import HandDetectCoords


parser = argparse.ArgumentParser(
    prog='inference',
    description='Collect hand landmark data for predict gesture.',
    epilog='Example: python inference.py -f 18 -t -a -A -bp -ant'
    )
parser.add_argument('-f', '--num_of_frames', type=int, default=18, help="Number of frames/video to record")
parser.add_argument('-t', '--confidence_thresh', type=float, default=0.5, help="Confidence threshold")
parser.add_argument('-a', '--predict_palm_and_arm_gesture', action='store_true', help="Enable both hands recording mode")
parser.add_argument('-A', '--predict_only_arm_gesture', action='store_true', help="Start recording")
parser.add_argument('-bp', '--record_both_palms', action='store_true', help="Enable both palms recording mode")
parser.add_argument('-ant', '--annotate', action='store_true', help="Annotate the gesture")

args = parser.parse_args()
num_of_frames = args.num_of_frames
confidence_thresh = args.confidence_thresh
predict_palm_and_arm_gesture = args.predict_palm_and_arm_gesture
predict_only_arm_gesture = args.predict_only_arm_gesture
record_both_palms = args.record_both_palms
annotate = args.annotate

output_dir = 'outputs'

palm = PalmDetect(0.1)
arm = ArmDetect('./utils/pose_landmarker_heavy.task')
hand = HandDetectCoords(palm, arm)

with open(os.path.join(output_dir, 'label_encoder.pkl'),'rb') as f:
    label_encoder = pickle.load(f)

with open(os.path.join(output_dir, 'model.pkl'),'rb') as f:
    model = pickle.load(f)

def predict(probs, label_encoder, encoded=False):
    predicted_label_idx = np.argmax(probs)
    conf = probs[predicted_label_idx]
    if encoded:
        label = predicted_label_idx[0]
    else:
        label = label_encoder.inverse_transform([predicted_label_idx])[0]
    return label, conf



cam = cv2.VideoCapture(0)

prediction = 'Predicting...'

while True:
    ret, frame_det = cam.read()
    frame = frame_det.copy()

    length, width, _ = frame.shape
    text_x = int(length * 0.04)
    text_y = int(width * 0.05)
    
    if predict_palm_and_arm_gesture:
        detector = hand
        sequence_length = 156
        left_detected, right_detected, side1_detected, side2_detected, coords = detector.convert_coords_to_array_for_training(frame)
        palm_detect_condition = left_detected and right_detected if record_both_palms else left_detected or right_detected
        arm_detect_condition = side1_detected or side2_detected
        frame_detect_condition = palm_detect_condition and arm_detect_condition
    elif predict_only_arm_gesture:
        detector = arm
        sequence_length = 33
        side1_detected, side2_detected, coords = detector.convert_coords_to_array_for_training(frame)
        frame_detect_condition = side1_detected or side2_detected
    else:
        detector = palm
        sequence_length = 123
        left_detected, right_detected,  coords = detector.convert_coords_to_array_for_training(frame)
        frame_detect_condition = left_detected and right_detected if record_both_palms else left_detected or right_detected

    if frame_detect_condition:

        if annotate:
            frame = detector.annotate(frame)
        else:
            frame = cv2.flip(frame, 1)
            

        probs = model.predict_proba([coords])[0]
        predicted, conf = predict(probs, label_encoder)
        if conf >= confidence_thresh:
            prediction = f"{predicted}: {int(conf*100)}%"
            array = []
    else:
        frame = cv2.flip(frame, 1)

    frame = cv2.putText(frame, prediction, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)        

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

