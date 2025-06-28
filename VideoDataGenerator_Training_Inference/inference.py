import cv2
import numpy as np
import os
import pickle
import torch
import argparse
from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect
from utils.hand_detect import HandDetectCoords

parser = argparse.ArgumentParser(
    prog='inference',
    description='Collect hand landmark data for predict gesture.',
    epilog='Example: python inference.py -d 0 -f 18 -t -a -A -bp -ant'
    )
parser.add_argument('-d', '--video_device', type=int, default=0, help="Video device")
parser.add_argument('-f', '--num_of_frames', type=int, default=18, help="Number of frames/video to record")
parser.add_argument('-t', '--confidence_thresh', type=float, default=0.5, help="Confidence threshold")
parser.add_argument('-a', '--predict_palm_and_arm_gesture', action='store_true', help="Enable both hands recording mode")
parser.add_argument('-A', '--predict_only_arm_gesture', action='store_true', help="Start recording")
parser.add_argument('-bp', '--record_both_palms', action='store_true', help="Enable both palms recording mode")
parser.add_argument('-ant', '--annotate', action='store_true', help="Annotate the gesture")

args = parser.parse_args()
video_device = args.video_device
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.jit.load(os.path.join(output_dir, 'model.pt'))

def predict(logits:torch.Tensor, label_encoder, encoded=False):
    probs = torch.nn.functional.softmax(logits, dim=1)
    predicted_label_idx = torch.argmax(probs).numpy().reshape((-1))
    conf = probs.detach().numpy()[0][predicted_label_idx]
    if encoded:
        label = predicted_label_idx[0]
    else:
        label = label_encoder.inverse_transform(predicted_label_idx)[0]
    return label, conf


model.eval()

cam = cv2.VideoCapture(video_device)

array = []
initial_ref1, initial_ref2 = np.array([0,0,0]), np.array([0,0,0])
initial_ref1_image_size, initial_ref2_image_size = 0, 0
prediction = 'Predicting...'

while True:
    ret, frame_det = cam.read()
    frame = frame_det.copy()

    length, width, _ = frame.shape
    text_x = int(length * 0.04)
    text_y = int(width * 0.05)
    
    if predict_palm_and_arm_gesture:
        detector = hand
        sequence_length = 156 + 6 + 6
        left_detected, right_detected, ref1, ref2, ref1_image_size, ref2_image_size, side1_detected, side2_detected, _, _, _, _, coords = detector.convert_coords_to_array_for_training(frame)
        palm_detect_condition = left_detected and right_detected if record_both_palms else left_detected or right_detected
        arm_detect_condition = side1_detected or side2_detected
        frame_detect_condition = palm_detect_condition and arm_detect_condition
    elif predict_only_arm_gesture:
        detector = arm
        sequence_length = 33 + 6
        side1_detected, side2_detected, ref1, ref2, ref1_image_size, ref2_image_size, coords = detector.convert_coords_to_array_for_training(frame)
        frame_detect_condition = side1_detected or side2_detected
    else:
        detector = palm
        sequence_length = 123 + 6
        left_detected, right_detected, ref1, ref2, ref1_image_size, ref2_image_size, coords = detector.convert_coords_to_array_for_training(frame)
        both_palms_detected = left_detected and right_detected
        frame_detect_condition = both_palms_detected if record_both_palms else ((left_detected or right_detected) and not both_palms_detected)

    if frame_detect_condition:

        if annotate:
            frame = detector.annotate(frame)
        else:
            frame = cv2.flip(frame, 1)
        
        ref1 = np.array(ref1)
        ref2 = np.array(ref2)
        if len(array) == 0:
            velocity_ref1 = list(initial_ref1)
            velocity_ref2 = list(initial_ref2)
        else:
            ref1_mean_img_size = (initial_ref1_image_size + ref1_image_size)/2
            ref2_mean_img_size = (initial_ref2_image_size + ref2_image_size)/2
            velocity_ref1 = [i/ref1_mean_img_size if ref1_mean_img_size > 0 else i for i in list(ref1 - initial_ref1)]
            velocity_ref2 = [i/ref2_mean_img_size if ref2_mean_img_size > 0 else i for i in list(ref2 - initial_ref2)]

        initial_ref1 = ref1
        initial_ref2 = ref2
        initial_ref1_image_size = ref1_image_size
        initial_ref2_image_size = ref2_image_size          

        array.append(coords + velocity_ref1 + velocity_ref2)

        if len(array) % num_of_frames == 0:
            coords_tensor = torch.tensor(np.array(array).reshape((1,num_of_frames,sequence_length)),
                                         dtype=torch.float32)

            predicted, conf = predict(model(coords_tensor), label_encoder)
            if conf >= confidence_thresh:
                prediction = f"{predicted}: {int(conf*100)}%"
            array = []
            initial_ref1, initial_ref2 = np.array([0,0,0]), np.array([0,0,0])
            initial_ref1_image_size, initial_ref2_image_size = 0, 0
    else:
        frame = cv2.flip(frame, 1)

    frame = cv2.putText(frame, prediction, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)        

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

