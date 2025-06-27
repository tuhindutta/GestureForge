import cv2
import os
import pandas as pd
import numpy as np
import pickle
import argparse
from utils.palm_detect import PalmDetect
from utils.arm_detect import ArmDetect
from utils.hand_detect import HandDetectCoords

parser = argparse.ArgumentParser(
    prog='lndmrk',
    description='Collect hand landmark data for gesture classification.',
    epilog='Example: python data_collect.py label_example -n 10 -f 20 -bp -r -N -a -A'
    )
parser.add_argument('label', type=str, help="Label for the current sample")  # positional argument
parser.add_argument('-n', '--num_of_samples', type=int, default=8, help="Number of samples to record")
parser.add_argument('-f', '--num_of_frames', type=int, default=18, help="Number of frames/video to record")  # optional with default
parser.add_argument('-bp', '--record_both_palms', action='store_true', help="Enable both palms recording mode")  # flag: True if present
parser.add_argument('-r', '--record', action='store_true', help="Start recording")  # flag: True if present
parser.add_argument('-N', '--create_new_data_file', action='store_true', default=False, help="Remove the old data and create new file")
parser.add_argument('-a', '--record_arm_coords', action='store_true', default=False, help="Record arm coordinates")
parser.add_argument('-A', '--record_only_arm_coords', action='store_true', default=False, help="Record only arm coordinates")


args = parser.parse_args()
label = args.label
no_of_frames_per_video = args.num_of_frames
record_both_palms = args.record_both_palms
record_data = args.record
no_of_samples_required = args.num_of_samples if record_data else 1
create_new_data_file = args.create_new_data_file
record_arm_coords = args.record_arm_coords
record_only_arm_coords = args.record_only_arm_coords

palm = PalmDetect(0.1)
arm = ArmDetect('./utils/pose_landmarker_heavy.task')
hand = HandDetectCoords(palm, arm)


output_dir = 'outputs'
data_path = os.path.join(output_dir, 'data.pkl')
record_path = os.path.join(output_dir, 'record.csv')
metadata_path = os.path.join(output_dir, 'data_prep_metadata.csv')

if create_new_data_file:
    if 'data.pkl' in os.listdir(output_dir):
        os.remove(data_path)
    if 'record.csv' in os.listdir(output_dir):
        os.remove(record_path)
    if 'data_prep_metadata.csv' in os.listdir(output_dir):
        os.remove(metadata_path)


if 'data_prep_metadata.csv' in os.listdir(output_dir):
    metadata = pd.read_csv(metadata_path)
else:
    metadata = pd.DataFrame(columns=['label', 'no_of_frames_per_video', 'record_both_palms',
                                     'record_data', 'no_of_samples_required', 'create_new_data_file',
                                     'record_arm_coords', 'record_only_arm_coords'])
metadata.loc[len(metadata)] = [label, no_of_frames_per_video, record_both_palms,
                               record_data, no_of_samples_required, create_new_data_file,
                               record_arm_coords, record_only_arm_coords]
metadata.to_csv(metadata_path, index=False)

# print(label, no_of_samples_required, no_of_frames_per_video, both_hands, record_data, create_new_data)
def data_shape_match(shape):
    if shape == 156:
        return 'record_arm_coords'
    elif shape == 33:
        return 'record_only_arm_coords'
    elif shape == 123:
        return 'record_only_palm_coords'

for _ in range(no_of_samples_required):
    array = []

    if 'data.pkl' in os.listdir(output_dir):
        with open(data_path,'rb') as f:
            data = pickle.load(f)
        data_shape = np.array([i[0] for i in data]).shape[-1]
    else:
        data = []
        data_shape = -1


    if 'record.csv' in os.listdir(output_dir):
        present_count = len([j for _,j in data if j==label])
        record = pd.read_csv(record_path, index_col='label')
    else:
        present_count = 0
        record = pd.DataFrame(columns=['label','count']).set_index('label')



    if (present_count < no_of_samples_required) or not record_data:


        cam = cv2.VideoCapture(0)

        frame_count = 1

        while (frame_count <= no_of_frames_per_video):
            ret, frame = cam.read()

            if record_only_arm_coords:
                detector = arm
                side1_detected, side2_detected, coords = detector.convert_coords_to_array_for_training(frame)
                frame_detect_condition = side1_detected or side2_detected
            elif record_arm_coords:
                detector = hand
                left_detected, right_detected, side1_detected, side2_detected, coords = detector.convert_coords_to_array_for_training(frame)
                palm_detect_condition = left_detected and right_detected if record_both_palms else left_detected or right_detected
                arm_detect_condition = side1_detected or side2_detected
                frame_detect_condition = palm_detect_condition and arm_detect_condition
            else:
                detector = palm
                left_detected, right_detected,  coords = detector.convert_coords_to_array_for_training(frame)
                frame_detect_condition = left_detected and right_detected if record_both_palms else left_detected or right_detected

            assert (len(coords) == data_shape) or (data_shape == -1), f"Data record should be for {data_shape_match(data_shape)}"

            if frame_detect_condition:
                frame_count += 1
                array.append(coords)
                frame = detector.annotate(frame)

            else:
                frame = cv2.flip(frame, 1)

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        # out.release()
        cv2.destroyAllWindows()

        if (len(array) == no_of_frames_per_video) and record_data:
            data.append([array, label])
            with open(data_path,'wb') as f:
                pickle.dump(data, f)

            new_count = present_count + 1
            print(f"No. of instances recorded for '{label}' label: {new_count}/{no_of_samples_required}")
            record.loc[label] = [new_count]
            record.to_csv(record_path)

    else:
        print('Required number of samples gathered.')
        break