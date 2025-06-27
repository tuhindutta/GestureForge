import pickle
import random
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# import yaml
import json


# try:
#     with open('train_config.yaml', 'r') as file:
#         config = yaml.safe_load(file)
# except FileNotFoundError:
#     print("Error: 'config.yaml' not found.")
# except yaml.YAMLError as e:
#     print(f"Error parsing YAML: {e}")

output_dir = 'outputs'

with open(os.path.join(output_dir, 'data.pkl'),'rb') as f:
    data = pickle.load(f)

random.shuffle(data)

X = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

model = RandomForestClassifier()
model.fit(X, y)


with open(os.path.join(output_dir, 'model.pkl'),'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(output_dir, 'label_encoder.pkl'),'wb') as f:
    pickle.dump(label_encoder, f)

with open(os.path.join(output_dir, 'labels.json'), 'w') as json_file:
    json.dump(dict([(str(i), j) for i,j in list(enumerate(label_encoder.classes_))]),
              json_file, indent=4)

