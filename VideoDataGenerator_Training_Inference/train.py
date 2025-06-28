import pickle
import random
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
from torch.utils.data import DataLoader
from utils.trainer import HandDataset, GruModel, BiGruModel


try:
    with open('train_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: 'config.yaml' not found.")
except yaml.YAMLError as e:
    print(f"Error parsing YAML: {e}")

output_dir = 'outputs'

with open(os.path.join(output_dir, 'data.pkl'),'rb') as f:
    data = pickle.load(f)

random.shuffle(data)

X = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

batch_size = config['data']['batch_size']

dataset = HandDataset(X, y)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_size = X.shape[-1]
hidden_size = config['model']['gru_hidden_size']
num_layers = config['model']['gru_num_layers']
bidirectional = config['model']['bidirectional_gru']
Model = BiGruModel if bidirectional else GruModel
print(f'{Model.__name__} model invoked')
model = Model(input_size, hidden_size, num_layers, len(label_encoder.classes_), device).to(device)


epochs = config['training']['epochs']
learning_rate = float(config['training']['learning_rate'])

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr = learning_rate)

early_stopping_accuracy_thresh = config['training']['early_stopping_accuracy_thresh']
early_stopping_toll = config['training']['early_stopping_toll']
accuracies = []

def determine_early_stop(accuracies, early_stopping_accuracy_thresh, early_stopping_toll):
    early_stop = False
    if (len(accuracies) >= early_stopping_toll) and early_stopping_accuracy_thresh:
        selected_accuracies = accuracies[-1*early_stopping_toll:]
        min_acc = np.min(selected_accuracies)
        max_acc = np.max(selected_accuracies)
        diff = max_acc - min_acc
        if diff <= early_stopping_accuracy_thresh:
            early_stop = True            
    return early_stop

best_model_loss = 1
for epoch in range(epochs):
    if not determine_early_stop(accuracies, early_stopping_accuracy_thresh, early_stopping_toll):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch_x, batch_y in data_loader:

            total_samples += len(batch_x)
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = loss_func(outputs, batch_y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1)
            correct += (batch_y==predicted).sum().item()

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = 100 * correct / total_samples
        accuracies.append(avg_accuracy)
        if avg_loss < best_model_loss:
            best_model_accuracy = avg_accuracy
            best_model_loss = avg_loss
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(output_dir, 'model.pt'))
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")
    
    else:
        print('Early Stopping')
        break

print(f'Best model with {best_model_accuracy/100} accuracy and {best_model_loss:.3f} loss is saved.')

with open(os.path.join(output_dir, 'label_encoder.pkl'),'wb') as f:
    pickle.dump(label_encoder, f)

with open(os.path.join(output_dir, 'labels.json'), 'w') as json_file:
    json.dump(dict([(str(i), j) for i,j in list(enumerate(label_encoder.classes_))]),
              json_file, indent=4)

