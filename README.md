# GestureForge

End‑to‑end toolkit for collecting hand‑gesture landmark data, training gesture‑recognition models, and exporting ready‑to‑deploy ONNX weights.

## Overview
GestureForge lets you quickly record palm/arm landmarks (image or video), train a model (Random‑Forest for images, GRU for sequences), sanity‑check the result, ready to export portable ONNX weights for C++ / Web / Edge deployment.

## Directory Layout
```sql
GestureForge/
│   .gitignore
│   LICENSE                                  # MIT or Apache 2.0 (your pick)
│   README.md                                # short project summary
│   requirements.txt                         # exact library versions
│
├── ImageDataGenerator_Training_Inference/
│   ├── data_collect.py                      # record single‑frame samples
│   ├── train.py                             # RandomForest pipeline
│   ├── inference.py                         # quick accuracy sanity‑check
│   ├── train_config.yaml                    # (unused — RF uses defaults)
│   ├── outputs/                             # pickles + encoders after each run
│   └── utils/                               # utilities
│       ├── arm_detect.py                    # arm_detect
│       ├── hand_detect.py                   # hand_detect
│       ├── palm_detect.py                   # palm_detect
│       └── pose_landmarker_heavy.task       # pose_landmarker_model
│
└── VideoDataGenerator_Training_Inference/
    ├── data_collect.py                      # record multi‑frame sequences
    ├── train.py                             # GRU training driven by YAML
    ├── inference.py                         # sanity‑check sequence model
    ├── train_config.yaml                    # epochs, hidden_size, etc.
    ├── outputs/                             # *.pt, pickles, CSV metadata
    └── utils/                               # same detector modules as above
        ├── arm_detect.py                    # arm_detect
        ├── hand_detect.py                   # hand_detect
        ├── palm_detect.py                   # palm_detect
        └── trainer.py                       # trainer_classes
```

## Prerequisites
#### 1️⃣ Python 3.11 virtual‑env
```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
source .venv/bin/activate   # Windows: .venv\Scripts\activate

#### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start
#### 1️⃣ Image‑Gesture Workflow (single frame)
```bash
cd ImageDataGenerator_Training_Inference

# (1) Record N samples of label "wave" using only palm landmarks
python data_collect.py okay -r

# (2) Train Random‑Forest (defaults)
python train.py

# (3) Sanity‑check inference
python inference.py -ant
```

#### 2️⃣ Video‑Gesture Workflow (frame sequences)
```bash
cd VideoDataGenerator_Training_Inference

# (1) Record sequences (palm + arm landmarks, 60 frames each)
python data_collect.py clap -r

# (2) Configure training
vim train_config.yaml

# (3) Train GRU
python train.py         # writes model.pt & label_encoder.pkl

# (4) Sanity‑check inference
python inference.py
```

## Landmark Pipeline
```txt
MediaPipe (raw landmarks)
   └─ palm_detect  →  palm tensor  (scaled + normalised)
   └─ arm_detect   →  arm  tensor  (same normalisation)
      └─ hand_detect (fusion + relative geometry)
            → final hand vector      #  used by trainer / inference
```
#### Frame acceptance rules configured per run:
- By default, palms are always detected unless stated otherwise.
- Detects only Arms.
- Detects Arms along with Palms.

## Training Details
#### Random‑Forest (Images)
- `sklearn.ensemble.RandomForestClassifier` with default hyper‑params
- Works well because static landmark vectors are highly separable across gestures.

#### GRU (Video)
- Input shape `(batch, seq_len, features)`
- Configurable via `train_config.yaml`:
  ```yaml
  data:
  batch_size: 3
  model:
    gru_hidden_size: 32
    gru_num_layers: 2
  training:
    epochs: 20
    learning_rate: 1e-3
    early_stopping_accuracy_thresh: 0.05
    early_stopping_toll: 4
  ```

## Model Quality Check
Both pipelines include an `inference.py` script that reproduces preprocessing and runs the freshly‑trained model to ensure data & training are correct before downstream deployment.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes. For more details and updates, visit the [GitHub Repository](https://github.com/tuhindutta/GestureForge).

## Acknowledgements
- Google MediaPipe team for awesome real‑time landmark tracking
- PyTorch & scikit‑learn communities for the ML backbone
