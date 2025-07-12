# 🖐️ GestureForge

End‑to‑end toolkit for collecting hand‑gesture landmark data, training gesture‑recognition models, and exporting ready‑to‑deploy ONNX weights.

## 📌 Overview
GestureForge lets you quickly record palm/arm landmarks (static and dynamic gestures), train a model (Random‑Forest for static, GRU for dynamic gestures), sanity‑check the result, and use the trained model for export as portable ONNX weights, enabling deployment across C++, web, and edge environments.

## 📂 Directory Layout
```yaml
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
        ├── trainer.py                       # trainer_classes
        └── pose_landmarker_heavy.task 
```

## ⚙️ Prerequisites
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

## 🚀 Quick Start
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

## 🔄 Landmark Pipeline
```txt
MediaPipe (raw landmarks)
   └─ palm_detect  →  palm tensor  (scaled + normalised)
   └─ arm_detect   →  arm  tensor  (same normalisation)
      └─ hand_detect (fusion + relative geometry)
            → final hand vector      #  used by trainer / inference
```
#### Frame acceptance rules configured per run:
- By default, palms are always detected unless stated otherwise. (_Either detect single or both palms_)
- Detects only Arms. (_Either detect single or both arms_)
- Detects Arms along with Palms.

## 🧠 Training Details
#### Random‑Forest (Images)
- `sklearn.ensemble.RandomForestClassifier` with default hyper‑params
- Works well because static landmark vectors are highly separable across gestures.

#### GRU (Video)
- Input shape `(batch, seq_len, features)`
- Configurable via `train_config.yaml`:
  ```yaml
    data:
      batch_size: 2
    model:
      bidirectional_gru: false
      gru_hidden_size: 32
      gru_num_layers: 2
    training:
      epochs: 100
      learning_rate: 1e-3
      early_stopping_accuracy_thresh: false
      early_stopping_toll: 4
  ```

## 🔍 Model Quality Check
Both pipelines include an `inference.py` script that reproduces preprocessing and runs the freshly‑trained model to ensure data & training are correct before downstream deployment.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes. For more details and updates, visit the [GitHub Repository](https://github.com/tuhindutta/GestureForge).

## 🙏 Acknowledgements
- Google MediaPipe team for awesome real‑time landmark tracking
- PyTorch & scikit‑learn communities for the ML backbone

## 🔮 Future Prospects
GestureForge is built as a modular utility framework to streamline the process of recording, training, and validating hand gesture recognition models using landmark data. While the current implementation focuses on preparing models for accuracy verification and proof-of-concept testing, it lays the foundation for several impactful extensions in the future:
- 🚀 Quick Training for Real-world Applications
  - Enables rapid training of lightweight models (like RandomForest or GRU) on custom gestures, tailored to specific domains.
  - Helps developers and researchers prototype gesture-controlled interfaces without deep ML expertise.
- 📱 Edge & Embedded System Deployment
  - With lightweight models and structured landmark-based input, it becomes feasible to deploy trained models on microcontrollers, Raspberry Pi, or edge AI devices.
  - This opens doors to offline gesture-controlled IoT applications without reliance on cloud processing.
- 🌐 Integration with Internal Networks & Smart Environments
  - Can be embedded in internal enterprise systems to control operations via hand gestures — ideal for touchless interaction in secure or sterile environments.
  - Extendable to internal automation tools, remote devices, or closed-network industrial use-cases using IoT hubs.
- 🧠 Emergency Gesture Detection via CCTV Integration
  - GestureForge can power gesture-based emergency response systems using existing CCTV infrastructure.
  - Trained models can be embedded into surveillance workflows to allow people to silently signal distress or request help by making recognizable gestures in view of a camera.
  - This enables automated invocation of emergency protocols in public places like train stations, hospitals, campuses, or offices, thus providing an inclusive, silent, and accessible form of communication in critical moments.
- 🖥️ Cross-platform Support
  - Future ONNX export support (under development) allows integration with C++ or JavaScript runtimes for native desktop or web-based gesture controls.
- 📦 Custom Dataset & Gesture Expansion
  - Offers flexibility to continually update gesture datasets, enabling adaptation for different languages, sign systems, or domain-specific controls.
- 🧩 Plugin-based Architecture Possibility
  - Can evolve into a plugin-ready ecosystem where developers can swap detection logic, models, and inference layers based on the use case (e.g., robotics, home automation, AR/VR).
