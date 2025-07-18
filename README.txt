# 🚨 Real-Time Emergency Vehicle Siren Detection and Traffic Management

This project is the Final Year Project (FYP) of Dhanushka Salinda and [Your Partner’s Name] at SLTC Research University. It focuses on detecting emergency sirens in real-time and managing traffic accordingly using Machine Learning and edge computing (Jetson Nano).

---

## 🧠 Key Features

- Real-time siren sound detection using CNN
- Audio preprocessing and filtering (HPF, LPF, BPF, MFCC)
- Raspberry Pi/Jetson Nano edge deployment
- Traffic signal control logic for emergency vehicle prioritization

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Librosa (Audio processing)
- Jetson Nano (for edge inference)
- Flask API (for integration)

---

## 📁 Project Structure

📦 siren-detection
├── main.py
├── model/
│ └── siren_detection_model_v2.h5
├── utils/
│ └── filters.py
├── data/
│ └── siren_audio/
├── README.md
└── .gitignore