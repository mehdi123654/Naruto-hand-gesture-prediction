# Naruto Hand Sign Detector 🥷🔥

Detect Naruto-style hand signs (jutsus) in real-time using a custom-trained YOLOv8 model.

---

## 📂 Project Structure

- `data/`: Raw and processed dataset (images and labels)
- `models/`: Trained YOLOv8 models
- `scripts/`: Code for data preparation, training, and live detection
- `README.md`: Project info

---

## 🚀 How to Run

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python scripts/prepare_data.py
```

### 3. Train Model
```bash
python scripts/train.py
```

### 4. Run Live Detection
```bash
python scripts/live_test.py
```

---

## ⚡ Current Model Performance

- **Precision:** ~95%
- **Recall:** ~97%
- **mAP@0.5:** ~99.5%

---

## 🎯 Goal

- Recognize hand signs accurately
- Detect full jutsu sequences in the future

---

## ✍️ Author

Made with passion for Naruto and Machine Learning 🔥
