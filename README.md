# Arabic Simple Sign Language Interpreter (ArSL) - Real-Time Recognition System

![Project Banner](***Sooooooooon***) 






*Real-time Arabic Sign Language letter recognition using MediaPipe and Machine Learning*

## Project Overview

This project is a **complete real-time Arabic Sign Language (ArSL) recognition system** that translates isolated Arabic letters and combined forms (like "ال", "لا", "ة") from hand gestures into written Arabic text — instantly on screen.

Developed by a team of 5 passionate developers and computer vision enthusiasts, this system combines:
- Hand landmark detection using **Google MediaPipe**
- A custom-trained **LinearSVC** classifier
- Smart feature engineering (translation-invariant landmarks)
- Full Arabic text rendering support (reshaping + bidirectional text)
- Intuitive user interface with writing mode, space detection, undo, and clear

The result? A smooth, accurate, and **fully functional Arabic sign language translator** that works with any webcam — no additional hardware needed.

**Winner potential in accessibility & AI competitions**  
**Perfect for deaf education, communication tools, and inclusive tech**

## Contributors

- **Abdelaziz El-banna** – [GitHub](https://github.com/AbdelazizElbanna)
- **Salah AbdEldaim** – [GitHub](https://github.com/salahAbdeldaim)
- **[Mohamed Abdelhakeam]** – [GitHub](https://github.com/M7MD4260)
- **[Tarek Dorgam]** – [GitHub](https://github.com/tarekdorgam127-gif)
- **[Jana Hazem]** – [GitHub](https://github.com/janahazemothman-coder)

## Repository Structure

| File / Folder                                   | Description                                                                                             |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `data/Sign_Language_dataset.csv`                | Final processed dataset (~thousands of samples, 63 normalized 3D hand landmarks + Arabic labels)      |
| `models/linearSVC.pkl`                          | Trained LinearSVC model (32 classes) – Ready for real-time inference                                   |
| `notebooks/01_make_data.ipynb`                  | Data collection pipeline – Capture images per letter + automatically extract landmarks                 |
| `notebooks/02_data_processing_and_modeling.ipynb` | Full training pipeline – Preprocessing, model training, evaluation & saving the model               |
| `notebooks/03_model_use.ipynb`                  | **Main Application** – Real-time Arabic sign language interpreter with camera and live text output   |
| `requirements.txt`                              | All required packages with exact tested & working versions                                             |
| `README.md`                                     | Project documentation (you are here!)                                                                   |# You are here

## Key Features

- Real-time hand tracking using **MediaPipe Hands**
- Recognizes **32 Arabic letters & forms** including:  
  أ ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي ء ة ال لا
- **Two-hand gesture = Space** (natural spacing)
- **Writing Mode Toggle** (`S` key) – Prevents accidental typing
- Stability system (25-frame threshold) → **No flickering predictions**
- Full **Arabic text rendering** (correctly connected letters using `arabic_reshaper` + `bidi`)
- Live progress bar under hand showing prediction confidence
- Controls:  
  `S` → Toggle writing mode `Backspace` → Delete last letter `C` → Clear all `Q` → Quit

## Demo Video
https://github.com/user-attachments/assets/543bab53-6f0d-4703-b9ac-14f9bef1f828


## Key Insights & Technical Highlights

- Features are **translation-invariant**: all landmarks relative to wrist → works anywhere in frame
- Dataset collected manually with consistent lighting and hand pose discipline
- High accuracy achieved using **LinearSVC** on normalized 3D landmarks
- Robust against slight rotations and scale changes
- Arabic text displayed correctly (no disconnected letters or reverse order)

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/AbdelazizElbanna/Arabic-Sign-Language-Interpreter.git
cd Arabic-Sign-Language-Interpreter
