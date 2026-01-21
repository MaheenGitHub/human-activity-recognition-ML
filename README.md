# Human Activity Recognition (HAR) Project

This repository contains a Machine Learning semester project on **Human Activity Recognition (HAR)** using smartphone sensor data.  
The project evaluates and compares **classical machine learning models**, **fully connected deep neural networks (DNN)**, and **sequence-based deep learning models (LSTM)** for activity classification.
🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-lightgrey?style=for-the-badge&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.22-blue?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellow?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11-purple?style=for-the-badge&logo=seaborn)

---

## 📂 Repository Structure
```bash

human-activity-recognition-ML/
│
├── Time_Series_Data_Analysis_&_Preprocessing/
│   ├── Time_Series_Data_Analysis_&_Preprocessing.ipynb
│   ├── Assignment_Deliverables_01.pdf
│   ├── train.zip
│   └── test.zip
│
├── Feature_Engineering_and_Classical_Machine_Learning_Classification/
│   ├── Feature_Engineering_and_Classical_Machine_Learning_Classification.ipynb
│   ├── A5_2_results.csv
│   └── Assignment_Deliverables_02.docx
│
├── Fully_Connected_Deep_Neural_Network/
│   ├── Fully_Connected_Deep_Neural_Network.ipynb
│   ├── HAR_DNN_Model_5_3.keras
│   ├── NN_template.py
│   ├── A5_3_training_curves.png
│   ├── label_encoder.pkl
│   └── Assignment_Deliverables_3.docx
│
├── Sequence-Based Deep Learning Classification/
│   ├── Sequence_Based_Deep_Learning_Classification.ipynb
│   ├── best_lstm.keras
│   ├── activity_labels.txt
│   ├── features.txt
│   ├── features_info.txt
│   └── Assignment_deliverable_4.docx
│
├── Project_Master_Results.csv
├── LICENSE
└── README.md
```

---

## 📂 Project Deliverables

### Deliverable 1 – Time Series Data Analysis & Preprocessing
- **Notebook:** `Time_Series_Data_Analysis_&_Preprocessing.ipynb`  
- **Contents:**  
  - Dataset loading & preprocessing  
  - Timestamp generation & chronological sorting  
  - Missing values & duplicate check  
  - Activity & subject distribution visualization  
  - Feature domain & consistency check  
- **Assignment Document:** `Assignment_Deliverables_01.pdf`  

---

### Deliverable 2 – Feature Engineering & Classical Machine Learning
- **Notebook:** `Feature_Engineering_and_Classical_Machine_Learning_Classification.ipynb`  
- **Models Implemented:** Logistic Regression (LR), SVM, Decision Tree (DT), Random Forest (RF), XGBoost  
- **Features:** 561 handcrafted time and frequency domain features  
- **Results (Best Model – SVM):**  
  - **Accuracy:** 96%  
  - **F1-Score:** 0.96  
- **Results CSV:** `A5_2_results.csv`  
- **Assignment Document:** `Assignment_Deliverables_02.docx`  

---

### Deliverable 3 – Fully Connected Deep Neural Network (DNN)
- **Notebook:** `Fully_Connected_Deep_Neural_Network.ipynb`  
- **Model File:** `HAR_DNN_Model_5_3.keras`  
- **Architecture:** 
  - Input Layer: 561 neurons  
  - Hidden Layers: 256 → 128 → 64 neurons with ReLU activation, Dropout 30% on first 2 layers  
  - Output Layer: 6 neurons, Softmax activation  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 32  
- **Epochs:** 30 (with EarlyStopping & ReduceLROnPlateau)  
- **Results:**  
  - **Accuracy:** 92.94%  
  - **F1-Score:** 0.9286  
- **Training Curves:** `A5_3_training_curves.png`  
- **Label Encoder:** `label_encoder.pkl`  
- **Assignment Document:** `Assignment_Deliverables_3.docx`


  

---

### Deliverable 4 – Sequence-Based Deep Learning (Final Capstone)
- **Notebook:** `Sequence_Based_Deep_Learning_Classification.ipynb`  
- **Models Implemented:** Simple RNN , LSTM ,GRU
- **Input:** 3D sequential data (samples × time steps × features)
- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Batch Size:** 32  
- **Epochs:** 30
- **Results:**  
  - **RNN:** Accuracy ≈ 85.65% ,F1-score ≈  0.8555
  - **LSTM:** Accuracy ≈ 91.25%, F1-score ≈  0.9140
  - **GRU:** Accuracy ≈  90.46%, F1-score ≈ 0.9064
- **Artifacts:**
  - **Best model:** best_lstm.keras
  - **Metadata:** activity_labels.txt, features.txt, features_info.txt
> Sequence models operate on reshaped sequential representations derived from the original sensor windows.
- **Assignment Document:** `Assignment_deliverable_4.docx`

---

## 📊 Dataset Overview

- **Dataset Name:** UCI Human Activity Recognition (HAR) Using Smartphones  
- **Source:** [Kaggle / UCI Repository](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)  
- **Sensor Types:** Triaxial Accelerometer & Gyroscope  
- **Number of Samples:** 10,299 total (7,352 Training, 2,947 Testing)  
- **Number of Classes:** 6  
  - LAYING, SITTING, STANDING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS  
- **Features:** 561 engineered numeric features (time & frequency domain)  
- **Sample Representation:** Each row represents a 2.56-second window of raw sensor data  
- **Subjects:** 30  

---

## 🧹 Data Preprocessing

- Generated synthetic **timestamps** (50Hz sampling rate)  
- Sorted data chronologically and set Timestamp as index  
- Verified **no missing values** or duplicates  
- Normalized features using **StandardScaler** for DNN & sequence models  
- Encoded activity labels numerically (**Label Encoding** & **One-Hot Encoding** for DNN)  

---


## 📊 Performance Metrics

| Model | Accuracy | F1-Score | Notes |
| :--- | :--- | :--- | :--- |
| SVM | 96% | 0.96 | Strong baseline for low-power inference |
| DNN | 92.94% | 0.928 | Works well on high-dimensional features |
| LSTM | 91.25% | 0.914 | Captures temporal dependencies |

**Inference Speed:** ~0.47ms per activity window (LSTM).  
**Stability:** 30% Dropout prevents overfitting, ensuring robust predictions.

--
## 💡 Industrial Use Cases

**Healthcare:** Fall detection for elderly care.

**Logistics:** Worker ergonomics monitoring to reduce injuries.

**Insurance / FinTech:** Activity-based insurance premium modeling.

**Smart Home / IoT:** Gesture-based home automation.

---

## 🤝 Contributing & Collaboration

I’m actively looking to expand it for:

**Edge Deployment:** TFLite conversion for ESP32 / Arduino.

**UI/UX:** Streamlit or Flutter real-time dashboards.

**Mobile Integration:** Live sensor streaming & mobile app support.

**How to Contribute:**

  1.Fork the repo

  2.Create a branch: git checkout -b feature/AmazingFeature

  3.Commit changes: git commit -m 'Add AmazingFeature'

  4.Push branch: git push origin feature/AmazingFeature

  5.Open a Pull Request

---


---
## 📄 License
This project is licensed under the MIT License.

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
```

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MaheenGitHub/human-activity-recognition-ML.git

---

## ⭐ Show Your Support

Liked this Repo?  give it a ⭐️ on GitHub!

