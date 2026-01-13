# Human Activity Recognition (HAR) Project

This repository contains a Machine Learning semester project on **Human Activity Recognition (HAR)** using smartphone sensor data.  
The project evaluates and compares **classical machine learning models**, **fully connected deep neural networks (DNN)**, and **sequence-based deep learning models (LSTM)** for activity classification.

---

## 📂 Repository Structure

human-activity-recognition-ML/
- Feature_Engineering_and_Classical_Machine_Learning_Classification/
  - Feature_Engineering_and_Classical_Machine_Learning_Classification.ipynb
  - A5_2_results.csv
  - Assignment_Deliverables_02.docx
- Fully_Connected_Deep_Neural_Network/
  - Fully_Connected_Deep_Neural_Network.ipynb
  - HAR_DNN_Model_5_3.keras
  - NN_template.py
  - A5_3_training_curves.png
  - Assignment_Deliverables_3.docx
  - Project_Master_Results.csv
  - label_encoder.pkl
- Time_Series_Data_Analysis_&_Preprocessing/
  - Time_Series_Data_Analysis_&_Preprocessing.ipynb
  - Assignment_Deliverables_01.pdf
  - train.zip
  - test.zip
- LICENSE
- README.md

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

- ### Training Curves for Fully Connected DNN
![Training Curves](Fully_Connected_Deep_Neural_Network/A5_3_training_curves.png)
  

---

### Deliverable 4 – Sequence-Based Deep Learning (LSTM)
> **Coming Soon** – This deliverable will contain a comparative study using sequence-based deep learning models trained on raw sequential sensor data, capturing temporal patterns for human activity recognition.

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
  ```


