# Human Activity Recognition (HAR) Project


**KineticSense AI** is a state-of-the-art Machine Learning framework for recognizing human physical activities from smartphone sensor data (Accelerometer & Gyroscope).  
This project demonstrates **end-to-end ML pipelines**: from raw signal preprocessing → feature engineering → classical ML → deep learning (DNN & LSTM) → evaluation & visualization.  

It highlights **high-performance models**, **temporal sequence learning**, and **real-world applications** in healthcare, IoT, smart homes, and worker safety.  
> Perfect for anyone exploring **ML product development, AI research, or embedded systems deployment**.

🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-lightgrey?style=for-the-badge&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.22-blue?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellow?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11-purple?style=for-the-badge&logo=seaborn)

---
## 🌟 Key Highlights

**Multiple ML Pipelines:** Classical ML (SVM, Random Forest, XGBoost) + Deep Learning (Fully Connected DNN, LSTM, GRU)

**High Accuracy:**

- SVM: 96% | F1: 0.96
- DNN: 92.94% | F1: 0.9286
- LSTM: 91.25% | F1: 0.914

**Visual Insights:** Confusion matrices, training curves, feature distributions
**End-to-End Workflow:** From raw sensor data → preprocessing → modeling → evaluation → visualization

---
## 📁 Project Deliverables
| Deliverable | Focus                              | Key Files / Artifacts                                                                                                                                     |
| ----------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1          | Data Analysis & Preprocessing      | `Time_Series_Data_Analysis_&_Preprocessing.ipynb`, `Assignment_Deliverables_01.pdf`                                                                       |
| D2          | Feature Engineering & Classical ML | `Feature_Engineering_and_Classical_Machine_Learning_Classification.ipynb`, `A5_2_results.csv`, `Assignment_Deliverables_02.docx`                          |
| D3          | Fully Connected DNN                | `Fully_Connected_Deep_Neural_Network.ipynb`, `HAR_DNN_Model_5_3.keras`, `A5_3_training_curves.png`, `label_encoder.pkl`, `Assignment_Deliverables_3.docx` |
| D4          | Sequence Models (RNN / LSTM / GRU) | `Sequence_Based_Deep_Learning_Classification.ipynb`, `best_lstm.keras`, `features.txt`, `activity_labels.txt`, `Assignment_deliverable_4.docx`            |

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
## 💡 How It Works

**Data Preprocessing:** Timestamp generation, chronological sorting, missing/duplicate handling, normalization

**Feature Engineering:** 561 handcrafted features for classical ML and DNN models

**Modeling:** Classical ML pipelines → DNN → Sequence models (LSTM/GRU)

**Evaluation & Metrics:** Accuracy, F1-score, confusion matrices

**Visualization:** Feature distributions, training curves, prediction results

---

## 📊 Performance Metrics

| Model | Accuracy | F1-Score | Notes |
| :--- | :--- | :--- | :--- |
| SVM | 96% | 0.96 | Strong baseline for low-power inference |
| DNN | 92.94% | 0.928 | Works well on high-dimensional features |
| LSTM | 91.25% | 0.914 | Captures temporal dependencies |

**Inference Speed:** ~0.47ms per activity window (LSTM).  
**Stability:** 30% Dropout prevents overfitting, ensuring robust predictions.

--- 
## 💡 Industrial Use Cases

**Healthcare:** Fall detection for elderly care.

**Logistics:** Worker ergonomics monitoring to reduce injuries.

**Insurance / FinTech:** Activity-based insurance premium modeling.

**Smart Home / IoT:** Gesture-based home automation.

---
## 🤝 Collaboration & Contributions



**Opportunities to contribute include:**
- **Edge Deployment:** Convert models to TFLite for ESP32 / Arduino  
- **Interactive Dashboards:** Build Streamlit or Flutter real-time interfaces  
- **Mobile Integration:** Stream live sensor data and integrate with apps  

**How to Contribute:**  
1. Fork the repository  
2. Create a branch: `git checkout -b feature/AmazingFeature`  
3. Commit your changes: `git commit -m 'Add AmazingFeature'`  
4. Push to your branch: `git push origin feature/AmazingFeature`  
5. Open a Pull Request and join the discussion!
6. This project is actively evolving!

**Note:** This repository is currently **private**.  
If you would like to contribute or explore the code, please request access via:
- [LinkedIn Profile](https://www.linkedin.com/in/maheenfatimaa/ "Visit my LinkedIn for details")
- [GitHub Profile]((https://github.com/MaheenGitHub)"Visit my GitHub for details")
- [Email Me](mailto:maheen19pgc@gmail.com "Send me an email")




Let’s make **HAR solutions smarter, faster, and more accessible together!**


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

Liked this Repo?  Give it a ⭐️ on GitHub!

