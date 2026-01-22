# Human Activity Recognition (HAR) Project


This repository contains a Machine Learning semester project on Human Activity Recognition (HAR) using smartphone sensor data.

The project evaluates and compares classical machine learning models, fully connected deep neural networks (DNN), and sequence-based deep learning models (LSTM) for activity classification.

📘 **Project Walkthrough & Technical Blog**  
I’ve documented the complete technical journey, design decisions, experiments, and lessons learned in a detailed Hashnode blog:

Read here:  https://maheen-ai.hashnode.dev/human-activity-recognition-with-machine-learning


---

### 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-lightgrey?style=for-the-badge&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.22-blue?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellow?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11-purple?style=for-the-badge&logo=seaborn)
---


###  Motivation

Human Activity Recognition (HAR) is critical for **healthcare monitoring, workplace safety, and smart environments**.  
KineticSense AI aims to deliver **accurate, real-time activity detection** using smartphone sensors, providing a practical solution for **AI-powered IoT devices and wearable systems**.



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

## 📊 Dataset Overview

- **Dataset Name:** UCI Human Activity Recognition (HAR) Using Smartphones  
- **Source:** [Kaggle / UCI Repository](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)  
- **Sensor Types:** Triaxial Accelerometer & Gyroscope  
- **Number of Samples:** 10,299 total (7,352 Training, 2,947 Testing)  
- **Number of Classes:** 6 (LAYING, SITTING, STANDING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)  
- **Features:** 561 engineered numeric features (time & frequency domain)  
- **Sample Representation:** Each row = 2.56-second sensor window  
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

**Inference Speed:** ~0.47ms per activity window (LSTM)  
**Stability:** 30% Dropout prevents overfitting, ensuring robust predictions  

---

## 💡 Industrial Use Cases

- **Healthcare:** Fall detection for elderly care  
- **Logistics:** Worker ergonomics monitoring to reduce injuries  
- **Insurance / FinTech:** Activity-based insurance premium modeling  
- **Smart Home / IoT:** Gesture-based home automation  

---

## 🔮 Future Work

- Deploy models on **mobile and edge devices** (TFLite / CoreML)  
- Expand dataset to include **more activities and real-world noise**  
- Integrate with **real-time dashboards** for monitoring and analytics  
- Explore **multimodal sensors** (accelerometer + gyroscope + magnetometer) for enhanced accuracy  

---

## 🤝 Collaboration & Contributions

**Opportunities to contribute:**  
- **Edge Deployment:** Convert models to TFLite for ESP32 / Arduino  
- **Interactive Dashboards:** Build Streamlit or Flutter real-time interfaces  
- **Mobile Integration:** Stream live sensor data and integrate with apps  

**How to Contribute:**  
1. Fork the repository  
2. Create a branch: `git checkout -b feature/AmazingFeature`  
3. Commit changes: `git commit -m 'Add AmazingFeature'`  
4. Push to branch: `git push origin feature/AmazingFeature`  
5. Open a Pull Request  

**Note:** This repository is currently **private**.  
If you would like to contribute or explore the code, request access via:  

- [LinkedIn Profile](https://www.linkedin.com/in/maheenfatimaa/ "Visit my LinkedIn for details")  
- [GitHub Profile](https://github.com/MaheenGitHub "Visit my GitHub for details")  
- [Email Me](mailto:maheen19pgc@gmail.com "Send me an email")  

> Let’s make **HAR solutions smarter, faster, and more accessible together!**

---



## 📄 License
This project is licensed under the MIT License.

## ⚙️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost
```

## 🚀 How to Run
Clone the repository:
```bash
git clone https://github.com/MaheenGitHub/human-activity-recognition-ML.git
```


## ⭐ Show Your Support

Liked this Repo?  Give it a ⭐️ on GitHub!

