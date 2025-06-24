# 🫀 ECG-Abnormality-Detection

## 📌 Overview

This project implements a deep learning system for the automated classification of ECG signals into six diagnostic classes. By leveraging both the **MIT-BIH Arrhythmia** and **PTB Diagnostic ECG** datasets, this model introduces a **Transformer-based architecture** that outperforms traditional CNN and CNN-LSTM models in terms of classification accuracy and generalization. It is a step toward real-time, scalable, and accurate cardiovascular diagnostics.

---

## 🔍 Problem Statement

Manual ECG interpretation is time-consuming, error-prone, and resource-intensive. This project addresses:
- Class imbalance in ECG datasets
- Lack of generalization across multiple datasets
- The need for high-accuracy models for clinical deployment

---

## 🎯 Objectives

- Integrate MIT-BIH and PTB ECG datasets into a unified six-class classification task
- Apply SMOTE to balance minority classes
- Compare three models: CNN, CNN-LSTM, and Transformer
- Develop a highly accurate and generalizable ECG classifier using Transformers

---

## 🧠 Model Architectures

### ✅ Transformer-Based Model (Proposed)
- Conv1D layer + Positional Encoding
- Multi-Head Self-Attention (3 blocks)
- Feed-Forward Layers
- Global Average Pooling + Dense Softmax

### ✅ Baseline Models
- **Model A (CNN)**: Local feature extraction
- **Model B (CNN-LSTM)**: Temporal pattern learning via Bi-LSTM

---

## 🧪 Dataset Details

| Dataset     | Samples   | Timesteps | Source Classes                     |
|-------------|-----------|-----------|------------------------------------|
| MIT-BIH     | 109,446   | 187       | Normal, Atrial Premature, PVC, etc.|
| PTB         | 14,552    | 188 → 187 | Myocardial Infarction & others     |

Data is normalized, aligned, and augmented using **SMOTE** to address class imbalance.

---

## ⚙️ Technologies Used

- **TensorFlow 2.10**
- **Scikit-learn** (SMOTE, scaling, metrics)
- **Google Colab (GPU)** for training
- **Matplotlib** for visualizations

---

## 📊 Results

| Model        | Test Accuracy |
|--------------|---------------|
| CNN (Model A)| 99.45%        |
| CNN-LSTM     | 99.35%        |
| 🚀 Transformer | **99.7%**        |

- F1-Score (avg): **0.996**
- All classes classified with precision > 0.99
- Confusion matrix indicates minimal misclassification

---

## 📈 Visualizations

- Training & Validation Accuracy/Loss
- Confusion Matrix
- Class Distribution (before & after SMOTE)

---

## 🔮 Future Scope

- ⚡ Real-time ECG classification on edge devices
- 🧬 Use of GANs for improved minority class generation
- 👥 Inter-patient validation for clinical robustness

---

## 📁 Project Structure

```
ecg-abnormality-detection/
│
├── data/                 # Preprocessed ECG datasets
├── models/               # CNN, CNN-LSTM, Transformer models
├── notebooks/            # Training & evaluation notebooks
├── results/              # Accuracy plots, confusion matrices
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

---

## 📚 References

Key references include:
- Ghousia Begum et al., 2023 (Baseline CNN & LSTM models)
- Shukla & Alahmadi, 2021 (Transformer in ECG)
- PhysioNet MIT-BIH and PTB ECG Datasets

---

## 🙌 Contributors

- **Boddula Kaushik**  
- **Sagar Gujjunoori** *(Corresponding Author)*  
- **K. Gangadhara Rao**  
- **D. Jayaram**  

---

## 📬 Contact

For queries or collaboration:  
📧 **boddula.kaushik@gmail.com**
