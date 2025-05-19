# Sleep-Posture-Recognition-using-QCNN
A simple system that classifies three sleep postures using QCNN

Certainly! Here’s the complete, ready-to-copy `README.md` file for your GitHub repository:

```markdown
# 🛏️ Hybrid Quantum-Classical Neural Network for Sleep Posture Classification

This repository contains the implementation of a hybrid deep learning model combining classical neural networks and quantum computing to classify human sleep postures. It targets healthcare applications—particularly for patients with limited mobility, elderly individuals, and post-operative care—by enabling accurate posture detection using pressure sensor data.

---

## 📁 Repository Structure

| File/Filename             | Description |
|---------------------------|-------------|
| `experiment-ii.zip`       | Contains the original pressure sensor dataset. |
| `experiment-ii_labels.csv`| CSV file with posture labels corresponding to each sample in the dataset. |
| `qcnn3.ipynb`             | Jupyter notebook containing the full pipeline: preprocessing, dimensionality reduction, model training, and evaluation. |
| `qcnn3_model.pth`         | Saved hybrid model file (PyTorch + PennyLane). |
| `qcnn3_results.zip`       | Compressed folder containing evaluation plots and metrics. |
| `scaler3.pkl`             | Saved MinMaxScaler object used during preprocessing. |
| `umap3_model.pkl`         | UMAP dimensionality reduction model used to reduce feature space. |
| `other models.zip`        | Jupyter notebooks that had accuracy that qcnn3.ipynb. |

---

## 🚀 Project Overview

Incorrect sleep postures or long durations in a static position can lead to pressure ulcers and musculoskeletal strain. This project proposes a novel method for posture classification using a hybrid neural network that leverages quantum layers for enhanced expressiveness and classical layers for efficient learning.

The pressure matrices are extracted from sponge and air mattress settings using PhysioNet's Pressure Map Dataset. To handle high dimensionality and class imbalance, the project uses:
- MinMax normalization
- Dimensionality reduction using AutoEncoder + UMAP
- SMOTE for synthetic oversampling

The 11-dimensional latent features are then processed by a hybrid model with:
- Quantum feature embedding via `AngleEmbedding`
- Quantum convolution and pooling layers (PennyLane)
- Classical fully connected layers (PyTorch)

---

## 🧰 Requirements

Install required dependencies via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn umap-learn torch torchvision pennylane
```

---

## 📊 How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/sleep-posture-qcnn.git
cd sleep-posture-qcnn
```

### Extract Dataset

```bash
unzip experiment-ii.zip -d ./data/
```

### Open and Run the Jupyter Notebook

```bash
jupyter notebook qcnn3.ipynb
```

Make sure all files (`.pkl`, `.pth`, `.csv`) are in the same directory or adjust the paths in the notebook.

---

## 🧠 Model Details

### Preprocessing:
- Normalization with MinMaxScaler
- Dimensionality reduction using AutoEncoder and UMAP
- Class rebalancing using SMOTE

### Model Architecture:
- **Quantum layers:** AngleEmbedding, Quantum Conv, Quantum Pool (PennyLane)
- **Classical layers:** 5 Dense layers (PyTorch)
- **Optimizer:** Adam
- **Loss Function:** CrossEntropy

---

## 📈 Results

Found in `qcnn3_results.zip`:
- Accuracy and loss curves
- Confusion matrix
- Classification report (Precision, Recall, F1-score)

This architecture demonstrated strong performance in classifying postures while maintaining low computational overhead.

---

## 📚 Literature Survey Highlights

| Title | Authors | Year | Summary |
|------------------------------|---------------|------|-------------------------------------------|
| Quantum Convolutional Neural Networks | Cong, Choi, Lukin | 2019 | Demonstrated the ability to classify quantum phases with fewer parameters. |
| Posture Recognition via Transfer Learning | Hu, Tang, Tang | 2021 | Used shallow CNNs for real-time patient-specific posture recognition. |
| QCNN for COVID-19 X-ray Classification | Alharbi et al. | 2021 | Used QCNN with angle embedding to classify medical images. |
| Quantum Deep Learning | Farhi & Neven | 2018 | Proposed VQC and simplified QCNN showing quantum advantage. |
| QCNN for Text Classification | Wang et al. | 2021 | Applied QCNN to NLP tasks with promising results. |

---

## 💡 Key Takeaways

- Shows feasibility of quantum-classical hybrid models on real-world biomedical data.
- Enables scalable and efficient classification on high-dimensional, imbalanced datasets.
- Opens opportunities for smart bed technologies in hospitals and home care.

---

## 📜 Citation

```bibtex
@misc{qcnn_sleep_posture_classification_using_qcnn,
  author = {Kavyasri V J},
  title = {Hybrid Quantum-Classical Neural Network for Sleep Posture Classification},
  year = {2025},
  note = {GitHub repository},
  url = {https://github.com/kav-xx/sleep-posture-classification-using-qcnn}
}
```

---

## 👩‍💻 Author

**Your Name**  
📧 Email: kavyasrivj271@gmail.com  
🏫 Institution: [SSN College of Engineering]  

---

## 🛡️ License

This project is released under the MIT License. See `LICENSE` for more information.

---

## ⚛️ Built with quantum curiosity and classical logic to make healthcare smarter. 🚀

```

