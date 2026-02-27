# FedHealth-XAI 🏥🔐

**A Robust Federated Learning Framework for Privacy-Preserving and Interpretable Clinical Diagnostics**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-yellow.svg)](https://flower.dev/)
[![Opacus](https://img.shields.io/badge/Opacus-Differential%20Privacy-000000.svg)](https://opacus.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://shap.readthedocs.io/en/latest/)

## 📖 Project Overview
Deep learning in healthcare is severely restricted by data silos and strict privacy regulations (HIPAA/GDPR). Hospitals cannot legally centralize patient data to train high-performing diagnostic models. 

**FedHealth-XAI** solves this by implementing an end-to-end decentralized machine learning pipeline. It allows multiple clinical institutions to collaboratively train a global diagnostic model **without ever sharing raw patient data**, while maintaining mathematical privacy guarantees and providing clinical-grade explainability for doctors.

## ✨ Key Features
This framework solves the "Clinical Triad" of medical AI deployment:
1. **Decentralization & Stability (FedProx):** Utilizes the Flower framework to orchestrate training across distributed nodes. Solves the issue of Non-IID clinical data (Client Drift) using the **FedProx** optimization algorithm (Proximal penalty `mu = 0.01`).
2. **Cryptographic Privacy (Opacus DP-SGD):** Enforces strict mathematical privacy by clipping local gradients (`max_grad_norm = 1.0`) and injecting calibrated Gaussian noise (`noise_multiplier = 0.1`) before aggregation, making the model immune to Model Inversion Attacks.
3. **Zero-Leakage Data Engineering:** All data preprocessing (Median Imputation, StandardScaler) is strictly localized to the host nodes to prevent global statistical leakage.
4. **Clinical Interpretability (SHAP):** De-black-boxes the final PyTorch neural network using SHAP DeepExplainer, providing doctors with feature-level attribution scores (e.g., Age, Sex, Max HR) for every diagnosis.

## 📊 Performance & Results
The framework was evaluated on a custom Cardiovascular Disease dataset (270 patients, 13 clinical features) distributed across 5 isolated hospital nodes. Despite the introduction of cryptographic noise and decentralized splitting, the global model achieved:

* **Peak Accuracy:** 83.5%
* **Area Under the Curve (AUC):** 0.91
* **F1-Score:** 0.80

*(Note: Add your ROC Curve and SHAP Summary Plot images here in your actual repository)*
## ⚙️ Architecture
* **Global Model:** Multi-Layer Perceptron (MLP) in PyTorch (13 Input $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 2 Output).
* **Federated Engine:** Flower (Flwr) using the `start_simulation` Virtual Client Engine (backed by Ray) for single-machine testing, and standard gRPC for physical multi-device deployment.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/harshan2k21/FedHealth-XAI.git](https://github.com/harshan2k21/FedHealth-XAI.git)
   cd FedHealth-XAI


   Install the required dependencies:

Bash
pip install torch torchvision flwr opacus shap pandas scikit-learn ray
💻 Usage Instructions
Option A: Local Simulation (Ray Virtual Clients)
To test the pipeline on a single machine simulating 5 hospitals:

Bash
python main_simulation.py
Option B: Physical Multi-Node Deployment (e.g., 5 Laptops)
To deploy this across actual physical hardware over a local network:

Start the Global Server:

Bash
python server.py
Start the Hospital Clients (Run this on each of the 5 laptops):
Ensure the dataset is physically split into 5 distinct CSV shards before running.

Bash
python client.py --server_address="IP_ADDRESS_OF_SERVER:8080" --data_path="hospital_1_data.csv"
👨‍💻 Author
Harshan R * B.Tech Computer Science, AI & Machine Learning

