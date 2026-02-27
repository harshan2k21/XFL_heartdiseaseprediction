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

*Note: This repository contains the fully functional **simulated multi-node environment** utilizing Flower's Virtual Client Engine and Ray to prove the architectural concept on local hardware.*

## ✨ Key Features
This framework solves the "Clinical Triad" of medical AI deployment:
1. **Decentralization & Stability (FedProx):** Orchestrates training across simulated distributed nodes. Solves the issue of Non-IID clinical data (Client Drift) using the **FedProx** optimization algorithm (Proximal penalty `mu = 0.01`).
2. **Cryptographic Privacy (Opacus DP-SGD):** Enforces strict mathematical privacy by clipping local gradients (`max_grad_norm = 1.0`) and injecting calibrated Gaussian noise (`noise_multiplier = 0.1`) before aggregation, making the model resistant to Model Inversion Attacks.
3. **Zero-Leakage Data Engineering:** All data preprocessing (Median Imputation, StandardScaler) is strictly localized to the host nodes to prevent global statistical leakage.
4. **Clinical Interpretability (SHAP):** De-black-boxes the final PyTorch neural network using SHAP DeepExplainer, providing doctors with feature-level attribution scores (e.g., Age, Sex, Max HR) for every diagnosis.

## 📊 Performance & Results
The framework was evaluated on a custom Cardiovascular Disease dataset (270 patients, 13 clinical features) computationally partitioned across 5 isolated hospital nodes. Despite the introduction of cryptographic noise and decentralized splitting, the global model achieved:

* **Peak Accuracy:** 83.5%
* **Area Under the Curve (AUC):** 0.91
* **F1-Score:** 0.80

*(Note: Add your ROC Curve and SHAP Summary Plot images here in your actual repository)*
## ⚙️ Architecture
* **Global Model:** Multi-Layer Perceptron (MLP) in PyTorch (13 Input $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 2 Output).
* **Federated Engine:** Flower (Flwr) utilizing the `start_simulation` Virtual Client Engine backed by Ray for efficient concurrent execution on a single machine.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/harshan2k21/FedHealth-XAI.git](https://github.com/harshan2k21/FedHealth-XAI.git)
   cd FedHealth-XAI




   Install the required dependencies:

Bash
pip install torch torchvision flwr opacus shap pandas scikit-learn ray
💻 Usage Instructions
To run the federated pipeline simulation (which will automatically partition the data, spin up the 5 virtual hospital clients using Ray, execute 5 communication rounds, and generate the global model):

Bash
python main.py
(Replace main.py with the exact name of your execution script if different).

🔭 Future Scope
While this repository successfully proves the mathematical and architectural viability of the pipeline via simulation, the next deployment phase involves transitioning from Ray virtual clients to physical distributed hardware. Future iterations will involve sharding the dataset across physical edge devices (e.g., 5 independent laptops) and orchestrating the FedProx aggregation over a local network via gRPC.

👨‍💻 Author
Harshan R
