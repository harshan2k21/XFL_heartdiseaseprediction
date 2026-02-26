import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from task import Net
from data_partining import partition_data

print("🔍 Loading TRAINED Global Architecture...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# --- NEW: Load the smart weights we just saved! ---
model.load_state_dict(torch.load("global_model.pth", weights_only=True))
model.eval() # Set model to evaluation mode

# Load data
trainsets = partition_data(num_clients=1) 
data_loader = torch.utils.data.DataLoader(trainsets[0], batch_size=100, shuffle=True)

background_data, _ = next(iter(data_loader))
background_data = background_data.to(device)

# Let's test on more patients for a better graph (50 instead of 10)
test_data, _ = next(iter(data_loader))
test_data = test_data[:50].to(device)

print("🧠 Running SHAP DeepExplainer on Federated Data...")

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data, check_additivity=False)

feature_names = ["Age", "Sex", "Chest Pain", "Rest BP", "Cholesterol", 
                 "Fasting Blood Sugar", "Rest ECG", "Max Heart Rate", 
                 "Exercise Angina", "Oldpeak", "Slope", "Vessels", "Thal"]

print("📊 Generating Final Explanation Plot...")

shap_values_disease = shap_values[1] if isinstance(shap_values, list) else shap_values

shap.summary_plot(shap_values_disease, test_data.cpu().numpy(), feature_names=feature_names, show=False)
plt.title("XAI: Feature Importance (Trained Global Model)")
plt.tight_layout()
plt.savefig("xai_heart_disease_trained.png", bbox_inches='tight')
print("✅ Saved TRAINED XAI graph as 'xai_heart_disease_trained.png'!")
