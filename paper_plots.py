import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from task import Net
from data_partining import partition_data

print("📊 Generating Research Graphs...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(input_dim=13).to(device) # Change this dim if using Breast Cancer (30)
model.load_state_dict(torch.load("global_model.pth", weights_only=True))
model.eval()

# Load all data for a final global evaluation
trainsets = partition_data(num_clients=1)
data_loader = torch.utils.data.DataLoader(trainsets[0], batch_size=200)

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for data, labels in data_loader:
        data = data.to(device)
        outputs = model(data)
        probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1] # Get prob for Class 1
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# --- Plot 1: Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
plt.title('Global Model Confusion Matrix')
plt.ylabel('Actual Diagnosis')
plt.xlabel('Predicted Diagnosis')
plt.tight_layout()
plt.savefig("paper_confusion_matrix.png", dpi=300)
print("✅ Saved Confusion Matrix!")

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("paper_roc_curve.png", dpi=300)
print("✅ Saved ROC Curve!")
