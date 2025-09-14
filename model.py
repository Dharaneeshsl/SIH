import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.amp import GradScaler, autocast
import seaborn as sns

# ========== CLASS LABELS ==========
class_labels = {
    "ayrshire": 0,
    "brown_swiss": 1,
    "guernsey": 2,
    "hariana": 3,
    "holstein_friesian": 4,
    "jersey": 5
}
label_names = list(class_labels.keys())

# ========== PATHS ==========
train_dirs = {
    "ayrshire": r"D:\breed model\data\training\COW\Ayrshire\New folder",
    "brown_swiss": r"D:\breed model\data\training\COW\Brown Swiss\New folder",
    "guernsey": r"D:\breed model\data\training\COW\Guernsey\New folder",
    "hariana": r"D:\breed model\data\training\COW\Hariana\New folder",
    "holstein_friesian": r"D:\breed model\data\training\COW\Holstein Friesian\New folder",
    "jersey": r"D:\breed model\data\training\COW\Jersey\New folder"
}
valid_dirs = {
    "ayrshire": r"D:\breed model\data\training\COW\Ayrshire\Valid",
    "brown_swiss": r"D:\breed model\data\training\COW\Brown_Swiss\Valid",
    "guernsey": r"D:\breed model\data\training\COW\Guernsey\Valid",
    "hariana": r"D:\breed model\data\training\COW\Hariana\Valid",
    "holstein_friesian": r"D:\breed model\data\training\COW\Holstein Friesian\Valid",
    "jersey": r"D:\breed model\data\training\COW\Jersey\Valid"
}

# ========== CONFIG ==========
config = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 10,
    "patience": 5,
    "num_workers": 2,  # Adjust for Windows if needed
    "results_dir": r"D:\buffalo_dataset\results",
    "model_save_path": "multi_breed_efficientnetb0_6class.pth",
    "metrics_csv": "multi_breed_metrics.csv"
}

# Create results directory if it doesn't exist
os.makedirs(config["results_dir"], exist_ok=True)

# ========== TRANSFORM ==========
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== DATASET ==========
class BuffaloBreedDataset(Dataset):
    def __init__(self, directory, label, transform=None):
        self.paths = list(Path(directory).rglob("*.[jp][pn]g")) + list(Path(directory).rglob("*.jpeg"))
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, self.label
        except:
            return self.__getitem__((idx + 1) % len(self))

# ========== LOADERS ==========
def make_loader(dirs_dict, is_train=True):
    datasets = []
    transform = train_transform if is_train else valid_transform
    for cls, path in dirs_dict.items():
        datasets.append(BuffaloBreedDataset(path, class_labels[cls], transform))
    return ConcatDataset(datasets)

train_loader = DataLoader(make_loader(train_dirs, is_train=True), 
                          batch_size=config["batch_size"], 
                          shuffle=True, 
                          num_workers=config["num_workers"], 
                          pin_memory=True)
valid_loader = DataLoader(make_loader(valid_dirs, is_train=False), 
                          batch_size=config["batch_size"], 
                          shuffle=False, 
                          num_workers=config["num_workers"], 
                          pin_memory=True)

# ========== MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_labels))
model = model.to(device)

# ========== TRAIN CONFIG ==========
# Class weights for imbalance
class_counts = [len(list(Path(path).rglob("*.[jp][pn]g")) + list(Path(path).rglob("*.jpeg"))) for path in train_dirs.values()]
class_weights = 1.0 / (np.array(class_counts) + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
scaler = GradScaler('cuda')

# CSV metrics file setup
csv_path = os.path.join(config["results_dir"], config["metrics_csv"])
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Val Accuracy", "Val Precision", "Val Recall", "Val F1"])

# For plotting later
history = {"train_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}

# ========== TRAIN LOOP ==========
def train(model, epochs=config["epochs"], patience=config["patience"]):
    best_f1 = 0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # ========== VALIDATION ==========
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # === Metrics ===
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        report = classification_report(y_true, y_pred, target_names=label_names, digits=2)
        print(f"\n Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        print(" Classification Report:\n", report)

        # === Save metrics to CSV ===
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, acc, precision, recall, f1])

        # Store for plotting
        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        # LR Scheduler step
        scheduler.step(f1)

        # Early Stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config["results_dir"], "best_" + config["model_save_path"]))
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # === Final Epoch: Paper-ready Results ===
        if epoch + 1 == epochs or patience_counter >= patience:
            # Confusion Matrix (percentage form)
            cm = confusion_matrix(y_true, y_pred)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names)
            plt.title(f"Final Confusion Matrix (%) - Epoch {epoch+1}")
            plt.xlabel("Predicted Breed")
            plt.ylabel("True Breed")
            plt.tight_layout()
            plt.savefig(os.path.join(config["results_dir"], "multi_breed_confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()

            # Per-class metrics
            per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            support = np.bincount(y_true)

            df_results = pd.DataFrame({
                "Breed": label_names,
                "Precision": per_class_precision,
                "Recall": per_class_recall,
                "F1-score": per_class_f1,
                "Support": support
            })

            # Add Macro & Weighted Averages
            macro_vals = [precision_score(y_true, y_pred, average='macro'),
                          recall_score(y_true, y_pred, average='macro'),
                          f1_score(y_true, y_pred, average='macro'),
                          np.sum(support)]
            weighted_vals = [precision_score(y_true, y_pred, average='weighted'),
                             recall_score(y_true, y_pred, average='weighted'),
                             f1_score(y_true, y_pred, average='weighted'),
                             np.sum(support)]

            df_results = pd.concat([
                df_results,
                pd.DataFrame([["Macro Avg"] + macro_vals, ["Weighted Avg"] + weighted_vals],
                             columns=df_results.columns)
            ], ignore_index=True)

            # Save as CSV
            df_results.to_csv(os.path.join(config["results_dir"], "multi_breed_per_class_metrics.csv"), index=False)

            # Save LaTeX table (3 decimal places)
            with open(os.path.join(config["results_dir"], "multi_breed_per_class_metrics.tex"), "w") as texfile:
                texfile.write(df_results.to_latex(index=False, float_format="%.3f"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(config["results_dir"], config["model_save_path"]))
    print(f" Model saved as {config['model_save_path']} in {config['results_dir']}")

    # === Plot metrics (Research paper quality) ===
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    plt.figure(figsize=(12, 8), dpi=300)
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(history["val_precision"]) + 1), history["val_precision"], label="Val Precision", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Validation Precision")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(history["val_f1"]) + 1), history["val_f1"], label="Val F1", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Score")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "multi_breed_metrics_subplots.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ========== RUN ==========
if __name__ == '__main__':
    train(model)