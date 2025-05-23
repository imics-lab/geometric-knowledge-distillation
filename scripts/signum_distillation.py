import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import numpy as np
import math
from typing import Dict, List, Tuple
import time

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.transformer import TransformerClassifier
from src.signum_data import get_distillation_dataloader_from_folder
from src.utils import compute_teacher_probabilities

# Training the distillation model

def get_total_Y_distil(loader):
    all_Y = []
    for batch in loader:  # Iterate through the batches in the loader
        Y_batch = batch['y']  # Extract 'y' from each batch
        all_Y.append(Y_batch)
    
    total_Y = torch.cat(all_Y, dim=0)  # Concatenate all 'y' labels along the batch dimension
    return total_Y

def distillation_train_model(model: nn.Module, 
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           device: torch.device,
                           alpha: float = 0.5,
                           beta: float = 0.5, 
                           temperature: float = 3.0,
                           epochs: int = 10,
                           lr: float = 1e-4) -> Dict:
    """
    Train model using knowledge distillation
    Args:
        model: student model to be trained
        train_loader: training data loader with DTW distances
        val_loader: validation data loader
        device: device to train on
        alpha: weight for cross-entropy loss
        beta: weight for KL divergence loss
        temperature: temperature for softening distributions
        epochs: number of training epochs
        lr: learning rate
    Returns:
        training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    train_y = get_total_Y_distil(train_loader)
    num_classes = 450 #len(set([item['y'] for item in train_loader.dataset]))  # Number of unique glosses

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]',
                         file=sys.stdout, leave=True)
        
        for batch in train_pbar:
            X_batch = batch['X'].to(device)
            y_batch = batch['y'].to(device)
            dtw_distances = batch['dtw_distances'].to(device)
            
            optimizer.zero_grad()
            
            # Student predictions
            student_logits = model(X_batch)
            student_probs = F.softmax(student_logits / temperature, dim=1)
            
            # Teacher probabilities from DTW distances
            teacher_probs = compute_teacher_probabilities(
                dtw_distances, train_y, num_classes, temperature).to(device)
            
            # Compute losses
            ce_loss = criterion(student_logits, y_batch)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * ce_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(student_logits, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            epoch_loss = running_loss / (train_pbar.n + 1)
            epoch_acc = 100 * correct / total
            
            train_pbar.set_postfix({
                'loss': f'{epoch_loss:.4f}',
                'acc': f'{epoch_acc:.2f}%'
            })
        
        # Validation step
        val_metrics = validate_model(model, val_loader, criterion, device, temperature)
        
        # Store metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"Validation Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.2f}%\n")
    
    # Return history dictionary
    return {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }

def validate_model(model: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  temperature: float) -> Dict:
    """Validate model on validation set"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch = batch['X'].to(device)
            y_batch = batch['y'].to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    return {
        'loss': val_loss / len(val_loader),
        'accuracy': 100 * correct / total
    }

# Load the pre-computed DTW distances
dtw_distances = np.load('../results/signum/dtw_distances_new_left_right.npy')
folder_path = '../data/Signum_numpy'
# Print the shape to verify
print(f"Loaded DTW distances shape: {dtw_distances.shape}")
print(f"Contains NaN: {np.isnan(dtw_distances).any()}")
print(f"Contains Inf: {np.isinf(dtw_distances).any()}")

# Verify non-negativity
is_non_negative = (dtw_distances >= 0).all()
print(f"All distances are non-negative: {is_non_negative}")

# Dinstance nomralization
dtw_distances = (dtw_distances - dtw_distances.min()) / (dtw_distances.max() - dtw_distances.min())
# dtw_distances = (dtw_distances - dtw_distances.mean()) / dtw_distances.std()

train_loader, val_loader, test_loader = get_distillation_dataloader_from_folder(
    folder=folder_path, dtw_distances=dtw_distances, batch_size=16
)


# Train model

input_dim = 126  # Number of features (channels)
seq_length = 80
num_classes = 450 #len(set([item['y'] for item in train_loader.dataset]))  # Number of unique glosses
num_heads = 9
print(num_classes)

# Initialize model
model = TransformerClassifier(input_dim=input_dim, num_heads=9, num_classes=num_classes)
print(model)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model with distillation
history = distillation_train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    alpha=0.5,  # Weight for cross-entropy loss
    beta=0.5,   # Weight for KL divergence loss
    temperature=3.0,  # Temperature for softening distributions
    epochs=400,
    lr=1e-4
)

# Test the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        X_batch = batch['X']
        y_batch = batch['y']

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

model.eval()
correct = 0
total = 0
start=time.time()
with torch.no_grad():
    for batch in test_loader:
        X_batch = batch['X']
        y_batch = batch['y']

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
end = time.time()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
print("Time per sample:", (end-start)*1000/total)


# Classification Metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        X_batch = batch['X']
        y_batch = batch['y']

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())  # Collect predictions
        all_labels.extend(y_batch.cpu().numpy())  # Collect true labels

# Calculate precision, recall, F1 score (macro averaged), and accuracy
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
accuracy = accuracy_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy * 100}%")
print(f"Test Precision (Macro): {precision}")
print(f"Test Recall (Macro): {recall}")
print(f"Test F1 Score (Macro): {f1}")

# Extract values from history
train_loss = history["train_loss"]
val_loss = history["val_loss"]
train_acc = history["train_acc"]
val_acc = history["val_acc"]
epochs = range(1, len(train_loss) + 1)

# Plot Loss Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label="Train Loss")
plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.grid()

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b-',label="Train Accuracy")
plt.plot(epochs, val_acc, 'r-', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Validation Accuracy")
plt.legend()
plt.grid()

plt.show()


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        X_batch = batch['X']
        y_batch = batch['y']

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Generate classification report and confusion matrix
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for batch in test_loader:
        X_batch = batch['X']
        y_batch = batch['y']
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(y_batch.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

# Classification Report
print("Classification Report:")
report = classification_report(true_labels, predicted_labels, target_names=train_loader.dataset.idx_to_gloss.values(), zero_division=0)
print(report)

# Confusion Matrix (optional for large datasets)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualizing diagonal counts for correct predictions
diag_counts = np.diag(conf_matrix)
plt.figure(figsize=(10, 5))
plt.bar(range(len(diag_counts)), diag_counts, color='blue')
plt.xlabel("Class Index")
plt.ylabel("Correct Predictions")
plt.title("Correct Predictions per Class (Diagonal of Confusion Matrix)")
plt.show()

# Heatmap for confusion matrix (cropped or full)
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix[:50, :50], annot=False, fmt='d', cmap="YlGnBu")
plt.title("Confusion Matrix (First 50 Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

