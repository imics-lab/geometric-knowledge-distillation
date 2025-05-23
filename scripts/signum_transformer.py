import sys, os
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import math
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer import TransformerClassifier
from src.signum_data import get_all_dataloaders

# Code for the training of the transformer followed by validation steps. It also plots the loss and accuracy curves for training and validation

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                         file=sys.stdout, leave=True)
        
        for X_batch, y_batch in train_pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            epoch_loss = running_loss / (train_pbar.n + 1)
            epoch_acc = 100 * correct / total
            train_pbar.set_postfix({
                'loss': f'{epoch_loss:.4f}',
                'acc': f'{epoch_acc:.2f}%'
            })

        train_pbar.close()
        
        # Store training metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]', 
                       file=sys.stdout, leave=True)
        
        with torch.no_grad():
            for X_batch, y_batch in val_pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                
                current_val_loss = val_loss / (val_pbar.n + 1)
                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{val_acc:.2f}%'
                })

        val_pbar.close()
        
        # Store validation metrics
        val_losses.append(current_val_loss)
        val_accuracies.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"Validation Loss: {current_val_loss:.4f}, Accuracy: {val_acc:.2f}%\n")

    # Plot training history
    epochs_range = range(1, epochs + 1)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return history dictionary for further analysis if needed
    history = {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'val_loss': val_losses,
        'val_acc': val_accuracies
    }
    
    return history


#  Train model
folder_path = '../data/Signum_numpy' 
train_loader, val_loader, test_loader = get_all_dataloaders(folder_path, batch_size=16)

input_dim = 126  # Number of features (channels)
seq_length = 80
all_labels = np.array([label for _, label in train_loader.dataset])
num_classes = len(np.unique(all_labels))
num_heads = 9
print(num_classes)

# Initialize model
model = TransformerClassifier(input_dim=input_dim, num_heads = 9, num_classes=num_classes)
print(model)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train_model(model, train_loader, val_loader, device=device, epochs=400, lr=1e-4)

# Test the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
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
total_instances = 0
start_time = time.time()

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        batch_size = X_batch.size(0)
        total_instances += batch_size
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(y_batch.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

end_time = time.time()
total_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
per_instance_time_ms = total_time_ms / total_instances

# Print inference time information
print(f"Total inference time: {total_time_ms:.2f} ms")
print(f"Inference time per instance: {per_instance_time_ms:.2f} ms")

# Classification Report
print("\nClassification Report:")
report = classification_report(true_labels, predicted_labels, zero_division=0, digits=3)
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





