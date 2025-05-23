import os, sys
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
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.transformer import EnhancedTransformerClassifier
from src.mhad_data import get_all_dataloaders

train_loader, test_loader = get_all_dataloaders('../data/Skeleton_numpy', batch_size=16)
for data, labels in train_loader:
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    break


# ## Training model

def train_model(model, train_loader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()  # For classification task
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)  # Move model to CUDA if available

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Compute loss
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")



input_dim = 60  # Number of features (e.g., 75 landmarks × 3 coordinates)
num_classes = 27 #len(set([item['y'] for item in train_loader.dataset]))  # Number of unique glosses
num_heads = 12
print(num_classes)

# Initialize model
model = EnhancedTransformerClassifier(input_dim=input_dim, num_heads = num_heads, num_classes=num_classes)
print(model)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# Train the model
train_model(model, train_loader, device=device, epochs=2000, lr=1e-4)

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


# ## Test results
model.eval()
correct = 0
total = 0
start=time.time()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
end = time.time()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
print("Time per sample:", (end-start)*1000/total)


# ## Print metrics

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())  # Collect predictions
        all_labels.extend(y_batch.cpu().numpy())  # Collect true labels

# Calculate precision, recall, F1 score (macro averaged), and accuracy
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
accuracy = accuracy_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy * 100}%")
print(f"Test Precision (Macro): {precision}")
print(f"Test Recall (Macro): {recall}")
print(f"Test F1 Score (Macro): {f1}")

