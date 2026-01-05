import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../dataset_creator/generated_dataset')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
ANNOTATIONS_FILE = os.path.join(DATA_DIR, 'annotations.csv')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dataset Definition ---
class ShapeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {'inside': 0, 'outside': 1}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L') # Convert to grayscale
        label_str = self.img_labels.iloc[idx, 1]
        label = self.label_map[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_name

# --- Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1 x 128 x 128
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 32 x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 64 x 16 x 16
        
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2) # 2 classes: inside, outside

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # Hook for GradCAM
    def activations_hook(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients
    
    def get_activations(self, x):
        # Allow running partial forward pass to get conv outputs
        return self.forward_conv(x)
    
    def forward_conv(self, x):
        # Run convolutions only
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x)) # activations before pooling provided more spatial info? 
        # Actually standard GradCAM usually takes the last conv output *before* the final FC/GAP.
        # Here conv3 output is processed by pool. Let's capture conv3 ACTIVATIONS before pool?
        # But we defined pool in forward. 
        # Let's refactor slightly for clarity in hook attachment.
        return x

# A cleaner way to implement GradCAM without modifying model structure too much for simplicity
# We will attach hooks dynamically in the GradCAM logic.

# --- Training Loop ---
def train(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels, _) in enumerate(dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

# --- GradCAM Logic ---
def generate_gradcam(model, image_tensor, target_class=None):
    model.eval()
    
    # Register hooks
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
        
    def save_activation(module, input, output):
        activations.append(output)
        
    # Hook on the last convolutional layer (conv3)
    # Note: In our model definition, conv3 is followed by pool in forward(), 
    # but we want the conv3 feature map.
    # Let's attach to conv3 directly.
    handle_grad = model.conv3.register_full_backward_hook(lambda m, grad_i, grad_o: save_gradient(grad_o[0]))
    handle_act = model.conv3.register_forward_hook(save_activation)
    
    # Forward pass
    output = model(image_tensor.unsqueeze(0).to(DEVICE))
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
        
    # Backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    # Get gradients and activations
    grads = gradients[0].cpu().data.numpy()[0] # [channels, h, w]
    acts = activations[0].cpu().data.numpy()[0] # [channels, h, w]
    
    # Cleanup hooks
    handle_grad.remove()
    handle_act.remove()
    
    # GAP on gradients
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of activation maps
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    cam = cv2.resize(cam, (128, 128))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam, target_class, output

# --- Main Execution ---
if __name__ == "__main__":
    import json
    import shutil
    
    print(f"Using device: {DEVICE}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Data Loader
    full_dataset = ShapeDataset(ANNOTATIONS_FILE, IMAGES_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Model Setup
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Phase
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")
        
    print("Training complete.")
    
    # --- Visualization Phase (HTML Generation) ---
    print("Generating GradCAM gallery...")
    
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'gradcam_gallery')
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)
    
    model.eval()
    data_iter = iter(test_loader)
    classes = ['Inside', 'Outside']
    gallery_data = []
    
    NUM_SAMPLES = 50 # Generate for 50 test images
    
    for i in range(min(NUM_SAMPLES, len(test_dataset))):
        img_tensor, label, img_name = next(data_iter)
        img_tensor = img_tensor.squeeze(0)
        
        # GradCAM
        cam, pred_class_idx, output_logits = generate_gradcam(model, img_tensor)
        
        # Process images
        img_np = img_tensor.cpu().numpy().squeeze() * 255
        img_np = img_np.astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_overlay = heatmap + np.float32(img_bgr) / 255
        cam_overlay = cam_overlay / np.max(cam_overlay)
        cam_overlay = np.uint8(255 * cam_overlay)
        
        # Save images
        orig_filename = f"sample_{i:03d}_orig.png"
        grad_filename = f"sample_{i:03d}_grad.png"
        
        cv2.imwrite(os.path.join(RESULTS_DIR, orig_filename), img_bgr)
        cv2.imwrite(os.path.join(RESULTS_DIR, grad_filename), cam_overlay)
        
        # Probabilities
        probs = torch.softmax(output_logits, dim=1)[0]
        conf = probs[pred_class_idx].item()
        
        true_lbl = classes[label.item()]
        pred_lbl = classes[pred_class_idx]
        is_correct = (true_lbl == pred_lbl)
        
        gallery_data.append({
            'id': i,
            'orig_img': orig_filename,
            'grad_img': grad_filename,
            'true_label': true_lbl,
            'pred_label': pred_lbl,
            'confidence': f"{conf:.1%}",
            'is_correct': is_correct,
            'filename': img_name[0]
        })

    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GradCAM Analysis Results</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #444; }}
        .controls {{ text-align: center; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.2s; }}
        .card:hover {{ transform: translateY(-5px); }}
        .card-header {{ padding: 10px; background: #f8f9fa; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; font-weight: bold; font-size: 0.9em; }}
        .status-correct {{ color: green; }}
        .status-wrong {{ color: red; }}
        .images {{ display: flex; }}
        .images img {{ width: 50%; height: auto; display: block; }}
        .card-body {{ padding: 15px; font-size: 0.9em; }}
        .metric {{ margin-bottom: 5px; }}
        .label {{ font-weight: 600; color: #555; }}
    </style>
</head>
<body>
    <h1>GradCAM Model Analysis</h1>
    <div class="controls">
        Analyzing {len(gallery_data)} test samples. 
        <strong>Accuracy on this batch: {sum(d['is_correct'] for d in gallery_data)/len(gallery_data):.1%}</strong>
    </div>
    <div class="grid">
"""
    
    for item in gallery_data:
        status_class = "status-correct" if item['is_correct'] else "status-wrong"
        status_icon = "✓" if item['is_correct'] else "✗"
        
        html_content += f"""
        <div class="card">
            <div class="card-header">
                <span>#{item['id']}</span>
                <span class="{status_class}">{status_icon} {item['pred_label']}</span>
            </div>
            <div class="images">
                <img src="{item['orig_img']}" alt="Original" title="Original Input">
                <img src="{item['grad_img']}" alt="GradCAM" title="GradCAM Overlay">
            </div>
            <div class="card-body">
                <div class="metric"><span class="label">True Label:</span> {item['true_label']}</div>
                <div class="metric"><span class="label">Prediction:</span> {item['pred_label']}</div>
                <div class="metric"><span class="label">Confidence:</span> {item['confidence']}</div>
                <div class="metric" style="font-size:0.8em; color:#888; margin-top:5px;">{item['filename']}</div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""
    
    with open(os.path.join(RESULTS_DIR, 'index.html'), 'w') as f:
        f.write(html_content)
        
    print(f"Gallery generated at {os.path.join(RESULTS_DIR, 'index.html')}")
