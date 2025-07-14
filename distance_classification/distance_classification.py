
from __future__ import print_function, division

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# Enable CUDA benchmarking
torch.backends.cudnn.benchmark = True
plt.ion()  # Interactive plotting

# =====================
# Configuration
# =====================
IMG_SIZE = 224
BATCH_SIZE = 16
DATA_DIR = './dataset/ku_net_dataset'  # Update this if needed
MODEL_PATH = './MobileNetV2/mobilenetv2_best.pth'

# =====================
# Data Transforms
# =====================
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# =====================
# Load Validation Dataset
# =====================
image_datasets = {
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
}
dataloaders = {
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=0)
}
class_names = image_datasets['val'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =====================
# Visualization Helper
# =====================
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(std * inp + mean, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)

# =====================
# Visualize Predictions
# =====================
def visualize_model(model, num_images=4):
    images_shown = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            percentages = torch.nn.functional.softmax(outputs, dim=1) * 100

            for j in range(inputs.size(0)):
                images_shown += 1
                ax = plt.subplot(num_images // 2, 2, images_shown)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_shown == num_images:
                    return

# =====================
# Load Trained MobileNetV2 Model
# =====================
model = models.mobilenet_v2(pretrained=False)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =====================
# Run Visualization
# =====================
visualize_model(model)
plt.ioff()
plt.show()
