# DL-EXP-7
# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

In practical scenarios, images often contain noise that degrades the performance of computer vision models. A convolutional autoencoder learns compressed representations of images and reconstructs them, which can be used to remove noise.

* **Dataset**:
   MNIST (28×28 grayscale images of handwritten digits)
* **Noise**:
  Gaussian noise will be added to simulate real-world scenarios

## DESIGN STEPS

### Step 1: Setup Environment
Import required libraries: PyTorch, torchvision, matplotlib, and others for data handling and visualization.

### Step 2: Load Dataset
Download the MNIST dataset and apply transformations to convert images to tensors suitable for training.

### Step 3: Introduce Noise
Add Gaussian noise to the training and testing images using a custom noise-adding function.

### Step 4: Define Autoencoder Architecture
* **Encoder**:
Convolutional layers (Conv2D) with ReLU activations and MaxPooling
* **Decoder**:
  Transposed convolutional layers (ConvTranspose2D) with ReLU and Sigmoid activations to reconstruct the image

### Step 5: Prepare Training
* Initialize the autoencoder model
* Define Mean Squared Error (MSE) as the loss function
* Choose Adam optimizer for training

### Step 6: Model Training
Train the autoencoder using the noisy images as input and the original clean images as the target. Track the loss over epochs to monitor learning.

### Step 7: Evaluate and Visualize
* Compare the original, noisy, and denoised images
* Visualize results to assess the model’s performance in removing noise

## PROGRAM
### Name:POOJA PRIYA B
### Register Number:212224230196
```
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14 -> 7
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
summary(model, input_size=(1, 28, 28))

def train(model, loader, criterion, optimizer, epochs=5):

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in loader:

            images = images.to(device)

            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)

            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: POOJA PRIYA B")
    print("Register Number: 212224230196")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

## OUTPUT

### Model Summary

<img width="769" height="484" alt="image" src="https://github.com/user-attachments/assets/374c6ed8-e988-4a8c-af9e-bb1900a3d724" />

### Original vs Noisy Vs Reconstructed Image

<img width="1726" height="633" alt="image" src="https://github.com/user-attachments/assets/2ae030a3-91e7-4471-a840-b1ec2f2f8ebf" />




## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
