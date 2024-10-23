import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import time

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.convh1 = nn.Conv2d(1, 4, kernel_size=(5, 1), padding='same', bias = False)
        self.convv1 = nn.Conv2d(4, 8, kernel_size=(1, 5), padding='same', bias = False)
        self.convh2 = nn.Conv2d(8, 16, kernel_size=(3, 1), padding='same', bias = False)
        self.convv2 = nn.Conv2d(16, 16, kernel_size=(1, 3), padding='same', bias = False)
        self.dense1 = nn.Linear(28 * 28 * 16, 248)
        self.dense2 = nn.Linear(248, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convh1(x)) 
        x = self.relu(self.convv1(x)) 
        x = self.relu(self.convh2(x)) 
        x = self.relu(self.convv2(x)) 
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.dense1(x)) 
        x = self.dense2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

# Evaluation function
def evaluate():
    model.eval()
    correct = 0
    total = 0

    start_time = time.time()  # Start the timer

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Evaluation Time: {elapsed_time:.2f} seconds')

# Train the model
if __name__ == "__main__":
    train(epochs=10)
    evaluate()
    torch.save(model.state_dict(), 'model_weights.pth')