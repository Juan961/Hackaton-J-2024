import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


classes = [
    "aloevera",
    "banana",
    "bilimbi",
    "cantaloupe",
    "cassava",
    "coconut",
    "corn",
    "cucumber",
    "curcuma",
    "eggplant",
    "galangal",
    "ginger",
    "guava",
    "kale",
    "longbeans",
    "mango",
    "melon",
    "orange",
    "paddy",
    "papaya",
    "peperchili",
    "pineapple",
    "pomelo",
    "shallot",
    "soybeans",
    "spinach",
    "sweetpotatoes",
    "tobacco",
    "waterapple",
    "watermelon",
]


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # Open image as PIL Image and convert to RGB
        label = int(self.img_labels.iloc[idx, 1])  # Ensure label is int
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=15),  # Randomly rotate images
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 64

trainset = CustomImageDataset("./images/train.csv", "./images", transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CustomImageDataset("./images/test.csv", "./images", transform=transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third block (final convolution layer)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self._initialize_weights()
        self.fc1 = nn.Linear(self.fc_input_size, 256)  # Reduced size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 30)  # Output classes

    def _initialize_weights(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            output = F.relu(self.bn1(self.conv1(dummy_input)))
            output = self.pool1(output)
            output = F.relu(self.bn2(self.conv2(output)))
            output = self.pool2(output)
            output = F.relu(self.bn3(self.conv3(output)))
            output = self.pool3(output)
            self.fc_input_size = output.numel()

    def forward(self, input):
        # First block
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.pool1(output)

        # Second block
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool2(output)

        # Third block
        output = F.relu(self.bn3(self.conv3(output)))
        output = self.pool3(output)

        # Flatten and fully connected layers
        output = output.view(-1, self.fc_input_size)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)

        return output

# Instantiate a neural network model 
model = Network()

# Define the loss function with Classification Cross-Entropy loss and an optimizer with SGD optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Function to save the model
def saveModel():
    path = "../backend/machine/models/cnn.pth"
    torch.save(model.state_dict(), path)

# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"Starting epoch {epoch + 1}")
        model.train()  # Ensure the model is in training mode
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # Move data to the execution device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print statistics every 100 mini-batches
            if (i + 1) % 100 == 0:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Step the scheduler
        scheduler.step()

        # Compute and print the accuracy for this epoch
        accuracy = testAccuracy()
        print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy:.2f}%")

        # Save the model if this is the best accuracy so far
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")

def testAccuracy():
    model.eval()  # Set model to evaluation mode
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the execution device
            images, labels = images.to(device), labels.to(device)

            # Predict
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Update accuracy
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # Compute overall accuracy
    accuracy = 100 * accuracy / total
    return accuracy


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    def imageshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

def testClassess():
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__":
    start = time.time()
    print('======================= Starting Training =======================')
    train(20)
    print('======================= Finished Training =======================')
    end = time.time()

    print('The training took', end-start, 'seconds')

    # Test which classes performed well
    print('======================= Starting Accuracy =======================')
    testAccuracy()
    print('======================= Finished Accuracy =======================')

    print('======================= Starting Test Batch =======================')
    model = Network()
    path = "../backend/machine/models/cnn.pth"
    model.load_state_dict(torch.load(path, weights_only=True))
    testBatch() # Test with batch of images
    print('======================= Finished Test Batch =======================')

    print('======================= Starting Test Classes =======================')
    testClassess() # Test the accuracy per label
    print('======================= Finished Test Classes =======================')
