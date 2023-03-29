# Import dependencies
from __cuda__ import _isCuda
import torch
import os
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torchvision.transforms as transforms

class Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#! Set device.

device = torch.device('cuda' if _isCuda else 'cpu')

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 4

#* Download data.
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#! Initalize Network
model = Network(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


#* train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #! Get data to cuda if possible.
        data = data.to(device=device)
        targets = targets.to(device=device)

        #! Get to correct shape
        data = data.reshape(data.shape[0], -1)

        #! Forward
        scores = model(data)
        loss = criterion(scores, targets)

        #! backward

        optimizer.zero_grad()
        loss.backward()

        #* gradient descent or adam step
        optimizer.step()


#! Check the accuarcy on the training & test to see how good the model is.
def check_accuracy(loader,model):

    if loader.dataset.train:
        print("Checking accuracy on traning data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuarcy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)