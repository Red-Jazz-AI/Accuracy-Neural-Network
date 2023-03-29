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

"""class Network(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
#! Up there is the simple neural network code.


# TODO: Create a simple CNN
class ConvoNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):        
        super(ConvoNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=8,
                               kernel_size=(3,3), 
                               stride=(1,1),
                                 padding=(1,1))
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2),
                                  stride=(2,2))
        
        self.conv2 = nn.Conv2d(in_channels=8,
                                out_channels=16,
                               kernel_size=(3,3), 
                               stride=(1,1),
                                 padding=(1,1))
        
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


model = ConvoNet()
x = torch.randn(64 ,1 ,28, 28)



        
#! Set device.

device = torch.device('cuda' if _isCuda else 'cpu')

#* Hyperparameters
in_channel = 1
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
model = ConvoNet().to(device)
#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)


#* train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #! Get data to cuda if possible.
        data = data.to(device=device)
        targets = targets.to(device=device)

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

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuarcy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)