import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as ff
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

class ConvNet(nn.Module):
	def __init__(self, output_dim):
		super(ConvNet, self).__init__()

		self.l1 = nn.Conv2d(1, 20, 5)
		self.l2 = nn.MaxPool2d(2, 2)
		self.l3 = nn.Conv2d(20, 40, 5)
		self.l4 = nn.MaxPool2d(2, 2)
		self.l5 = nn.Linear(40 * 4 * 4, 100)
		self.dropout = nn.Dropout2d(p=0.5)
		self.l6 = nn.Linear(100, 10)

	def forward(self, x):
		out = self.l1(x)
		# print("l1", out.shape)
		out = ff.relu(self.l2(out))
		# print("l2", out.shape)
		out = self.l3(out)
		# print("l3", out.shape)
		out = ff.relu(self.l4(out))
		out = out.view((-1, 40 * 4 * 4))
		# print("l4", out.shape)
		out = self.dropout(self.l5(out))
		out = self.l6(out)
		# print(out.shape)


		return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
    	correct = 0
    	total = 0
    	for images, labels in test_loader:
    		images = images.to(device)
    		labels = labels.to(device)
    		outputs = model(images)
    		_, predicted = torch.max(outputs.data, 1)
    		total += labels.size(0)
    		correct += (predicted == labels).sum().item()
    	print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')