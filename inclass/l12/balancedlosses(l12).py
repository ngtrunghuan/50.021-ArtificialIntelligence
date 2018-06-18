import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

currDir = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cpu')

class Dataset(torch.utils.data.Dataset):
	def __init__(self, fileName, isTraining = True, transform = None):
		file = open(fileName)
		self.isTraining = isTraining
		self.transform = transform
		self.data = []
		for line in tqdm(file.readlines()):
			self.data.append(list(map(float,line.split())))
		self.data = np.array(self.data).astype(float)

		self.countPositives = int(np.sum(self.data[:,-1]))
		self.countNegatives = int(len(self) - self.countPositives)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		return (item[:-1], item[-1])

	def plot(self):
		ones = []
		zeroes = []
		data = list(self.data)
		for item in data:
			if item[-1] == 0:
				zeroes.append(item)
			else:
				ones.append(item)
		zeroes = np.matrix(zeroes)
		ones = np.matrix(ones)
		plt.plot(ones[:,0], ones[:,1], 'rs', zeroes[:,0], zeroes[:,1], 'bo')
		plt.show()

	def applyBalanceSampling(self, alpha = 0.5):
		# _, counts = np.unique(self.data, return_counts = True, axis = 0)
		originalLength = len(self)
		if self.countPositives < self.countNegatives:
			toDuplicate = 1
			numDuplicates = int(self.countNegatives / self.countPositives - 1)
		else:
			toDuplicate = 0
			numDuplicates = int(self.countPositives / self.countNegatives - 1)
		data = []
		for i in tqdm(range(originalLength)):
			item = self.data[i][:-1] 
			label = self.data[i][-1]
			if label == toDuplicate:
				for j in range(numDuplicates):
					# append on numpy flattens the array, making weird errors. Hence here we use much simpler list appending
					data.append(list(self.data[i])) 
			data.append(list(self.data[i]))
		self.data = np.array(data).astype(float)
		self.countPositives = int(np.sum(self.data[:,-1]))
		self.countNegatives = int(len(self) - self.countPositives)

	def getClassCount(self, label):
		if label == 0:
			return self.countNegatives
		if label == 1:
			return self.countPositives
		return 0


### MODEL PARAMETERS ###
class SimpleModelWithRelu(torch.nn.Module):
	def __init__(self):
		super(SimpleModelWithRelu, self).__init__()
		self.fc = torch.nn.Linear(2, 1)

	def forward(self, x):
		return F.relu(self.fc(x))

class SimpleModel(torch.nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.fc = torch.nn.Linear(2, 1)

	def forward(self, x):
		return self.fc(x)

model = SimpleModel()
criterion = torch.nn.BCEWithLogitsLoss()
learningRate = 0.1
batchSize = 128
optimiser = torch.optim.SGD(model.parameters(), lr = 0.1)
numEpoch = 10

def run(withBalancing = False):
	trainDataset = Dataset(fileName = currDir + "/samplestr.txt",
								isTraining = True)
	testDataset = Dataset(fileName = currDir + "/sampleste.txt",
								isTraining = False)
	
	if withBalancing:
		print("\nRUN WITH BALANCING")
		trainDataset.applyBalanceSampling()
	else:
		print("\nRUN WITHOUT BALANCING")
	trainLoader = torch.utils.data.DataLoader(dataset = trainDataset,
											batch_size = batchSize,
											shuffle = False)
	testLoader = torch.utils.data.DataLoader(dataset = testDataset,
											batch_size = batchSize,
											shuffle = False)
	
	totalSteps = len(trainLoader)
	print("\nNumber of training samples =", len(trainDataset))
	for epoch in range(numEpoch):
		print("\nEpoch {}".format(epoch))
		model.train()
		for i, (samples, labels) in enumerate(trainLoader):
			# Forward pass
			outputs = model(samples.float())
			loss = criterion(outputs, labels.view(len(labels), 1).float())

			# Backward and optimize
			loss.backward()
			optimiser.step()
	        
			# if (i+1) % 100 == 0:
			# 	print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
			# 	       .format(epoch+1, numEpoch, i+1, totalSteps, loss.item()))
		model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
		with torch.no_grad():
			correct = 0
			correctOnes = 0
			correctZeroes = 0
			total = 0
			for samples, labels in testLoader:
				labels = labels.view((len(labels),1))
				outputs = model(samples.float())
				predicted = (outputs.data > 0).long()

				total += labels.size(0)

				correct += (predicted == labels.long()).sum().item()
				correctMask = predicted == labels.long()

				predictedOnes = predicted == torch.ones(predicted.size(), dtype = torch.long)
				correctOnes += (correctMask * predictedOnes).sum().item()

				predictedZeroes = predicted == torch.zeros(predicted.size(), dtype = torch.long)
				correctZeroes += (correctMask * predictedZeroes).sum().item()
			print('Test Accuracy of the model on the {} test images: {} %'.format(len(testDataset), 100 * correct / total))
			print("True Positives Rate = {}%".format(correctOnes / testDataset.getClassCount(1) * 100))
			print("True Negatives Rate = {}%".format(correctZeroes / testDataset.getClassCount(0) * 100))
			

def plotData(trainData = True):
	if trainData:
		trainDataset = Dataset(fileName = currDir + "/samplestr.txt",
									isTraining = True)
		trainDataset.plot()
	else:
		testDataset = Dataset(fileName = currDir + "/sampleste.txt",
									isTraining = False)
		testDataset.plot()

#--------------------------# 
# USE THIS TO RUN THE CODE #
#--------------------------#
run(withBalancing = False)
run(withBalancing = True)
plotData(trainData = False)
#--------------------------# 
# USE THIS TO RUN THE CODE #
#--------------------------#
