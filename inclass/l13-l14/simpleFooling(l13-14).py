import torchvision.models as models
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import os
from torch.autograd import Variable
import cv2
import copy

def resizeAndSquare(image, size = 224):
	transformPipeline = transforms.Compose([transforms.Resize(size),
												transforms.CenterCrop(size)])
	return transformPipeline(image)

def convertToTensor(image):
	transformPipeline = transforms.Compose([transforms.ToTensor(),
											transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                              	std=[0.229, 0.224, 0.225])])
	return transformPipeline(image)

def processImage(image, toTensor = True):
	image = resizeAndSquare(image)
	if toTensor:
		image = convertToTensor(image)
	return image

def convertTensorToImage(tensor):
	reverseMean = [-0.485, -0.456, -0.406]
	reverseStd = [1/0.229, 1/0.224, 1/0.225]
	outImg = copy.copy(tensor.data.numpy()[0])
	for c in range(3):
		outImg[c] /= reverseStd[c]
		outImg[c] -= reverseMean[c]
	outImg[outImg > 1] = 1
	outImg[outImg < 0] = 0
	outImg = np.round(outImg * 255)

	outImg = np.uint8(outImg).transpose(1, 2, 0)
	# Convert RBG to GBR
	outImg = outImg[..., ::-1]
	return outImg

def computeDifferences(a, b):
	diff = a - b
	diffImg = convertTensorToImage(diff)
	meanAbsoluteDiff = diffImg.mean()
	cv2.imwrite(currDir + "/generated/foolDiff.jpg", diffImg)
	return diffImg, diff, meanAbsoluteDiff

def fool(model, image, targetClass, saveIterations = False):
	stepSize = .01
	processImage(image, toTensor = False).save(currDir + "/generated/foolOriginal.jpg")

	imageTensor = processImage(image).unsqueeze(0)
	imageTensor = Variable(imageTensor, requires_grad=True)

	optimiser = torch.optim.SGD([imageTensor], lr = stepSize)
	model.eval()
	predClass = -1

	iterations = 0
	while not predClass == targetClass:
		iterations += 1
		prediction = model(imageTensor)
		predClass = prediction.data.numpy().argmax()
		classLoss = -prediction[0, targetClass]

		print("Iteration {0:02d}, current prediction = {1}, target class loss = {2:.3f}".format(iterations, predClass, classLoss))
		
		if saveIterations:
			cv2.imwrite(currDir + "/generated/fool{0:02d}.jpg".format(iterations), convertTensorToImage(imageTensor))

		model.zero_grad()
		classLoss.backward()
		optimiser.step()

	cv2.imwrite(currDir + "/generated/foolFinal.jpg", convertTensorToImage(imageTensor))
	return imageTensor

if __name__ == "__main__":
	currDir = os.path.dirname(os.path.realpath(__file__))
	originalImage = Image.open(currDir + "/mrshout2.jpg")
	model = models.resnet18()
	model.load_state_dict(torch.load(currDir + "/../models/resnet18-5c106cde.pth"))
	fooledTensor = fool(model, originalImage, 949)

	diffImg, diff, meanAbsoluteDiff = computeDifferences(processImage(originalImage), fooledTensor)
	print("meanAbsoluteDiff = {}".format(meanAbsoluteDiff))
	print("View /generated for foolOriginal.jpg | foolDiff.jpg } foolFinal.jpg")

