import numpy as np
import matplotlib.pyplot as plt
import random as rd
import argparse

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None, xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def parseInput(fileName):
	f = open(fileName, 'r')
	lines = f.readlines()
	lines = [x.strip().split(" ") for x in lines]
	X = np.matrix(lines)[:,:-1].astype(float)
	y = np.matrix(lines)[:,-1].astype(float)
	return(X, y)

def prependOnesToMatrix(X):
	ones = np.ones(X.shape[0]).reshape((len(X), 1))
	newX = np.concatenate((ones, X), 1)
	return newX

def evaluateWeights(X, y, regularization = 0):
	w = np.linalg.inv(np.matmul(X.T, X) + np.identity(X.shape[1]) * regularization)
	w = np.matmul(np.matmul(w,X.T), y)
	return w

def generateTestPoints(numberOfPoints):
        Z = (np.random.rand(numberOfPoints, 2) - .5) * 10
        return Z

def Part1():
    X, y = parseInput("dataLinReg2D.txt")
    pX = prependOnesToMatrix(X)
    weights = evaluateWeights(pX, y)
    

    Z = generateTestPoints(10)
    pZ = prependOnesToMatrix(Z)
    predictions = np.matmul(pZ, weights)
    pos = []
    neg = []
    #TODO: fix the hack :)
    for i in range (len(predictions)):
    	if predictions[i][0] < 0:
    		pos.append(Z[i])
    	else:
    		neg.append(Z[i])
    pos = np.matrix(pos)
    neg = np.matrix(neg)
    xmin = min(Z[:,0])
    xmax = max(Z[:,0])
    ymin = min(Z[:,1])
    ymax = max(Z[:,1])
    ax = tidyPlot(xmin - 1, xmax + 1, ymin-1, ymax+1, xlabel = 'x1', ylabel = 'x2', title = 'Predictors')

    ax.plot(pos[:,0], pos[:,1], 'rs', neg[:,0], neg[:,1], 'bo')

def splitData(X, y, folds = 5):
    # print("X = {0}, y = {1}".format(X.shape, y.shape))
    valLength = int(len(X) / folds)
    count = 0
    valMask = np.zeros(X.shape[0], dtype=bool)
    trainMask = np.ones(X.shape[0], dtype=bool)

    while count < valLength:
        randIndex = rd.randint(0, len(X) - 1)
        if not valMask[randIndex]:
            valMask[randIndex] = True
            trainMask[randIndex] = False #TODO: fix the hack - negating the boolean mask yields a problem
            count += 1

    valX = X[valMask,:]
    valY = y[valMask]
    trainX = X[trainMask,:]
    trainY = y[trainMask]
    return trainX, trainY, valX, valY

def evaluateLoss(trainX, trainY, valX, valY, regularization = 0):
    trainX = prependOnesToMatrix(trainX)
    weights = evaluateWeights(trainX, trainY, regularization)
    predY = np.matmul(prependOnesToMatrix(valX), weights)
    loss = np.linalg.norm(predY - valY)
    print("Lambda = {0} | Loss = {1}".format(regularization, loss))
    return loss

def Part2(withNoise = False):
    X, y = parseInput("dataLinReg2D.txt")
    if withNoise:
        y = y + rd.gauss(0, 10)
    # regs = [0, .025, .125, .25, .5, 1, 2]
    regs = sorted(rd.sample(range(1, 10), 4))
    regs.insert(0, 0)
    optimalRegs = []
    for i in range (10):
        print("\n### Iteration {0} ###".format(i))
        trainX, trainY, valX, valY = splitData(X, y)
        # Checking the shape
        # print("trainX = {0}, trainY = {1}, valX = {2}, valY = {3}".format(trainX.shape, trainY.shape, valX.shape, valY.shape))
        optimalReg = -1
        minLoss = 99
        for reg in regs:
            loss = evaluateLoss(trainX, trainY, valX, valY, reg)
            if minLoss > loss:
                minLoss = loss
                optimalReg = reg
        print(optimalReg)
        optimalRegs.append(optimalReg)
    binsEdge = [x for x in range (10)]
    plt.hist(optimalRegs, bins = binsEdge)
    # plt.plot(regs, losses, "rs")

#--------------------------# 
# USE THIS TO RUN THE CODE #
#--------------------------#
Part1()
Part2()
Part2(True)
plt.show()
#--------------------------# 
# USE THIS TO RUN THE CODE #
#--------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Extract PBM pattern.')
    parser.add_argument('-p', dest = 'part',help = 'Part number')
    parser.add_argument('-n', dest = 'noise',help = 'Adding noise to validation val', const = True, default = False)
    args = parser.parse_args()
    
    part = args.part
    noise = args.noise

    if part == 1:
        Part1()
    else:
        Part2(noise)

    plt.show()

		

