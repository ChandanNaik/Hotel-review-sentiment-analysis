
import numpy as np
import csv
import random

# Initialization
trainDataFile = "processedTrainHotelReviews.csv"
testDataFile = "processedTestHotelReviews.csv"

learningRates = [1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1e-1]
epochs = [5, 10, 15, 20, 30]
bias = 1

# Reads the feature vector csv, adds a bias term and returns an np array
def processData(inputFile):
    with open(inputFile, 'r') as f:
        featureList = list(csv.reader(f, delimiter=','))
    for row in featureList:
        del row[0]
        row.insert(len(row)-1, bias)
    cleanedFeatureVectors = np.asarray(featureList, dtype=np.float64)

    return cleanedFeatureVectors

# Sigmoid function
def sigmoid(score):
    return (1 / (1 + np.exp(-score)))

# Stochastic Gradient Descent Algorithm implementation
def sgdUpdate(featureVectors, weights, learningRate, epoch):
    i = 0
    while(i<epoch):
        np.random.shuffle(featureVectors)
        for i in range(0,2000):
            randomIndex = random.randint(0, len(featureVectors)-1)
            randomFeature = featureVectors[randomIndex]
            trueLabel = float(randomFeature[-1])
            randomFeature = randomFeature[:len(featureVectors[0])-1]
            rawTrainScore = np.dot(weights,randomFeature)
            trainScore = sigmoid(rawTrainScore)
#             print("************Start************")
#             print("Feature: ",randomFeature)
#             print("Actual Answer: ", trueLabel)
#             print("Obtained Score: ", score)
#             print("weight: ", weights)
#             print("************End************")
            predictionDifference = (trainScore - trueLabel)
            gradient = predictionDifference * randomFeature
            weights = weights - (learningRate * gradient)
        i+=1
    return weights

# Calculates Loss and Accuracy with updated weights
def progress(featureData, updatedWeight):
    logprob = 0.0
    progressCorrectCount = 0
    totalData,y = featureData.shape
    for i in range(totalData):
        iFeature = featureData[i]
        progressTrueLabel = float(iFeature[-1])
        iFeature = iFeature[:len(featureData[0])-1]
        rawProgressTestScore = np.dot(updatedWeight, iFeature)
        progressProbability = sigmoid(rawProgressTestScore)

        if progressTrueLabel == 1:
            logprob += np.log(progressProbability)
        else:
            logprob += np.log(1.0 - progressProbability)

        if abs(progressTrueLabel - progressProbability) < 0.5:
            progressCorrectCount += 1

    return logprob, float(progressCorrectCount) / float(totalData)

# This function classifies test data using the trained model/weight
def classifyTestData(trainedWeights, testData, testID):
    correctCount = 0.0
    x,y = testData.shape
    finalScoreDictionary = {}
    for i in range(x):
        accuracy = 0.0
        testDataFeature = testData[i]
        rawTestScore = np.dot(trainedWeights, testDataFeature)
        testScore = sigmoid(rawTestScore)
        if testScore > 0.5:
            finalScoreDictionary[testID[i]] = "POS"
        else:
            finalScoreDictionary[testID[i]] = "NEG"
#         print("Test Score for ",testID[i]," is:- ", testScore)
    return finalScoreDictionary

# Main
if __name__ == '__main__':
    # Set initial weight to Zero
    initialWeight = np.zeros(shape=(1,7))

    # Read Feature Vectors from csv
    featureVectors = processData(trainDataFile)

    # Shuffling the feature vectors
    np.random.shuffle(featureVectors)

    # Splitting the training data to 80 training set and 20 test set
    trainingID = np.random.randint(featureVectors.shape[0], size=80)
    testID = np.random.randint(featureVectors.shape[0], size=20)
    trainingData, testData = featureVectors[trainingID,:], featureVectors[testID,:]

    # |-------------------------Start Training-------------------------|
    for l in learningRates:
        for e in epochs:
            updatedWeight = sgdUpdate(trainingData, initialWeight, l, e)
            for epoch in range(e):
                trainLoss, trainAccuracy = progress(trainingData, updatedWeight)
                validLoss, validAccuracy = progress(testData, updatedWeight)
#                 print("***********************")
#                 print("Train Loss : " + str(np.round(trainLoss, 2)))
#                 print("Train Accuracy : " + str(np.round(trainAccuracy * 100, 3)) + "%")
#                 print("Valid Loss : " + str(np.round(validLoss, 2)))
#                 print("Valid Accuracy : " + str(np.round(validAccuracy * 100, 3)) + "%")
    # |-------------------------End Training-------------------------|


    # |-------------------------Start Testing-------------------------|
    print("Positive or Negative prediction for test data")
    testID = np.genfromtxt(testDataFile, dtype=str, delimiter=',', usecols=(0))
    testFeatureVectors = processData(testDataFile)
    finalResult = classifyTestData(updatedWeight, testFeatureVectors, testID)
    print(finalResult)
    # |-------------------------End Testing-------------------------|

    # |-------------------------Start Writing Dictionary to CSV-------------------------|
    try:
        with open('modelPrediction.txt', 'w') as f:
            writeString = ""
            for key, value in finalResult.items():
                writeString += str(key) + '\t'+ (str(value)) +'\n'
            f.write(writeString)
        print("Pridiction result written to a txt file(modelPrediction.txt)")
    except:
        print("Failed to write prediction result to a txt file")

    # |-------------------------End Writing Dictionary to CSV-------------------------|
