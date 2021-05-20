import math
import sys

import pandas as pd
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
trainingFileName = "F:/College/4th year/second term/Computer Vision/Assignments/ass 1/MNIST/mnist_test.csv"
testFileName = "F:/College/4th year/second term/Computer Vision/Assignments/ass 1/MNIST/mnist_test.csv"
featureVectorDataFrame = pd.DataFrame()


# read file
def readFile(name):
    data_list = pd.DataFrame()
    data_list = pd.read_csv(name)
    data_list = data_list.drop(data_list.columns[0], axis=1)  # ignore first column and row
    return data_list


# format the row into 32x32 image
def getImage(data_list, imageIndex):
    # reading 28x28 image
    imageBefore = np.array(data_list.values[imageIndex], dtype=float).reshape(28, 28)
    # adding 4 more columns right and 4 more rows down to make it 32x32
    zeros = np.zeros(28)
    imageAfter = np.vstack([imageBefore, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    imageAfter = np.vstack([imageAfter, zeros])
    zeros2 = np.zeros(32)
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    imageAfter = np.column_stack((imageAfter, zeros2))
    return imageAfter


# calculate magnitude and theta
def getMagnitudeAndTheta(image):
    magnitude = np.zeros((32, 32), dtype=float)
    theta = np.zeros((32, 32), dtype=float)
    for i in range(32):
        for j in range(32):
            if i == 0:
                y = image[i + 1][j]
            elif i == 31:
                y = image[i - 1][j]
            else:
                y = abs(image[i - 1][j] - image[i + 1][j])
            if j == 0:
                x = image[i][j + 1]
            elif j == 31:
                x = image[i][j - 1]
            else:
                x = abs(image[i][j - 1] - image[i][j + 1])
            magnitude[i][j] = math.sqrt(x ** 2 + y ** 2)
            if y == 0:
                theta[i][j] = 0
            else:
                theta[i][j] = math.degrees(math.atan(x / y)) % 180
    return magnitude, theta


# extracting feature vector for each cell
def extractCellsFeatureVector(magnitude, theta):
    cellsFeatureVector = np.zeros((4, 4, 1, 9), float)
    for k in range(0, 32, 8):  # image rows
        for m in range(0, 32, 8):  # image columns
            featureVector8x8 = np.zeros((1, 9))
            for i in range(k, k + 8):  # cell rows
                for j in range(m, m + 8):  # cell columns
                    angle = theta[i][j]
                    if angle < 10:  # handling special cases
                        featureVector8x8[0][0] += magnitude[i][j]
                    elif angle > 170:  # handling special cases
                        featureVector8x8[0][8] += magnitude[i][j]
                    else:
                        # get the first middle value where angle falls in
                        startMid = (angle // 10) * 10 if (angle // 10) % 20 != 0 else (angle // 10) * 10 - 10
                        # calculate both fractions
                        firstFraction = abs(angle - startMid) / 20
                        secondFraction = 1 - firstFraction
                        # get the amount of magnitude that will be added to that pin
                        currentPinValue = max(firstFraction, secondFraction) * magnitude[i][j]
                        # adding the value to it and the rest to the neighbor pin
                        featureVector8x8[0][int(angle // 20)] += currentPinValue
                        midValue = (angle // 20) * 20 + 10
                        if angle < midValue:  # it's in the first half of the pin
                            featureVector8x8[0][int(angle // 20 - 1)] += magnitude[i][j] - currentPinValue
                        else:
                            featureVector8x8[0][int(angle // 20 + 1)] += magnitude[i][j] - currentPinValue
            cellsFeatureVector[int(k / 8)][int(m / 8)] = featureVector8x8
    return cellsFeatureVector


def normalize(array, total):
    if total == 0:
        return array
    result = np.zeros((1, 9), float)
    for i in range(9):
        result[0][i] = array[0][i] / total
    return result


# concatenating feature vectors for each 4 cells to make a block feature vector
def getImageFeatureVector(cellsFeatureVector, imageIndex):
    blocksCounter = 0
    blocksFeatureVector = np.zeros(324, float)  # 9 blocks x 36 values
    for i in range(0, 3):  # imageFeatureVector rows
        for j in range(0, 3):  # imageFeatureVector columns
            # normalizing the cells before adding them to the block vector
            featureVector16x16 = np.zeros((36, 1), float)
            totalSum = sum(
                sum(cellsFeatureVector[i][j][0]) + sum(cellsFeatureVector[i][j + 1][0]) + cellsFeatureVector[i + 1][j][
                    0] + cellsFeatureVector[i + 1][j + 1][0])
            featureVector16x16[0:9, ] = np.transpose(normalize(cellsFeatureVector[i][j], totalSum))
            featureVector16x16[9:18, ] = np.transpose(normalize(cellsFeatureVector[i][j + 1], totalSum))
            featureVector16x16[18:27, ] = np.transpose(normalize(cellsFeatureVector[i + 1][j], totalSum))
            featureVector16x16[27:36, ] = np.transpose(normalize(cellsFeatureVector[i + 1][j + 1], totalSum))

            blocksFeatureVector[blocksCounter:blocksCounter + 36, ] = featureVector16x16[:][0]
            blocksCounter += 36

    featureVectorDataFrame.insert(imageIndex, imageIndex, blocksFeatureVector)


# writing results into an excel file
def writeDataFrame(DataFrame, fileName):
    DataFrame.to_csv(fileName, index=False)


# apply hog algorithm on a file and writing the results back into an excel file
def applyHog(fileName, outputFileName):
    # clear data frame before filling it
    featureVectorDataFrame.drop(featureVectorDataFrame.iloc[:, :], inplace=True, axis=1)
    data_list = readFile(fileName)
    for imageIndex in range(len(data_list.index)):
        image = getImage(data_list, imageIndex)
        magnitude, theta = getMagnitudeAndTheta(image)
        cellsFeatureVector = extractCellsFeatureVector(magnitude, theta)
        getImageFeatureVector(cellsFeatureVector, imageIndex)
    writeDataFrame(featureVectorDataFrame, outputFileName)


applyHog(trainingFileName, "training-output.csv")
