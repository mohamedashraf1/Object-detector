import math
import sys

import pandas as pd
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# read file
filename = "F:/College/4th year/second term/Computer Vision/Assignments/ass 1/MNIST/mnist_test.csv"
data_list = pd.DataFrame()
data_list = pd.read_csv(filename)
data_list = data_list.drop(data_list.columns[0], axis=1)  # ignore first column and row


# reading 28x28 image
imageBefore = np.array(data_list.values[0], dtype=float).reshape(28, 28)
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

# calculate magnitude and theta
magnitude = np.zeros((32, 32), dtype=float)
theta = np.zeros((32, 32), dtype=float)
for i in range(32):
    for j in range(32):
        if i == 0:
            y = imageAfter[i + 1][j]
        elif i == 31:
            y = imageAfter[i - 1][j]
        else:
            y = abs(imageAfter[i - 1][j] - imageAfter[i + 1][j])
        if j == 0:
            x = imageAfter[i][j + 1]
        elif j == 31:
            x = imageAfter[i][j - 1]
        else:
            x = abs(imageAfter[i][j - 1] - imageAfter[i][j + 1])
        magnitude[i][j] = math.sqrt(x ** 2 + y ** 2)
        if y == 0:
            theta[i][j] = 0
        else:
            theta[i][j] = math.degrees(math.atan(x / y)) % 180

imageFeatureVector = np.zeros((0, 0), float)
for k in range(0, 32, 8):  # image rows
    for m in range(0, 32, 8):  # image columns
        featureVector8x8 = np.zeros(9)
        for i in range(k, k + 8):  # cell rows
            for j in range(m, m + 8):  # cell columns
                angle = theta[i][j]
                if angle < 10:  # handling special cases
                    featureVector8x8[0] += magnitude[i][j]
                elif angle > 170:  # handling special cases
                    featureVector8x8[8] += magnitude[i][j]
                else:
                    # get the first middle value where angle falls in
                    startMid = (angle//10)*10 if (angle//10) % 20 != 0 else (angle//10)*10 - 10
                    # calculate both fractions
                    firstFraction = abs(angle - startMid) / 20
                    secondFraction = 1 - firstFraction
                    # get the amount of magnitude that will be added to that pin
                    currentPinValue = max(firstFraction, secondFraction) * magnitude[i][j]
                    # adding the value to it and the rest to the neighbor pin
                    featureVector8x8[int(angle // 20)] += currentPinValue
                    midValue = (angle//20)*20 + 10
                    if angle < midValue:  # it's in the first half of the pin
                        featureVector8x8[int(angle // 20 - 1)] += magnitude[i][j] - currentPinValue
                    else:
                        featureVector8x8[int(angle // 20 + 1)] += magnitude[i][j] - currentPinValue
        imageFeatureVector = np.append(imageFeatureVector, featureVector8x8)

