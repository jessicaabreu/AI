import sys
import numpy as np
import csv

def readFile(path_to_file):
    with open(path_to_file, 'rU') as f:
        reader = csv.reader(f)
        data = map(tuple, reader)
    return data

def plotData(data, coefs):
    '''
    Receives data, which is a list of tuples. Each tuple represents a 2D point. 
    Each tuple is of the form (x, y, label). The label could be -1 or 1.

    The function plots the points, using different colors for different labels.
    The function also plots the line that divides the points according to
    coefficients [w0, w1, w2] in coefs (w1 x + w2 y + w0 = 0)
    '''
    import matplotlib.pyplot as plt
    x_points = np.array([x[0] for x in data[1:]])
    x_points = x_points.astype(float)
    y_points = np.array([y[1] for y in data[1:]])
    y_points = y_points.astype(float)
    color = np.array([c[2] for c in data[1:]])
    color = color.astype(float)
    plt.scatter(x_points, y_points, c= color)
    x_line = np.linspace(np.min(x_points), np.max(x_points), 100)
    y_line = - x_line *(coefs[1]/ coefs[2]) - coefs[0]/coefs[2]
    plt.plot(x_line, y_line)
    plt.show()

def perceptron(data, output_path):
    '''
    Function receives data and writes weights to a csv file until they converge.
    Wrights are: [w0, w1, w2] = > w0 + w1 x + w2 y = 0
    In the output file, the order is: [w1, w2, w0]
    '''
    import random
    # Starting the weights of the perceptron
    output = open(output_path, 'w+')
    writer = csv.writer(output)
    w0 = random.uniform(-0.2, 0.2)
    w1 = random.uniform(-0.2, 0.2)
    w2 = random.uniform(-0.2, 0.2)
    weights = np.array([w0, w1, w2])
    array_data = np.array(data).astype(float)
    change = 1
    while change > 0:
        copy_weights = np.array(weights)
        for i in range(array_data.shape[0]):
            f = np.sign(np.sum(weights[1:] * array_data[i, 0:2]) + weights[0])
            if array_data[i, 2] * f <= 0:
                copy_weights[1:] = weights[1:] + array_data[i, 0:2] * array_data[i, 2]
                copy_weights[0] = weights[0] + array_data[i, 2]
        writer.writerow([copy_weights[1], copy_weights[2], copy_weights[0]])
        change = np.linalg.norm(weights - copy_weights)
        weights = np.array(copy_weights)
    output.close()
    return weights

file_path = sys.argv[1]
output_path = sys.argv[2]
data = readFile(file_path)
weights = perceptron(data, output_path)
#plotData(data, weights)
