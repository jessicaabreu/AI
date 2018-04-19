import sys
import numpy as np
import csv

def readFile(path_to_file):
    with open(path_to_file, 'rU') as f:
        reader = csv.reader(f)
        data = map(tuple, reader)
    return data

def prepareData(data):
    array_data = np.array(data).astype(float)
    mean = np.mean(array_data, axis=0)
    std = np.std(array_data, axis=0)
    normalized = (array_data - mean) / std
    ones_to_add = np.ones(shape=(normalized.shape[0],
                                1))
    final_data = np.hstack((ones_to_add, normalized))
    # Only the features are normalized, not the labels
    final_data[:, -1] = array_data[:, -1]
    return final_data

def plotRegression(data, betas):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    array_data = np.array(data).astype(float)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(array_data[:, 0], array_data[:, 1], array_data[:, 2])
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Weight (kg)')
    ax.set_zlabel('Height (m)')
    xx, yy  = np.meshgrid(np.linspace(np.min(array_data[:, 0]),
                                      np.max(array_data[:, 0]), 100),
                          np.linspace(np.min(array_data[:, 1]),
                                     np.max(array_data[:, 1]), 100))
    zz = betas[0]  + betas[1] * xx + betas[2] * yy
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.5)
    plt.show()

def gradientDescent(data, a, numberInt, output_path, file_writer):
    '''
    data: [n x 4] => [1, feature1, feature2, y]
        data is already normalized

    a: learning rate
    '''
    beta = np.array([0, 0, 0]).T
    y = np.array(data[:, -1])
    x = np.array(data[:, 0:3])
    r = (1.0/(2*x.shape[0])) * np.dot((y - np.dot(x, beta)).T, y - np.dot(x, beta))

    count = 0
    epsilon = 0.0000000001
    change = 1

    while count < numberInt and change > epsilon:
        dr_db = -(1.0/x.shape[0]) * np.dot(x.T, y - np.dot(x, beta))
        new_beta = beta - a * dr_db
        beta = np.array(new_beta)
        new_r = (1.0/(2*x.shape[0])) * np.dot((y - np.dot(x, beta)).T, y -
                                              np.dot(x, beta))
        change = r - new_r
        count += 1
        r = new_r
    file_writer.writerow([a, numberInt, beta[0], beta[1], beta[2]])
    # plotRegression(data[:, 1:], beta)
    return beta

file_path = sys.argv[1]
output_path = sys.argv[2]
data = readFile(file_path)
prepared = prepareData(data)
output = open(output_path, 'w+')
writer = csv.writer(output)
for l in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
    beta = gradientDescent(prepared, l, 100, output_path, writer)
beta_final = gradientDescent(prepared, 1, 12, output_path, writer)
output.close()
