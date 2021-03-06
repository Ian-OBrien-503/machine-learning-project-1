import numpy
import random
import xlsxwriter


# for writing to excel sheet to look at clean form of data
workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()


# global variables
wow = 0
weights = []
outputs = []
targets = []
learning_rate = 0.1
epoch = 0


# one hot encoding for target matrix
target_mat = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
]


# initialize weight matrix included weight for bias
weights = [[random.uniform(-.05, .05) for j in range(785)] for i in range(10)]


# load data into matrix and pre-process the data, also adding bias
matrix = numpy.loadtxt(open('train.csv', 'rb'), delimiter=",", skiprows=1)
for each in matrix[:, 0]:
    targets.append(each)
matrix = matrix[:, 1:785]
matrix = (1/255) * matrix
n, m = matrix.shape
X0 = numpy.ones((n,1))
matrix = numpy.hstack((X0, matrix))
print(targets)


# computer the dot product for all inputs and all weights for any given example from training set
def train():
    correct = 0
    for x in range(59999):
        global wow
        wow = x
        x0 = numpy.dot(matrix[x], weights[0])
        x1 = numpy.dot(matrix[x], weights[1])
        x2 = numpy.dot(matrix[x], weights[2])
        x3 = numpy.dot(matrix[x], weights[3])
        x4 = numpy.dot(matrix[x], weights[4])
        x5 = numpy.dot(matrix[x], weights[5])
        x6 = numpy.dot(matrix[x], weights[6])
        x7 = numpy.dot(matrix[x], weights[7])
        x8 = numpy.dot(matrix[x], weights[8])
        x9 = numpy.dot(matrix[x], weights[9])
        global outputs
        outputs = numpy.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])
        predict = max(outputs)
        count = 0
        for test in outputs:
            if predict == test:
                break
            count = count + 1
        for x in range(10):
            if outputs[x] > 0:
                outputs[x] = 1
            else:
                outputs[x] = 0
        if count == targets[int(x)]:
            correct = correct + 1
            # print(correct)
        else:
            update_weights(count, wow)
    return calc_accuracy(correct, epoch)


def data_test():
    correct = 0
    for x in range(9999):
        global wow
        wow = x
        x0 = numpy.dot(matrix[x], weights[0])
        x1 = numpy.dot(matrix[x], weights[1])
        x2 = numpy.dot(matrix[x], weights[2])
        x3 = numpy.dot(matrix[x], weights[3])
        x4 = numpy.dot(matrix[x], weights[4])
        x5 = numpy.dot(matrix[x], weights[5])
        x6 = numpy.dot(matrix[x], weights[6])
        x7 = numpy.dot(matrix[x], weights[7])
        x8 = numpy.dot(matrix[x], weights[8])
        x9 = numpy.dot(matrix[x], weights[9])
        outputs = numpy.array([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9])
        predict = max(outputs)
        count = 0
        for test in outputs:
            if predict == test:
                break
            count = count + 1
        for x in range(10):
            if outputs[x] > 0:
                outputs[x] = 1
            else:
                outputs[x] = 0
        if count == targets[int(x)]:
            correct = correct + 1
            # print(correct)

        else:
            update_weights(count, wow)
    return calc_accuracy_test(correct, epoch)


# pass in the index of which traning example we are on through z and update weights of the whole weight matrix
def update_weights(count, z):
    target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target[count] = 1
    for p in range(10):
        for y in range(785):
            weights[y][p] = weights[y][p] + learning_rate*(p-outputs[p])*matrix[z][y]


# calculate accuracy for training set
def calc_accuracy(correct, x):
    print("accuracy for train set (correct/59999) at epoch", x, "=", correct/59999)
    global epoch
    epoch = x
    epoch = epoch + 1
    return (correct/59999)


# calculate accuracy for test set
def calc_accuracy_test(correct, x):
    print("accuracy for test set (correct/10000) at epoch", x, "=", correct/9999)
    return(correct/9999)


# MAIN PROG, start by training the model
for x in range(70):
    test = train()

# load data into matrix and pre-process the data, also adding bias
matrix = numpy.loadtxt(open('test.csv', 'rb'), delimiter=",", skiprows=1)
for each in matrix[:, 0]:
    targets.append(each)
matrix = matrix[:, 1:785]
matrix = (1/255) * matrix
n, m = matrix.shape
X0 = numpy.ones((n,1))
matrix = numpy.hstack((X0, matrix))

# call to run test data after training the model
data_test()

# output to excel sheet to check data
# row = 0
# for col, data in enumerate(matrix):
#    worksheet.write_column(row, col, data)
# workbook.close()
