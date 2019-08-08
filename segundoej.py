from numpy import random
import matplotlib.pyplot as plt
random.seed(56)


class Perceptron:

    def __init__(self, weights=None, bias=None, number_of_weights=0):
        self.number_of_weights = number_of_weights
        if number_of_weights < 1:
            assert (type(self.weights) == type(list))
            self.weights = weights  # a list
            self.bias = bias
        else:
            self.weights = []
            for i in range(number_of_weights):
                self.weights.append(random.randint(-2, 2))
            self.bias = random.randint(-2, 2)

    def fit(self, input_list, desired_output=None):
        if len(self.weights) == len(input_list):
            output = self.bias
            for i in range(len(self.weights)):
                output = output + self.weights[i]*input_list[i]
            if output >= 0:
                answer = 1
            else:
                answer = 0
            if answer == desired_output:  # TP and TN
                if answer:
                    return "TP"
                else:
                    return "TN"
            else:
                # print("Wrong answer :(")
                # We have to readjust the Perceptron
                diff = desired_output - output
                lr = 0.001  # Class variable?
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] + (lr * input_list[j] * diff)
                self.bias = self.bias + (lr + diff)
                if answer:
                    return "FP"
                else:
                    return "FN"
        else:
            print("Error, input length not equal to weight length")


def is_above(xi, yi, a, b):
    if a * xi + b >= yi:
        return 0
    else:
        return 1


# f(x) = 2*x + 3

number_of_training = 100
TP = 0
FP = 0
TN = 0
FN = 0
trained = [5, 10, 20, 35, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
pres = []
for xtrained in trained:
    p = Perceptron(number_of_weights=2)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for k in range(xtrained):
        rango = 50
        x = random.randint(-rango, rango)
        y = random.randint(-rango, rango)
        answ = p.fit([x, y], desired_output=is_above(x, y, 2, 3))
        if answ == "TP":
            TP += 1
        elif answ == "TN":
            TN += 1
        elif answ == "FN":
            FN += 1
        else:
            TN += 1
    pres.append((TP+TN)/(FN+FP+TP+TN))
#Precision = TP/(TP + FP)
#print( TP, TN, FP, FN)
#print((TP+TN)/(FN+FP+TP+TN))
#print(pres)


plt.plot(trained, pres)
plt.axis([0, 500, 0.7, 1])
plt.show()
