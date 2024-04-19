import numpy as np
from classes import Dense, NeuralNetwork
import matplotlib.pyplot as plt


X = []
Y = []

for i in range(10000):
    a = np.random.randint(0, 10000)
    X.append(a)
    Y.append(5*a + 22)

# plt.scatter(X, Y)
# plt.show()

EPOCHS = 10000
lr = 0.1
model = NeuralNetwork(1, 1)
losses = []
ws = []

for i in range(EPOCHS):

    x = np.array([X[i]]) / 10000
    true = Y[i]
    pred = model(x)[0, 0]

    print(true)
    print(pred)

    loss = (true - pred)**2

    step_size_b1 = -2 * (true - pred) * lr
    step_size_x1 = -2 * (true - pred) * x * lr

    print(f"Loss: {loss}")

    model.layer1.weights[0, 0] = model.layer1.weights[0, 0] - step_size_x1[0]
    model.layer1.bias[0, 0] = model.layer1.bias[0, 0] - step_size_b1

    print(f"Updated Weight: {model.layer1.weights[0, 0]}")
    print(f"Updated Bias: {model.layer1.bias[0, 0]}")

    ws.append(model.layer1.weights[0, 0])
    losses.append(loss)


plt.scatter(ws, losses)
plt.xlabel("Weight Value")
plt.ylabel("Loss")
plt.title("x1 vs Loss")
plt.show()

print(model(np.array([5 / 10000])))

# pred = w1*X + b1

# MSE = (pred - true)^2

# dMSE/dw1 = dMSE/dpred dpred/dw1
# dMSE/db1 = dMSE/dpred dpred/db1

# dMSE/DPRED = 2 * 1 * (true - pred)

# dpred/db1 = 1
