import numpy as np
import  matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import accuracy_score


(train_X, train_y), (test_X, test_y) = mnist.load_data()
X_train = train_X.reshape(60000,28*28)/255
X_test = test_X.reshape(10000,28*28)/255


def softmax(z):
  return np.exp(z) / np.sum(np.exp(z),axis = -1, keepdims = True)

def ohe(y):
  y = y.squeeze()
  examples, features = y.shape[0], len(np.unique(y))
  zeros_matrix = np.zeros((examples, features))
  for i, (row, digit) in enumerate(zip(zeros_matrix, y)):
    zeros_matrix[i][digit] = 1

  return zeros_matrix

def Softmax_regresion(x, y, x_val, y_val, epochs = 100, step = 0.9):
  x = x.copy()
  x_val = x_val.copy()
  y_enc = ohe(y)
  y_enc1 = ohe(y_val)
  n,m = x.shape
  weight = np.zeros((m, y_enc.shape[1]))
  bias = np.zeros(y_enc.shape[1])

  for i in range(epochs):
    y_pred_linear = np.dot(x, weight) + bias
    # print('y_pred_linear',y_pred_linear)

    y_pred_softmax = softmax(y_pred_linear)
    dw = 1 / n * np.dot(x.T, (y_pred_softmax - y_enc))
    db = 1 / n * np.sum(y_pred_softmax - y_enc)
    weight = weight - step * dw
    bias = bias - step * db
    print('progress:', i/epochs*100)


  y_pred = softmax(np.dot(x_val, weight) + bias)
  # print(f"Итерация: {i}")
  a = np.argmax(y_pred, axis = 1)
  print(f"Logloss {-np.sum(y_enc1 * np.log(y_pred)) / n}")
  print(f"Accuracy {accuracy_score(a,y_val)}")
  print("--------------------------------------------------------")
  return np.argmax(y_pred, axis = 1)


B = Softmax_regresion(X_train, train_y, X_test, test_y)
print(B)
fig = plt.figure(figsize=(15, 10))

for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1)
    ax.imshow(test_X[i], cmap=plt.get_cmap('gray'))
    ax.set_title(f'y: {test_y[i]} y_val: {B[i]}')
    plt.axis('off')
plt.show()



