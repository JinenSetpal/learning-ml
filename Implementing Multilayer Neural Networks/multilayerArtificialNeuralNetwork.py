from neuralnet import NeuralNetMLP
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

mnist = np.load('mnist_scaled.npz')
X_train, y_train, X_test, y_test = [mnist[i] for i in mnist.files]

if os.path.exists("model.pkl"):
    nn = pickle.load(open(os.path.join(os.path.dirname(__file__), 'model.pkl'), 'rb'))
else:
    nn = NeuralNetMLP(n_hidden=100,
                      l2=0.01,
                      epochs=200,
                      eta=0.0005,
                      minibatch_size=100,
                      shuffle=True,
                      seed=1)
    nn.fit(X_train=X_train[:55000],
           y_train=y_train[:55000],
           X_valid=X_train[55000:],
           y_valid=y_train[55000:])

    pickle.dump(nn,
                open(os.path.join('model.pkl'), 'wb'),
                protocol=4)

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.plot(range(nn.epochs), nn.eval_['train_acc'],
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
         label='validation', linestyle='--')
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(float) / X_test.shape[0])
print('Test Accuracy: %.2f%%' % (acc * 100))
