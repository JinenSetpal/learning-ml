import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

mnist = np.load("mnist_scaled.npz")
X_train, y_train, X_test, y_test = [mnist[i] for i in mnist.files]
nn = pickle.load(open(os.path.join(os.curdir, "model.pkl"), 'rb'))
y_test_pred = nn.predict(X_test)

miscl_img = X_test[y_test != y_test_pred][:25]
correct_label = y_test[y_test != y_test_pred][:25]
pred_label = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5,
                       ncols=5,
                       sharex=True,
                       sharey=True)
ax = ax.flatten()

for i in range(len(miscl_img)):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('t: %d, p: %d' % (correct_label[i], pred_label[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
