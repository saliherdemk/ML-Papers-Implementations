from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

from conv2d import Conv2d
from pooling import Pooling
from dense import Dense

from ops import Relu, Flatten, CrossEntropyLoss

from model import Model

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist["data"]
y = mnist["target"].astype(np.int32)

X = X / 255.0
X = X.reshape(-1, 28, 28)

X_train, X_test = X[:-10], X[-10:]
y_train, y_test = y[:-10], y[-10:]


model = Model(
    [
        Conv2d(kernel_size=3),  # 28x28 -> 26x26
        Relu(),
        Pooling(pool_type="max"),  # 26x26 -> 24x24
        Conv2d(kernel_size=3),  # 24x24 -> 22x22
        Relu(),
        Pooling(pool_type="max"),  # 22x22 -> 20x20
        Flatten(),
        Dense(400, 128),  # 20x20 = 400
        Relu(),
        Dense(128, 10),
    ]
)


def train(model, trainX, trainY, step=50_000):
    lr = 1e-3

    for i in range(step):
        index = i % trainX.shape[0]
        x = trainX[index]
        y = trainY[index]

        pred = model.forward(x)
        loss_fn = CrossEntropyLoss()

        loss = loss_fn.forward(pred, y)
        if i % 10_000 == 0:
            print(loss)

        grad = loss_fn.backward()
        model.backward(grad)

        model.update(lr)


def test(model, testX):
    num_images = testX.shape[0]
    _fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    for i in range(num_images):
        image = testX[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"{model.predict(image)}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train(model, X_train, y_train, step=100_000)
    test(model, X_test)
