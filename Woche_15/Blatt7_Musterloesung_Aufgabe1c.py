from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

KERNEL = 3
POOLING = 4
EPOCHS = 10

RETRAIN = False


def main():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_shape = train_images.shape
    train_images = train_images.reshape((train_shape[0], train_shape[1], train_shape[2], 1))
    test_shape = test_images.shape
    test_images = test_images.reshape((test_shape[0], test_shape[1], test_shape[2], 1))
    
    if(RETRAIN):
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(KERNEL,KERNEL), input_shape=(train_shape[1],train_shape[2],1)))
        model.add(MaxPooling2D(pool_size=(POOLING,POOLING)))
        model.add(Flatten())
        model.add(Dense(128, activation=tensorflow.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tensorflow.nn.softmax))
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x=train_images, y=train_labels, epochs=EPOCHS)
        model.save("Woche_15/cnn_model.h5")
    else:
        model = load_model("Woche_15/cnn_model.h5")
    
    prediction = model.predict(test_images)
    wrong_images = []
    labels = []
    for (i,p) in enumerate(prediction):
        if(np.argmax(p) != test_labels[i]):
            wrong_images.append(test_images[i])
            labels.append((np.argmax(p), test_labels[i], p[np.argmax(p)]))
    print(f"Found {len(wrong_images)}/{len(test_images)} wrong predictions")
    plt.show()
    for i in range(len(wrong_images)):
        plt.imshow(wrong_images[i].reshape(28,28), cmap="gray")
        plt.title(f"Prediction: {labels[i][0]}\nCorrect: {labels[i][1]}\nConfidence: {labels[i][2]}")
        if plt.waitforbuttonpress():
            break

    plt.close()
    

if __name__ == "__main__":
    main()
