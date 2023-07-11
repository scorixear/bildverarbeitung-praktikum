from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import tensorflow
import matplotlib.pyplot as plt
import numpy as np

# best configuration here
# 3,4 gives best accuracy
KERNEL = 3
POOLING = 4
# heavily affects training time
EPOCHS = 10
# if false, cnn_model.h5 will be loaded instead of retraining
RETRAIN = False


def main():
    # load in mnist data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # reshape to [n, x, y, 1] since we have only 1 color channel
    train_shape = train_images.shape
    train_images = train_images.reshape((train_shape[0], train_shape[1], train_shape[2], 1))
    test_shape = test_images.shape
    test_images = test_images.reshape((test_shape[0], test_shape[1], test_shape[2], 1))
    
    if(RETRAIN):
        # initialize model
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(KERNEL,KERNEL), input_shape=(train_shape[1],train_shape[2],1)))
        model.add(MaxPooling2D(pool_size=(POOLING,POOLING)))
        model.add(Flatten())
        model.add(Dense(128, activation=tensorflow.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tensorflow.nn.softmax))
        # and train it
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x=train_images, y=train_labels, epochs=EPOCHS)
        # save to cnn_model.h5
        model.save("Woche_15/cnn_model.h5")
    else:
        model = load_model("Woche_15/cnn_model.h5")
    
    # predict test/validation images
    prediction = model.predict(test_images)
    # list of predictions that failed
    wrong_images = []
    # there labels as Tuples (prediction, correct, confidence)
    labels = []
    # for each prediction
    for (i,p) in enumerate(prediction):
        # if the current prediction is wrong
        if(np.argmax(p) != test_labels[i]):
            # append the prediction to wrong_images
            wrong_images.append(test_images[i])
            labels.append((np.argmax(p), test_labels[i], p[np.argmax(p)]))

    print(f"Found {len(wrong_images)}/{len(test_images)} wrong predictions")
    # prepare plot
    # by showing the empty plot
    plt.show()
    # go through all wrong predictions
    for i in range(len(wrong_images)):
        # show the image (drawing over the plot)
        plt.imshow(wrong_images[i].reshape(28,28), cmap="gray")
        # and resetting the title
        plt.title(f"Prediction: {labels[i][0]}\nCorrect: {labels[i][1]}\nConfidence: {labels[i][2]}")
        # if button = Mouse, this is False -> show next image
        # if button = Keyboard, this is True -> End programm
        if plt.waitforbuttonpress():
            break
    # close plot
    plt.close()
    

if __name__ == "__main__":
    main()
