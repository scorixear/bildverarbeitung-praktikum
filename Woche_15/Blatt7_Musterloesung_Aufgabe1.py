from time import time
import sklearn.metrics as metrics
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EPOCHS = 10
DROPOUT = 0.2#
ENABLE_RETRAINING = True

def main():
    now = time()
    # load in 2d images and their labels
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # reshape images into 1d vectors
    train_shape = train_images.shape
    train_images = train_images.reshape((train_shape[0], train_shape[1], train_shape[2], 1))
    test_shape = test_images.shape
    test_images = test_images.reshape((test_shape[0], test_shape[1], test_shape[2], 1))
    
    df = pd.DataFrame(columns= ["Kernel", "Pooling", "Loss", "Accuracy", "Precision", "Recall", "F1"])
    
    if(ENABLE_RETRAINING):
        for k in range(3,6):
            for p in range(2,6):
                print(f"Training with Kernel size: {k} and Pooling size: {p}")
                st = time()
                # create model
                model = Sequential()
                model.add(Conv2D(28, kernel_size=(k,k), input_shape=(train_shape[1],train_shape[2],1)))
                model.add(MaxPooling2D(pool_size=(p,p)))
                model.add(Flatten())
                model.add(Dense(128, activation=tensorflow.nn.relu))
                model.add(Dropout(DROPOUT))
                model.add(Dense(10, activation=tensorflow.nn.softmax))
                
                
                # train model
                model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                model.fit(x=train_images, y=train_labels, epochs=EPOCHS)
                (loss, accuracy) = model.evaluate(test_images, test_labels)
                
                prediction = model.predict(test_images)
                pred_bool = np.argmax(prediction, axis=1)
                precision = metrics.precision_score(test_labels, pred_bool, average="macro")
                recall = metrics.recall_score(test_labels, pred_bool, average="macro")
                f1_score = metrics.f1_score(test_labels, pred_bool, average="macro")
                
                df.loc[len(df)] = [k, p, loss, accuracy, precision, recall, f1_score] # type: ignore
                end = time() - st
                print(f"Training finished in {end}")
        print(f"Total time taken: {time() - now}")
        df.to_csv("Woche_15/cnn_results.csv")
    else:
        df = pd.read_csv("Woche_15/cnn_results.csv")

    save_plots(df)

def save_plots(dataset: pd.DataFrame):
    loss_k = []
    accuracy_k = []
    precision_k = []
    recall_k = []
    f1_score_k = []
    pooling_values = [0,1,2,3]
    pooling_labels = [2,3,4,5]
    for k in range(3,6):
        
        loss_k.append([dataset.query(F"Kernel == {k} & Pooling == {val}")["Loss"].item() for val in pooling_labels])
        accuracy_k.append([dataset.query(F"Kernel == {k} & Pooling == {val}")["Accuracy"].item() for val in pooling_labels])
        precision_k.append([dataset.query(F"Kernel == {k} & Pooling == {val}")["Precision"].item() for val in pooling_labels])
        recall_k.append([dataset.query(F"Kernel == {k} & Pooling == {val}")["Recall"].item() for val in pooling_labels])
        f1_score_k.append([dataset.query(F"Kernel == {k} & Pooling == {val}")["F1"].item() for val in pooling_labels])
    
    fig, axis = plt.subplots(2,3, figsize=(10,5))
    
    
    for k in range(3,6):
        axis[0,0].plot(pooling_values, loss_k[k-3], label=F"Kernel size: {k}")
        axis[0,1].plot(pooling_values, accuracy_k[k-3], label=F"Kernel size: {k}")
        axis[0,2].plot(pooling_values, precision_k[k-3], label=F"Kernel size: {k}")
        axis[1,0].plot(pooling_values, recall_k[k-3], label=F"Kernel size: {k}")
        axis[1,1].plot(pooling_values, f1_score_k[k-3], label=F"Kernel size: {k}")
        
    
    set_axis(pooling_values, pooling_labels, axis[0,0],"Pooling", "Loss")
    set_axis(pooling_values, pooling_labels, axis[0,1],"Pooling", "Accuracy")
    set_axis(pooling_values, pooling_labels, axis[0,2],"Pooling", "Precision")
    set_axis(pooling_values, pooling_labels, axis[1,0],"Pooling", "Recall")
    set_axis(pooling_values, pooling_labels, axis[1,1],"Pooling", "F1 Score")

    
    fig.tight_layout(pad=1.0)
    plt.show()

def set_axis(x_values, x_labels, axis, x_label, y_label):
    axis.set_xticks(ticks=x_values, labels=x_labels)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(F"{y_label} over {x_label}")
    axis.legend()
    
    
        
if __name__ == "__main__":
    
    main()
    