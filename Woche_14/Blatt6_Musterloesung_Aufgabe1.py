from time import time
from enum import Enum
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# stops annoying warnings from sklearn
import warnings
warnings.filterwarnings('ignore')

# represent the required solvers and the possible solvers in sklearn
class MLPSolver(str, Enum):
    Adam = "adam"
    Momentum = "sgd"
    Standard = "lbfgs"
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value

def main():
    print("Loading MNIST data")
    # load in 2d images and their labels
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # reshape images into 1d vectors
    train_shape = train_images.shape
    train_images = train_images.reshape((train_shape[0], train_shape[1]*train_shape[2]))
    test_shape = test_images.shape
    test_images = test_images.reshape((test_shape[0], test_shape[1]*test_shape[2]))
    
    # specify parameter ranges
    # Adam = adam, Momentum = sgd, Standard = lbfgs
    solver_range = [MLPSolver.Adam, MLPSolver.Momentum, MLPSolver.Standard]
    layer_range = [2,3,4]
    learning_rate_range = [0.1, 0.01]
    maximum_iteration_range = [5, 10, 15]

    # dataframe for storing results
    df = pd.DataFrame(columns = ["Solver", "Layers", "Learning_Rate", "Maximum_Iteration", "Training_Accuracy", "Test_Accuracy", "Training_Precision", "Training_Recall", "Training_F1_Score", "Training_Time"])
    # row index in dataframe
    i = 0
    # for each configuration
    for solver in solver_range:
        for layer_num in layer_range:
            for learning_rate in learning_rate_range:
                for maximum_iteration in maximum_iteration_range:
                    print(f"Training MLP with {solver.value} solver, {layer_num} layers, {learning_rate} learning rate and {maximum_iteration} maximum iteration")
                    # start measuring time
                    st = time()
                    # layers [2, 3, 4] into (100, 100,), (100, 100, 100, ) and (100, 100, 100, 100,)
                    # 100 is the number of nerons each layer has
                    layers = tuple([100 for _ in range(layer_num)])
                    
                    # create classifier
                    mlp = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=learning_rate, max_iter=maximum_iteration, solver=solver.value)
                    # and train it
                    # may result in non-finite weights
                    # which will crash this
                    try:
                        mlp.fit(train_images, train_labels)
                    except Exception as error:
                        print("Error occurded", error)
                        continue
                    # stop measuring time
                    end = time() - st
                    print(f"Training finished in {end}")
                    print("Calculating stats")
                    
                    # calculate stats from the predictions
                    tr_predict = mlp.predict(train_images)
                    test_predict = mlp.predict(test_images)
                    acc_tr = metrics.accuracy_score(train_labels, tr_predict)
                    acc_test = metrics.accuracy_score(test_labels, test_predict)
                    prec_tr = metrics.precision_score(train_labels, tr_predict, average="macro")
                    recall_tr = metrics.recall_score(train_labels, tr_predict, average="macro")
                    f1_tr = metrics.f1_score(train_labels, tr_predict, average="macro")
                    
                    # and save predictions to dataframe
                    df.loc[i] = [solver.value, layer_num, learning_rate, maximum_iteration, acc_tr, acc_test, prec_tr, recall_tr, f1_tr, end] # type: ignore
                    i = i+1
    # save results to csv
    df.to_csv("Woche_14/MLP_Results.csv")
    
    # create plots for each configuration, plotting the maximum iteration range
    for solver in solver_range:
        for layer_num in layer_range:
            for learning_rate in learning_rate_range:
                x_ticks = np.arange(len(maximum_iteration_range))
                save_plots(maximum_iteration_range, x_ticks, df.query(f'Solver == "{solver}" & Layers == {layer_num} & Learning_Rate == {learning_rate}'), "Maximum_Iteration", f"maximum_iteration_{solver}_{layer_num}_{learning_rate}_x", "Maximum Iteration")
    # create plots for each configuration, plotting the learning rate range
    for solver in solver_range:
        for layer_num in layer_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(learning_rate_range))
                save_plots(learning_rate_range, x_ticks, df.query(f'Solver == "{solver}" & Layers == {layer_num} & Maximum_Iteration == {maximum_iteration}'), "Learning_Rate", f"learning_Rate_{solver}_{layer_num}_x_{maximum_iteration}", "Learning Rate")
    # create plots for each configuration, plotting the layer range
    for solver in solver_range:
        for learning_rate in learning_rate_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(layer_range))
                save_plots(layer_range, x_ticks, df.query(f'Solver == "{solver}" & Maximum_Iteration == {maximum_iteration} & Learning_Rate == {learning_rate}'), "Layers", f"layers_{solver}_x_{learning_rate}_{maximum_iteration}", "Layers")
    # save plots for each configuration, plotting the solver range
    for layer_num in layer_range:
        for learning_rate in learning_rate_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(solver_range))
                save_plots(solver_range, x_ticks, df.query(f'Maximum_Iteration == {maximum_iteration} & Layers == {layer_num} & Learning_Rate == {learning_rate}'), "Solver", f"solver_x_{layer_num}_{learning_rate}_{maximum_iteration}", "Solver")
    
    # results show the following:
    # lbgfs give best results, while adam performs equally good with less layers
    # 3 layers are optimum, 4 takes longer, but gives no significant improvement
    # lbgs does like less layers
    # 0.01 is the best learning rate overall
    # 15 iterations are the best overall, except for adam
    # sgd is stoic in general, having no effect in layers, learning rate and maximum iteration
    # this could be due to not small enough learning rate or not enough iterations

def save_plots(x_labels, x_values, dataset, column, file_name, f_label):
    """Saves the given dataset to one plot with 6 figures, one for each metric

    Args:
        x_labels (list[str]): The actual labels on the x axis
        x_values (list[int]): The actual values on the x axis (for space between labels)
        dataset (pandas.dataframe): the pandas dataframe to get the value from
        column (str): the column to get the y values from
        file_name (str): file name the plot will be saved to
        f_label (str): The metric that is plotted in human writeable form (will be displayed in the plot)
    """
    try:
        # if the column is solver, the values are strings and need to be handled
        # differently in a pandas query (this is dumb btw)
        if(column == 'Solver'):
            # go through all values 
            acc_tr = [dataset.query(F'{column} == "{val}"')["Training_Accuracy"].item() for val in x_labels]
            acc_test = [dataset.query(F'{column} == "{val}"')["Test_Accuracy"].item() for val in x_labels]
            precision = [dataset.query(F'{column} == "{val}"')["Training_Precision"] for val in x_labels]
            recall = [dataset.query(F'{column} == "{val}"')["Training_Recall"].item() for val in x_labels]
            f1_score = [dataset.query(F'{column} == "{val}"')["Training_F1_Score"].item() for val in x_labels]
            tr_time = [dataset.query(F'{column} == "{val}"')["Training_Time"].item() for val in x_labels]
        else:
            acc_tr = [dataset.query(F"{column} == {val}")["Training_Accuracy"].item() for val in x_labels]
            acc_test = [dataset.query(F"{column} == {val}")["Test_Accuracy"].item() for val in x_labels]
            precision = [dataset.query(F"{column} == {val}")["Training_Precision"].item() for val in x_labels]
            recall = [dataset.query(F"{column} == {val}")["Training_Recall"].item() for val in x_labels]
            f1_score = [dataset.query(F"{column} == {val}")["Training_F1_Score"].item() for val in x_labels]
            tr_time = [dataset.query(F"{column} == {val}")["Training_Time"].item() for val in x_labels]
    except:
        return
    
    fig, axis = plt.subplots(3,2,figsize=(10, 5))
    if (len(acc_tr) < len(x_labels)):
        return
    plot_single_axis(axis[0,0], x_labels, x_values, acc_tr, f_label, "Accuracy", "Training Accuracy", "Training Accuracy versus " + f_label)
    plot_single_axis(axis[0,1], x_labels, x_values, acc_test, f_label, "Accuracy", "Test Accuracy", "Test Accuracy versus " + f_label)
    plot_single_axis(axis[1,0], x_labels, x_values, precision, f_label, "Precision", "Training Precision", "Training Precision versus " + f_label)
    plot_single_axis(axis[1,1], x_labels, x_values, recall, f_label, "Recall", "Training Recall", "Training Recall versus " + f_label)
    plot_single_axis(axis[2,0], x_labels, x_values, f1_score, f_label, "F1 Score", "Training F1 Score", "Training F1 Score versus " + f_label)
    plot_single_axis(axis[2,1], x_labels, x_values, tr_time, f_label, "Time", "Training Time", "Training Time versus " + f_label)
    
    fig.tight_layout(pad=1.0)
    plt.savefig("Woche_14/results/" + file_name + ".png", dpi=300)

def plot_single_axis(axis, x_labels, x_values, y_values, x_label, y_label, f_label, title):
    axis.plot(x_values, y_values, label=f_label)
    axis.set_xticks(ticks=x_values, labels=x_labels)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title, fontsize=12)

if __name__ == "__main__":
    now = time()
    main()
    print(f"Total time taken: {time() - now}")