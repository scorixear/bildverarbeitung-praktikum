from time import time
from enum import Enum
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

class MLPSolver(str, Enum):
    Adam = "adam"
    Momentum = "sgd"
    Standard = "lbfgs"
    
    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value

class MLPConfig:
    def __init__(self, maximum_iteration: int, learning_rate: float, layers: int, solver: MLPSolver):
        self.maximum_iteration = maximum_iteration
        self.learning_rate = learning_rate
        self.layers = layers
        self.solver = solver
class MLPResults:
    def __init__(self, config: MLPConfig, accuracy: float, precision: float, recall: float, f1_score: float):
        self.config = config
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
def main():
    print("Loading MNIST data")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_shape = train_images.shape
    train_images = train_images.reshape((train_shape[0], train_shape[1]*train_shape[2]))
    test_shape = test_images.shape
    test_images = test_images.reshape((test_shape[0], test_shape[1]*test_shape[2]))
    configurations: list[MLPConfig] = []
    
    # solver_range: list[str] = [MLPSolver.Adam]
    # layer_range = [3]
    # learning_rate_range = [0.1]
    # maximum_iteration_range = [5]
    
    solver_range: list[str] = [MLPSolver.Adam, MLPSolver.Momentum]
    layer_range = [3,4]
    learning_rate_range = [0.1, 0.01]
    maximum_iteration_range = [5,10]
    
    # solver_range = [MLPSolver.Adam, MLPSolver.Momentum, MLPSolver.Standard]
    # layer_range = [2,3,4]
    # learning_rate_range = [0.1, 0.01]
    # maximum_iteration_range = [5, 10, 15]
    
    for solver in solver_range:
        for layers in layer_range:
            for learning_rate in learning_rate_range:
                for maximum_iteration in maximum_iteration_range:
                    configurations.append(MLPConfig(maximum_iteration, learning_rate, layers, solver))
    df = pd.DataFrame(columns = ["Solver", "Layers", "Learning_Rate", "Maximum_Iteration", "Training_Accuracy", "Test_Accuracy", "Training_Precision", "Training_Recall", "Training_F1_Score", "Training_Time"])
    i = 0
    for config in configurations:
        st = time()
        print(f"Training MLP with {config.solver.value} solver, {config.layers} layers, {config.learning_rate} learning rate and {config.maximum_iteration} maximum iteration")
        # layers [2, 3, 4] into (100, 100,), (100, 100, 100, ) and (100, 100, 100, 100,)
        layers = tuple([100 for _ in range(config.layers)])
        
        mlp = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=config.learning_rate, max_iter=config.maximum_iteration, solver=config.solver.value)
        try:
            mlp.fit(train_images, train_labels)
        except Exception as error:
            print("Error occurded", error)
            continue
        
        end = time() - st
        print(f"Training finished in {end}")
        print("Calculating stats")
        
        tr_predict = mlp.predict(train_images)
        test_predict = mlp.predict(test_images)
        acc_tr = metrics.accuracy_score(train_labels, tr_predict)
        acc_test = metrics.accuracy_score(test_labels, test_predict)
        prec_tr = metrics.precision_score(train_labels, tr_predict, average="macro")
        recall_tr = metrics.recall_score(train_labels, tr_predict, average="macro")
        f1_tr = metrics.f1_score(train_labels, tr_predict, average="macro")
        
        df.loc[i] = [config.solver.value, config.layers, config.learning_rate, config.maximum_iteration, acc_tr, acc_test, prec_tr, recall_tr, f1_tr, end]
        i = i+1
    df.to_csv("Woche_14/MLP_Results.csv")
    
    for solver in solver_range:
        for layers in layer_range:
            for learning_rate in learning_rate_range:
                x_ticks = np.arange(len(maximum_iteration_range))
                save_plots(maximum_iteration_range, x_ticks, df.query(f'Solver == "{solver}" & Layers == {layers} & Learning_Rate == {learning_rate}'), "Maximum_Iteration", f"maximum_iteration_{solver}_{layers}_{learning_rate}_x", "Maximum Iteration")
    for solver in solver_range:
        for layers in layer_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(learning_rate_range))
                save_plots(learning_rate_range, x_ticks, df.query(f'Solver == "{solver}" & Layers == {layers} & Maximum_Iteration == {maximum_iteration}'), "Learning_Rate", f"learning_Rate_{solver}_{layers}_x_{maximum_iteration}", "Learning Rate")
    for solver in solver_range:
        for learning_rate in learning_rate_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(layer_range))
                save_plots(layer_range, x_ticks, df.query(f'Solver == "{solver}" & Maximum_Iteration == {maximum_iteration_range} & Learning_Rate == {learning_rate}'), "Layers", f"layers_{solver}_x_{learning_rate}_{maximum_iteration}", "Layers")
    for layers in layer_range:
        for learning_rate in learning_rate_range:
            for maximum_iteration in maximum_iteration_range:
                x_ticks = np.arange(len(solver_range))
                save_plots(solver_range, x_ticks, df.query(f'Maximum_Iteration == {maximum_iteration} & Layers == {layers} & Learning_Rate == {learning_rate}'), "Solver", f"solver_x_{layers}_{learning_rate}_{maximum_iteration}", "Solver")
    
    

def save_plots(x_labels, x_values, dataset, column, file_name, f_label):
    try:
        if(column == 'Solver'):
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
    except Exception as error:
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