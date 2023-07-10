from re import X
import numpy as np


def pca(x: np.ndarray, Q: int) -> np.ndarray:
    # dataset size
    N = x.shape[0]
    
    x_bar = 1/N * np.sum(x, axis=0)
    
    # calculate scatter matrix
    S = []
    for i in range(N):
        new_s = np.matrix((x_bar - x[i]))
        new_s_T = new_s.T
        prod = np.dot(new_s_T, new_s)
        S.append(prod)
    S = np.sum(S, axis=0)
   
    
    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    
    # sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]

    # get Q top eigenvectors
    top_eigenvectors = eigenvectors[idx]
    top_eigenvectors = np.matrix(top_eigenvectors[:Q,:]).T
    
    # get projection
    Z = np.dot(top_eigenvectors.T , x.T)
    
    return Z.T
    
    
def main():
    # generate dataset
    dataset_size = 100
    dataset_dimension = 3
    np.random.seed(0)
    dataset = np.random.randint(0, 100, (dataset_size, dataset_dimension))
    
    print("Dataset: ",dataset)
    
    z = pca(dataset, 2)
    print ("PCA: ", z)

if __name__ == "__main__":
    main()
