import numpy as np

def pca(x: np.ndarray, Q: int) -> np.ndarray:
    # dataset size
    N = x.shape[0]
    # get average of dataset
    x_bar = 1/N * np.sum(x, axis=0)
    
    # calculate scatter matrix
    # should result in DxD matrix
    S = []
    # for each entry in dataset
    for i in range(N):
        s_entry = np.matrix((x_bar - x[i]))
        S.append(np.dot(s_entry.T, s_entry))
    # S is here a NxDxD Matrix
    # Collapse to DxD Matrix
    S = np.sum(S, axis=0)
   
    # get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    # sort eigenvalues and eigenvectors
    # argsort() sort ascending
    # [::-1] reverses the order most efficiently
    idx = eigenvalues.argsort()[::-1]

    # sort eigenvectors by eigenvalues
    top_eigenvectors = eigenvectors[idx]
    # and get the top Q eigenvectors
    # Q is the number of dimensions we want to reduce to
    # transpose this to be a DxQ matrix
    top_eigenvectors = np.matrix(top_eigenvectors[:Q,:]).T
    
    # get projection
    # Z is a NxQ matrix
    # x is a NxD matrix, transposed to be a DxN matrix
    # top_eigenvectors is a DxQ matrix, transposed to be a QxD matrix
    # QxD * DxN = QxN, transposed to NxQ
    Z = np.dot(top_eigenvectors.T , x.T)
    return Z.T

def main():
    # generate dataset
    dataset_size = 100
    dataset_dimension = 4
    # set seed for reproducibility
    np.random.seed(0)
    # fill dataset with random values
    # results in a NxD matrix (np.ndarray)
    dataset = np.random.randint(0, 100, (dataset_size, dataset_dimension))
    
    print("Dataset: ",dataset)
    
    # calculate pca
    # returns the projected dataset
    z = pca(dataset, 2)
    print ("PCA: ", z)

if __name__ == "__main__":
    main()
