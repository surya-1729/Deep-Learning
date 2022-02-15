import numpy as np


def load_data(path, num_train):
    """ Load the data matrices
    Input:
    path: string describing the path to a .csv file
          containing the dataset
    num_train: number of training samples
    Output:
    X_train: numpy array of shape num_train x 11
             containing the first num_train many
             data rows of columns 1 to 11 of the
             .csv file.
    Y_train: numpy array of shape num_train
             containing the first num_train many
             data rows of column 12 of the .csv
             file.
    X_test: same as X_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    Y_test: same as Y_train only corresponding to
            the remaining rows after the first 
            num_train many rows.
    """
       
    X_train = np.loadtxt(path, dtype='float', delimiter=';', usecols= np.arange(0, 11), skiprows= 1, max_rows=num_train)
    Y_train = np.loadtxt(path, dtype='float', delimiter=';', usecols=11, skiprows= 1, max_rows= num_train)
    X_test  = np.loadtxt(path, dtype='float', delimiter=';', usecols= np.arange(0, 11), skiprows= num_train+1)
    Y_test  = np.loadtxt(path, dtype='float', delimiter=';', usecols=11,  skiprows= num_train+1)
    return X_train, Y_train, X_test, Y_test


def fit(X, Y):
    """ Fit linear regression model
    Input:
    X: numpy array of shape N x n containing data
    Y: numpy array of shape N containing targets
    Output:
    theta: nump array of shape n + 1 containing weights
           obtained by fitting data X to targets Y
           using linear regression
    """
    X = np.column_stack((X,np.ones(X.shape[0])))
    Xt = np.transpose(X)
    theta = np.matmul(np.linalg.inv(np.matmul(Xt,X)),np.matmul(Xt,Y))
    return theta


def predict(X, theta):
    """ Perform inference using data X
        and weights theta
    Input:
    X: numpy array of shape N x n containing data
    theta: numpy array of shape n + 1 containing weights
    Output:
    Y_pred: numpy array of shape N containig predictions
    """
    X = np.column_stack((X,np.ones(X.shape[0])))
    Y_pred = np.matmul(X,theta)
    return Y_pred


def energy(Y_pred, Y_gt):
    """ Calculate squared error
    Input:
    Y_pred: numpy array of shape N containing prediction
    Y_gt: numpy array of shape N containing targets
    Output:
    se: squared error between Y_pred and Y_gt
    """
    se = np.sum(np.square(Y_pred-Y_gt))
    return se
