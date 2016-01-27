import numpy as np
import sys

def load_file(fname):
    ''' load a file into memory '''
    data = []
    with open(fname,'rb') as f:
        data = f.readlines()
    return data

def preparse(data):
    ''' preparse given data and compose a data matrix and target vector
    Data is of the form: +1 5:1 8:1 18:1 22:1 36:1 40:1 51:1 61:1 67:1 72:1 75:1 76:1 80:1 83:1 '''
    X = []
    y = []
    for i, vector in enumerate(data):
        X.append([])
        for j in range(124): # 124 is given dimension
            X[i].append(0) # initialize data matrix to zeroes
        vector = vector.split()
        y.append(int(vector.pop(0))) # pop our target vector
        for feature in vector:
            t = feature.split(':')
            X[i][int(t[0])] = 1

    X_mat = np.matrix(X, dtype=int) # convert our data array to matrix
    y_mat = np.array(y,dtype = int) # convert our target to vector
    return X_mat, y_mat

def compute_model(X, y, l = 78):
    ''' identity matrix X, target vector y, arbitrary lambda l '''
    n_col = X.shape[1]
    f = np.linalg.lstsq(X.T.dot(X) + l * np.identity(n_col), np.squeeze(np.asarray(X.T.dot(y)))) # Regularized linear regression w/ normal equation -> solve for t, t = (X^TX + lI)^{-1}X^Ty.
    return f

def main():
    train_file = load_file("a7a.train")
    X, y= preparse(train_file)
    W = compute_model(X,y)

    test_file = load_file(sys.argv[1])
    X2, y2 = preparse(test_file)
    result = np.squeeze(np.asarray(X2)).dot(W[0]) # compute our result

    c = 0.0
    for i, r in enumerate(result): # calculate accuracy
        if r > 0 and y2[i] == 1:
            c = c + 1.0
        if r < 0 and y2[i] == -1:
            c = c + 1.0

    print(str(c) + " correct predictions for " + str(y2.shape[0]) + " points")
    print("The accuracy is " + str(c/y2.shape[0]))

if __name__ == "__main__":
    main()
