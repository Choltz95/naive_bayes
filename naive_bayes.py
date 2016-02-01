import numpy as np
from itertools import combinations,permutations
import sys

class NaiveBayesClassifier(object):
    """
    Naive Bayes Classifier with additive smoothing

    Params
        :alpha - hyperparameter for additive smoothing - \theta_i = (x_i + \alpha)/(N+ \alpha * d). From a Bayesian point of view,
        this corresponds to the expected value of the posterior, using a symmetric posterior distribution.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """
        Trains Naive Bayes Classifier on data X with labels y

        Input
            :X - numpy.array with shape (num_points, num_features)
            :y - numpy.array with shape (num_points, )

        Sets attributes
            :pos_prior - estimate of prior probability of label 1
            :neg_prior - estimate of prior probability of label -1
        """
        self._pos = X[y==1] # all possitively labled vectors
        self._neg = X[y==-1] # all negatively labled vectors
        self._total_pos = float(sum(y==1)) #
        self._total_neg = float(sum(y==-1))

        # sum all instances of a given feature for the training set
        self.given_pos = []
        self.given_neg = []
        for column in self._pos.T:
            self.given_pos.append(np.count_nonzero(column))
        for column in self._neg.T:
            self.given_neg.append(np.count_nonzero(column))

        total = self._total_pos + self._total_neg
        self.pos_prior = self._total_pos / total
        self.neg_prior = self._total_neg / total
        print 'positive prior: ' + str(self.pos_prior)
        print 'negative prior: ' + str(self.neg_prior)

    def predict(self, X):
        """
        Returns log ((P(c=1) / P(c=-1)) * prod_i P(x_i | c=1) / P(x_i | c=-1))
        using additive smoothing

        Input
            :X - numpy.array with shape (num_points, num_features)
                 num_features must be the same as data used to fit model
        """
        m,n = X.shape
        alpha = self.alpha
        tot_neg = self._total_neg
        tot_pos = self._total_pos
        preds = np.zeros(m)
        for i, xi in enumerate(X):
            Pxi_neg = np.zeros(n)
            Pxi_pos = np.zeros(n)
            xi = np.array(xi).flatten()
            d = len(xi)
            for j, v in np.ndenumerate(xi):
                nc = 0
                pc = 0
                if v == 0:
                    nc = self.given_neg[j[0]]
                elif v == 1:
                    pc = self.given_pos[j[0]]
                # Compute probabilities with additive smoothing
                Pxi_neg[j] = (nc + alpha) / (tot_neg + alpha * d)
                Pxi_pos[j] = (pc + alpha) / (tot_pos + alpha * d)

            # Compute log pos / neg class ratio
            preds[i] = np.log(self.pos_prior) + np.sum(np.log(Pxi_pos)) - \
                       np.log(self.neg_prior) - np.sum(np.log(Pxi_neg))
        return preds

def load_file(fname):
    """
    load a file into memory
    """
    data = []
    with open(fname,'rb') as f:
        data = f.readlines()
    return data

def preparse(data):
    """ preparse given data and compose a data matrix and target vector
    Data is of the form: +1 5:1 8:1 18:1 22:1 36:1 40:1 51:1 61:1 67:1 72:1 75:1 76:1 80:1 83:1
    """
    X = []
    y = []
    for i, vector in enumerate(data):
        X.append([])
        for j in range(124): # 124 is given dimension
            X[i].append(0) # initialize data matrix to zeroes
        vector = vector.split()
        y.append(int(vector.pop(0))) # pop elements composing our target vector
        for feature in vector:
            t = feature.split(':')
            X[i][int(t[0])] = 1

    X_mat = np.matrix(X, dtype=int) # convert our data array to matrix
    y_mat = np.array(y,dtype = int) # convert our target to vector
    return X_mat, y_mat

def compute_accuracy(y, y_prime):
    c = 0.0
    a = 0.0
    for i, r in enumerate(y): # calculate accuracy
        if r > 0 and y_prime[i] == 1:
            c = c + 1.0
        if r < 0 and y_prime[i] == -1:
            c = c + 1.0
    a = c / y_prime.shape[0]
    return a

def main():
    train_file = load_file("a7a.train")
    X, y= preparse(train_file)

    model = NaiveBayesClassifier(alpha=1.0)
    print 'training...'
    model.fit(X, y)

    print 'testing...'
    test_file = load_file(sys.argv[1])
    X2,y2 = preparse(test_file)
    result = model.predict(X2)

#    print 'optimize alpha...'
#    for i in range(1, 1000):
#        model = NaiveBayesClassifier(alpha = i)
#        model.fit(X, y)
#        result = model.predict(X2)
#        print i, compute_accuracy(result, y2)

    a = compute_accuracy(result, y2)
    print(str(y2.shape[0] * a) + " correct predictions for " + str(y2.shape[0]) + " points")
    print("The accuracy is " + str(a))

if __name__ == "__main__":
    main()
