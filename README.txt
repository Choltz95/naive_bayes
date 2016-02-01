me: Chester Holtz
Email: choltz2@u.rochester.edu
Course: CSC246
Homework: Implement naive bayes with Dirichlet smoothing for the adult income dataset. Plot classification error on the dev set as a function of alpha, and report the performance of the best alpha on the test set.

************ Files *********
README.txt This file
naive_bayes.py My implementation of a naive bayes classifier.
a7a.train data file required to train the weights for our classifier.

************ Algorithm *****
Preprocessing
I first perform preprocessing on the dataset using regular expressions - I import the file into memory and producing a numpy data matrix and associated target vector of training/dev/test sets.

Naive Bayes
The goal of the Naive Bayes classification is to compute probabilities for each label given a vector of features (x_1,...,x_d).

************ Instructions ***
python naive_bayes.py {your test file}

************ Results *******
Results on the dev and test set where good - mid to high 70% accuracy with little to no interaction with the smoothing constant alpha (testing done on l = 1, 5, 10,..., 1000 with the optimal accuracy on the dev set being achieved at a = 1.)
Specifically, the output of my program running on the test set with alpha = 1.0 was:

choltz2@cycle2 naive_bayes]$ python naive_bayes.py a7a.test
training...
positive prior: 0.243354037267
negative prior: 0.756645962733
testing...
6403.0 correct predictions for 8461 points
The accuracy is 0.756766339676

************ Your interpretation *******
I was not expecting Naive Bayes to perform with worse average accuracy compared to linear regression. I can attribute this underperformace to an underdependence on our smooth constant alpha. With more time I would have liked to further experiment with various alphas, and perhaps implement a  scheme to derive the optimal regularization coefficient given a dev set. I provide a graphical representation of the relationship between alphas and the resultant accuracy (accuracy as a function of alpha) when used with the algorithm on the adult dataset using gnuplot below.

I tested 1000 alphas from 1 to 1000 with the hope of performing binary search to reduce error. I found that varying my alpha had no affect on the resulting error on the dataset. The error remained constant at 0.766875 on the dev dataset.

  0.766 +-+----+-----+------+------+-----+------+------+------+-----+----+-+
        +      +     +      +      +     +      +      +      +     +      +
  0.764 +-+                                                   f(x) +-----+-+
        |                                                                  |
  0.762 +-+                                                              +-+
        |                                                                  |
   0.76 +-+                                                              +-+
        |                                                                  |
  0.758 +-+                                                              +-+
        |                                                                  |
        |++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++|
  0.756 +-+                                                              +-+
        |                                                                  |
  0.754 +-+                                                              +-+
        |                                                                  |
  0.752 +-+                                                              +-+
        |                                                                  |
   0.75 +-+                                                              +-+
        +      +     +      +      +     +      +      +      +     +      +
  0.748 +-+----+-----+------+------+-----+------+------+------+-----+----+-+
        0     100   200    300    400   500    600    700    800   900    1000

************ References ************
Textbook
