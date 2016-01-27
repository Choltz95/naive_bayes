Name: Chester Holtz
Email: choltz2@u.rochester.edu
Course: CSC246
Homework: Implement linear regression for the adult income dataset using Python.

************ Files *********
README.txt This file
regression.py My implementation of a linear regression based classifier.
a7a.train data file required to train the weights for our classifier.

************ Algorithm *****
Preprocessing
I first preform preprocessing on the dataset using regular expressions - I import the file into memory and producing a numpy data matrix and associated target vector.

Regression
The goal of regression is to minimize a cost function with respect to lambda.
For this project we implement simple linear regression with normal regularization. The normal equation solution to regularized linear expression for learning weights is defined as 
t = (X^TX + lI)^{-1}X^Ty.
where X is the data matrix (^T denotes transpose), y is the target vector of the train/dev set, and the matrix following the regularization coeficient l is an nxn identity matrix. I solve for t using numpys least squares solution solver.

************ Instructions ***
python regression.py {your test file}

************ Results *******
Results on the dev and test set where good - mid to high 80% accuracy with little to no interaction with the regularization constant lambda (testing done on l = 0, 1, 5, 10,..., 100 with the optimal accuracy on the dev set being achieved at l = 78.)

************ Your interpretation *******
I was not expecting such good accuracy with such a simple classification scheme. With more time I would have liked to further experiment with various lambdas, and perhaps implement a  scheme to derive the optimal regularization coefficient given a dev set.

************ References ************
Texbook