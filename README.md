# COMP551-Fall2016-Project2
2nd project for COMP 551 Fall 2016

# To generate training features:
Run `$python feature_vectorize.py` to generate the feature matrices. This will produces several .npz files.

# To run Naive Bayes:
1. Make sure all `X*.npz` and `Y*.csv` files are in the same directory as `naive_bayes.py`
2. Run `$ python naive_bayes.py`

# To run random forest:
1. Make sure all Y matrices are in the root folder (the folder where all .py files are).
2. Make sure all X feature matrices are in the appropriate folder (i.e. 55000 for 55000 features and/or 10000 for 100000 features).
3. Run `$ python random_forest.py -h` to see the help menu
4. Run `$ python random_forest.py` with appropriate parameters.

# To run lib_random_forest: (random forest with scikit-learn model)
Same as random forest but the file is `lib_random_forest.py`. Parameters are hard coded in this file as well (not configurable as inputs).

# To run SVM:
1. Make sure all `X*.npz` and `Y*.csv`files are in the same directory as `svm.py`
2. Run `$ python svm.py`
