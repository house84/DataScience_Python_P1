Author: Nick House
Language: Python
Class: CS-4200 Python for Scientific Computing and Data Science
Project: P1

# DataScience_Python_P1
KNN Classification (using numpy broadcasting and vectorization)

This program Solves the following requirments:

In this project you will implement K-nearest-neighbor classification following the given specifications. Given a number of examples of d-dimensional vectors and their class labels (a label is one-dimensional), the problem is to find (predict or estimate) the class label of points that are not in the given set of examples. The K-NN algorithm predicts the class of a new point by looking at the classes of K of the new point’s neighbors in the existing examples, the underlying assumption being that “a man is known by the company he keeps.”
In what follows, the use of the word set is not to be understood as meaning a mathematical set or the Python sequence type set.
The K-NN Algorithm may be described as follows:
Input: (1) The (labeled) training set of size n; (2) test points (assumed labeled); (3) integer K (number of nearest neighbors)
Output: (1) The predicted class labels of the test points; (2) the accuracy of classification (success rate, expressed as a fraction of predictions matching known labels out of all the test points)
For each test point:
1. Find its Euclidean distance from each of the n points in the training data set
2. Pick the nearest K points
3. Output the class by weighted voting using the K nearest neighbors in the above step (use the inverse of the distance as the weight)
Read the description of the classification problem and the K-NN algorithm from Chapter 15 (in particular Section 15.2.1) of the textbook. (But do NOT use anything from the scikit-learn package.)

Specifications: Please use the exact names (identifiers) specified below (for quantities not mentioned here, you are free to choose your own names):
• Number of samples (data points or vectors) in the training set: n
• Total number of samples (data points or vectors) in the training set plus test set: TOTAL_SAMPLE_SIZE
• Number of features: d
• Stipulated number of nearest neighbors: K
• Mean and standard deviation of the normal distribution: m and s, respectively
• Data structure(s) (typically ndarray) holding all the training data and the corresponding labels: X_train, y_train
• Data structure(s) (typically ndarray) holding all the test data and the corresponding labels: X_test, y_test

Randomly generate a total of TOTAL_SAMPLE_SIZE labeled data points (vectors), each d-dimensional. Each of the d features (variables) should be created from a normal (Gaussian) distribution with mean m and standard deviation s. There are three classes (labels), each represented as a string: “good”, “bad”, and “ugly”. Thus each data point is a vector of d real-valued components and has a label chosen uniformly randomly from the above three strings. Do NOT check for possible repetitions in the data set of size TOTAL_SAMPLE_SIZE. Do NOT transform the data by standardization, normalization or any other method. You may hold the feature vectors and the corresponding class labels in the same data structure (e.g., ndarray) or in two different data structures. Of the entire data set of size TOTAL_SAMPLE_SIZE, take out n items (via uniform random sampling without replacement) as the training data or, equivalently, take out TOTAL_SAMPLE_SIZE – n items (via uniform random sampling without replacement) as the test data.
Define and use a simple function called weight() that takes as input a distance and outputs a weight as 1/distance (or something like 1 / (0.001 + distance) to avoid potential division-by-zero situations).

Use the normal method of default_rng from numpy (see “Quick Start” at https://numpy.org/doc/stable/reference/random/index.html#quick-start), not Python’s own random module. Set the seed at the beginning of your program so that your results are reproducible.
Please use the fewest possible number of (i) explicit loops and (ii) list construction/comprehension. You will have to look up quite a few methods/functions from the numpy doc. The aim of this project is to make students acquire a certain level of mastery in the use of the most important feature in Python/Numpy, namely vectorization/broadcasting. As in all programming projects, one can solve this problem in multiple ways. (You might find the following hint useful: np.XXX…XXX.choice, np.linalg.nXXX, np.argXXXX.)
Do NOT use any package where the K-NN algorithm is already available. Also, do NOT use data from any repository or any package for creating and/or loading ready-made data. This project requires you to create and analyze your own data, which is in and of itself an extremely valuable learning exercise. In short, do not use any module/package/function other than those in Python and in Numpy and the modules discussed in the lectures.
Your program should be able to handle any (within reasonable limits, of course) n, TOTAL_SAMPLE_SIZE, d, K, m, and s. Before final submission on Canvas, please run your program with the following numerical values (run the whole notebook once so that the cells are numbered from 1; do not show any temporary or debugging code).
• TOTAL_SAMPLE_SIZE = 10
• n = 7
• d = 3
• K = 3
• m = 5
• s = 2
The very last cells in the notebook should show, separately, the output of two different runs (executions) of your program (corresponding to two different seeds), in a reasonably easy-to-see 2D array form (just use python’s print function; nothing fancy is needed). The output of each run should include: all the training data, training labels, test data, test (known) labels, predicted labels, success rate, and the seed value.
