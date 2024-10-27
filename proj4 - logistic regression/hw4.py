import numpy as np
import pandas as pd
from numpy.random import multivariate_normal
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    mk = np.mean(x)
    my = np.mean(y)
    xk_mk = x - mk
    yk_my = y - my
    xk_mk_yk_my = np.dot(xk_mk, yk_my)
    sumx = np.sum(xk_mk ** 2)
    sumy = np.sum(yk_my ** 2)
    r = xk_mk_yk_my / (np.sqrt(sumx * sumy))

    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.
be
    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """

    pearson_list = []
    best_features = []

    for i in range(2, X.shape[1]):
        curr_p = pearson_correlation(X.iloc[:, i], y)
        pearson_list.append((curr_p, i))

    pearson_list = sorted(pearson_list, key=lambda x: abs(x[0]), reverse=True)

    for i in range(n_features):
        _, index = pearson_list[i]
        best_features.append(X.columns[index])

    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        X = np.insert(X, 0, 1, axis=1)  # bias trick
        self.theta = np.random.random(X.shape[1])

        for i in range(self.n_iter):
            x_theta = np.dot(X, self.theta)
            s = 1.0 / (1.0 + np.exp(-x_theta))  # sigmoid function
            self.theta = self.theta - self.eta * np.dot(X.T, (s - y))

            # calculate cost function
            J = self.cost_function(X, y)
            self.Js.append(J)
            self.thetas.append(self.theta.copy())

            if len(self.Js) > 1 and abs(
                    J - self.Js[-2]) < self.eps:  # stop gradiant desent if the improvment is less then aps
                break

    def cost_function(self, X, y):
        x_theta = np.dot(X, self.theta)
        s = 1.0 / (1.0 + np.exp(-x_theta))  # sigmoid function
        J = (-1.0 / len(y)) * (np.dot(y.T, np.log(s + self.eps)) + np.dot((1 - y).T, np.log(1 - s + self.eps)))
        return J

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        X = np.insert(X, 0, 1, axis=1)  # bias trick
        x_theta = np.dot(X, self.theta)
        s = 1.0 / (1.0 + np.exp(-x_theta))  # sigmoid function
        preds = np.round(s)  # round the probability from the sigmoid to 0 or 1 for classification

        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_size = X.shape[0] // folds
    accuracies = []

    for i in range(folds):
        validation_start = i * fold_size
        validation_end = (i + 1) * fold_size if i < folds - 1 else X.shape[0]

        X_validation = X[validation_start:validation_end]
        y_validation = y[validation_start:validation_end]

        X_train = np.concatenate((X[:validation_start], X[validation_end:]), axis=0)
        y_train = np.concatenate((y[:validation_start], y[validation_end:]), axis=0)

        algo.fit(X_train, y_train)
        pred = algo.predict(X_validation)

        accuracy = np.mean(pred == y_validation)
        accuracies.append(accuracy)

    return np.mean(accuracies)


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    fraction = (1 / (sigma * np.sqrt(2 * np.pi)))
    exponent = np.square((data - mu) / sigma)
    e = np.exp(-0.5 * exponent)
    return fraction * e


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes].reshape(self.k)
        self.sigmas = np.random.random_integers(self.k)
        self.weights = np.ones(self.k) / self.k

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """

        res = self.weights * norm_pdf(data, self.mus, self.sigmas)
        sum = np.sum(res, axis=1, keepdims=True)
        self.responsibilities = res / sum

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """

        self.weights = np.mean(self.responsibilities, axis=0)
        self.mus = np.sum(self.responsibilities * data.reshape(-1, 1), axis=0) / np.sum(self.responsibilities, axis=0)
        variance = np.mean(self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0)
        self.sigmas = np.sqrt(variance / self.weights)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        self.costs.append(self.compute_cost(data))
        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.compute_cost(data)
            if self.costs[-1] - cost < self.eps:
                if self.costs[-1] > cost:
                    self.costs.append(cost)
                break
            self.costs.append(cost)

    def compute_cost(self, data):
        sum = 0
        costs = self.weights * norm_pdf(data, self.mus, self.sigmas)
        for i in range(len(data)):
            sum = sum + costs[i]
        return -1 * np.sum(np.log(sum))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    pdf = np.sum(weights * norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)

    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = {}
        self.em_models = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # Calculate prior of each class
        y_labels, y_count = np.unique(y, return_counts=True)
        for i in range(len(y_count)):
            self.prior[y_labels[i]] = y_count[i] / len(y)

        for class_label in y_labels:
            # Create a dictionary for each class to store EM models for each feature
            feature_models = {}

            for feature in range(X.shape[1]):
                # Create an EM model for each feature
                feature_models[feature] = EM(self.k)

            # Assign the feature models dictionary to the current class label
            self.em_models[class_label] = feature_models

        for label in self.em_models.keys():
            for feature in self.em_models[label].keys():
                self.em_models[label][feature].fit(X[y == label][:, feature].reshape(-1, 1))

    def likelihoods(self, X, label):
        likelihood = 1
        for feature in range(X.shape[0]):
            weights, mus, sigmas = self.em_models[label][feature].get_dist_params()
            likelihood *= gmm_pdf(X[feature], weights, mus, sigmas)
        return likelihood

    def posterior(self, X, label):
        return self.prior[label] * self.likelihoods(X, label)

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []

        # Iterate over each instance in the dataset X
        for instance in X:
            # Initialize a list to store the posterior probabilities for the current instance
            posteriors = []

            # Calculate the posterior probability for each class
            for class_Label in self.prior.keys():
                # Compute the posterior probability of the current instance belonging to the current class
                posterior = self.posterior(instance, class_Label)

                # Store the posterior probability along with the class label
                posteriors.append((posterior, class_Label))

            # Find the class with the highest posterior probability
            best_class = max(posteriors, key=lambda t: t[0])[1]

            # Append the predicted class to the list of predicted classes
            preds.append(best_class)
        predsnp = np.asarray(preds)
        return predsnp

#copied from notebook
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    lor_train_acc = np.mean(lor.predict(x_train) == y_train)
    lor_test_acc = np.mean(lor.predict(x_test) == y_test)

    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    bayes_train_acc = np.mean(bayes.predict(x_train) == y_train)
    bayes_test_acc = np.mean(bayes.predict(x_test) == y_test)

    plot_decision_regions(x_train, y_train, lor,
                          title="Logistic regression decision boundary")
    plot_decision_regions(x_train, y_train, bayes,
                          title="Naive Bayes decision boundary")

    # plot lor's cost(iterations) function
    cost_plot(lor)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def cost_plot(lor):
    plt.figure()
    plt.plot(list(range(len(lor.Js))), lor.Js)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Cost as a function of iterations')
    plt.show()


def generate_datasets():
    np.random.seed(42)

    # Dataset A - Naive Bayes will work better
    # data is in 4 "groups" in 3d so not linearly separable
    dataset_a_features = np.empty((5000, 3))
    dataset_a_labels = np.empty((5000))
    a_means = [[2, 2, 2], [5, 5, 5], [8, 8, 8], [11, 11, 11]]
    a_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    a_sizes = [1250, 1250, 1250, 1250]
    a_labels = [0, 1, 0, 1]

    start_idx = 0
    for mean, size, label in zip(a_means, a_sizes, a_labels):
        end_idx = start_idx + size
        dataset_a_features[start_idx:end_idx] = np.random.multivariate_normal(mean, a_cov, size)
        dataset_a_labels[start_idx:end_idx] = label
        start_idx = end_idx

    # Dataset B - Logistic Regression will work better
    mean_b0 = [1, 1, 1]
    cov_b0 = [[2, 0.95, 5], [0.95, 1, 2], [4, 7, 1]]

    mean_b1 = [5, 5, 5]
    cov_b1 = [[1, 0.95, 0.9], [0.95, 1, 0.95], [0.9, 0.95, 1]]

    data_b0 = np.random.multivariate_normal(mean=mean_b0, cov=cov_b0, size=1000)
    data_b1 = np.random.multivariate_normal(mean=mean_b1, cov=cov_b1, size=1000)

    dataset_b_features = np.vstack((data_b0, data_b1))
    dataset_b_labels = np.hstack((np.zeros(1000), np.ones(1000)))

    return {
        'dataset_a_features': dataset_a_features,
        'dataset_a_labels': dataset_a_labels,
        'dataset_b_features': dataset_b_features,
        'dataset_b_labels': dataset_b_labels
    }

