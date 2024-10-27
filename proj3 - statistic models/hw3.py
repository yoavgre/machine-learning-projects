import numpy as np



class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0,
            (0, 1): 0.4,
            (1, 0): 0.4,
            (1, 1): 0.4
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): self.X[0] * self.C[0],
            (0, 1): self.X[0] * self.C[1],
            (1, 0): self.X[1] * self.C[0],
            (1, 1): self.X[1] * self.C[1]
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): self.Y[0] * self.C[0],
            (0, 1): self.Y[0] * self.C[1],
            (1, 0): self.Y[1] * self.C[0],
            (1, 1): self.Y[1] * self.C[1]
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            # p(X,C)/p(c)*p(y,c)/p(c)*p(c)
            (0, 0, 0): (self.X_C[(0, 0)] / self.C[0]) * (self.Y_C[(0, 0)] / self.C[0]) * self.C[0],
            (0, 0, 1): (self.X_C[(0, 1)] / self.C[1]) * (self.Y_C[(0, 1)] / self.C[1]) * self.C[1],
            (0, 1, 0): (self.X_C[(0, 0)] / self.C[0]) * (self.Y_C[(1, 0)] / self.C[0]) * self.C[0],
            (0, 1, 1): (self.X_C[(0, 1)] / self.C[1]) * (self.Y_C[(1, 1)] / self.C[1]) * self.C[1],
            (1, 0, 0): (self.X_C[(1, 0)] / self.C[0]) * (self.Y_C[(0, 0)] / self.C[0]) * self.C[0],
            (1, 0, 1): (self.X_C[(1, 1)] / self.C[1]) * (self.Y_C[(0, 1)] / self.C[1]) * self.C[1],
            (1, 1, 0): (self.X_C[(1, 0)] / self.C[0]) * (self.Y_C[(1, 0)] / self.C[0]) * self.C[0],
            (1, 1, 1): (self.X_C[(1, 1)] / self.C[1]) * (self.Y_C[(1, 1)] / self.C[1]) * self.C[1],
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x, y in X_Y.keys():
            if not np.isclose(X[x] * Y[y], X_Y[(x, y)]):  # there is a cell in the probability table that's not equal to
                # the probabilities product
                return True
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are independent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        # p(X=x, Y=y |c) = p(x|c) * p(Y|c)
        for x, y, c in X_Y_C.keys():  # using the third formula from the notebook
            if not np.isclose(X_Y_C[(x, y, c)] / C[c], (X_C[(x, c)] / C[c]) * (Y_C[(y, c)] / C[c])):
                return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    prob = (rate ** k) * (np.exp(-rate)) / np.math.factorial(k)  # using formula
    log_p = np.log(prob)
    return log_p


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))

    for i, rate in enumerate(rates):
        likelihood = 0

        for sample in samples:
            likelihood += poisson_log_pmf(sample, rate)  # sums the likelihoods of each rate

        likelihoods[i] = likelihood

    return likelihoods


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    rate = rates[np.argmax(likelihoods)]
    return rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    mean = np.mean(samples)
    return mean


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.p
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    p = (1 / np.sqrt(2 * np.pi * std ** 2)) * (np.exp(-(1 / 2) * ((x - mean) / std) ** 2))
    return p


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = dataset[dataset[:, -1] == class_value]
        self.mean = np.mean(self.class_dataset, axis=0)  # list of the mean of each feature
        self.std = np.std(self.class_dataset, axis=0)  # list of the std of each feature

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = len(self.class_dataset) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        for i in range(self.class_dataset.shape[1] - 1):  # go over all features
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        return likelihood * prior


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_pred = self.ccd0.get_instance_posterior(x)
        ccd1_pred = self.ccd1.get_instance_posterior(x)
        pred = 0 if ccd0_pred > ccd1_pred else 1
        return pred


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = 0
    for instance in test_set:
        pred = map_classifier.predict(instance)  # predicting the label
        if pred == instance[-1]:  # checking if our prediction was right
            acc += 1
    return acc / len(test_set)


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.s
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """

    pdf = None
    d = len(x) - 1
    pdf = (1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov))) * (np.exp(-0.5 * ((x[:-1] - mean).T @
                                                                                   np.linalg.inv(cov) @ (
                                                                                           x[:-1] - mean))))

    return pdf


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = dataset[dataset[:, -1] == class_value][:, :-1]
        self.mean = np.mean(self.class_dataset, axis=0)
        self.cov_matrix = np.cov(self.class_dataset, rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        prior = len(self.class_dataset) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """

        likelihood = multi_normal_pdf(x, self.mean, self.cov_matrix)
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        return likelihood * prior


class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """

        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_pred = self.ccd0.get_prior()
        ccd1_pred = self.ccd1.get_prior()
        pred = 0 if ccd0_pred > ccd1_pred else 1
        return pred


class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_pred = self.ccd0.get_instance_likelihood(x)
        ccd1_pred = self.ccd1.get_instance_likelihood(x)
        pred = 0 if ccd0_pred > ccd1_pred else 1
        return pred


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_dataset = dataset[dataset[:, -1] == class_value][:, :-1]

    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = None
        prior = len(self.class_dataset) / len(self.dataset)
        return prior

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
       """

        likelihood = 1
        # saves the uniq elements of each feature to compute v_j
        v_j_arr = [np.unique(self.class_dataset[:, i]) for i in range(self.class_dataset.shape[1])]
        n_i = self.class_dataset.shape[0]
        for j in range(len(x) - 1):  # go over the futures of X and compute the probability
            n_i_j = len(self.class_dataset[self.class_dataset[:, j] == x[j]])  # n_i_j from formula
            v_j = len(v_j_arr[j])
            curr_likelihood = (n_i_j + 1) / (n_i + v_j)
            likelihood *= EPSILLON if x[j] not in v_j_arr[j] else curr_likelihood  # we use epsilon instead of zero
        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ccd0_pred = self.ccd0.get_instance_posterior(x)
        ccd1_pred = self.ccd1.get_instance_posterior(x)
        pred = 0 if ccd0_pred > ccd1_pred else 1
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = 0
        for instance in test_set:
            pred = self.predict(instance)  # predicting the label
            if pred == instance[-1]:  # checking if our prediction was right
                acc += 1
        return acc / len(test_set)
