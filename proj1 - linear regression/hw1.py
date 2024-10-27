###### Your ID ######
# ID1: 315678185
# ID2: 316591536
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """

    mean_x = np.mean(X, axis=0)
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    mean_y = np.mean(y)
    min_y = np.min(y)
    max_y = np.max(y)
    X = (X - mean_x) / (max_x - min_x)
    y = (y - mean_y) / (max_y - min_y)

    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    oneColum = np.ones(X.shape[0])
    X = np.c_[oneColum, X]  # add colum of ones to the X matrix
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    m = X.shape[0]
    h = np.dot(X, theta)  # dot product of X in theta (saves t0*1+t1*xi)
    J = np.sum((h - y) ** 2) / (2 * m)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    m = X.shape[0]
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        h = X.dot(theta)  # dot product of X in theta (saves t0*1+t1*xi)
        theta = theta - alpha * (1 / m) * np.dot(X.T, h - y)  # the instances are the rows of X that's why you need X.T
        newJ = compute_cost(X, y, theta)
        J_history.append(newJ)
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    X_transpose = X.T
    XT_dot_X = np.dot(X_transpose, X)
    inverse = np.linalg.inv(XT_dot_X)
    pinvX = np.dot(inverse, X_transpose)
    pinv_theta = np.dot(pinvX, y)  # compute pinv according to formula

    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    m = X.shape[0]
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    for i in range(num_iters):
        h = np.dot(X, theta)  # dot product of X in theta (saves t0*1+t1*xi)
        theta = theta - (alpha * (1 / m)) * np.dot(X.T,
                                                   (h - y))  # the instances are the rows of X that's why you need X.T
        newJ = compute_cost(X, y, theta)
        J_history.append(newJ)
        if (i > 0 and J_history[-2] - J_history[-1] < 1e-8):
            break
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    np.random.seed(42)
    start_theta = np.random.random(X_train.shape[1])  # start in each gradiant descent with the same random theta
    for alpha in alphas:
        theta = efficient_gradient_descent(X_train, y_train, start_theta, alpha, iterations)[0]  # find the theta using
        # gardiant descent
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)  # compute the cost function using the validation data
        # and save with the alpha
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    i = 0
    while len(selected_features) < 5:  # stop when reaching the 5 best features
        best = 0.5  # saves the best feature index
        bestCost = float('inf')
        rnd_theta = np.random.random(size=i + 2)  # creates a random theta vector
        for j in range(X_train.shape[1]):  # go over all features
            if (j not in selected_features):  # try every feature that has not already taken
                selected_features.append(j)
                currentXTrain = X_train[:, selected_features]  # take all rows of the selected features
                currentXTrain = apply_bias_trick(currentXTrain)  # add colum of ones
                currentXVal = X_val[:, selected_features]
                currentXVal = apply_bias_trick(currentXVal)
                theta, _ = efficient_gradient_descent(currentXTrain, y_train, rnd_theta, best_alpha,
                                                      iterations)  # gradiant descent based on the current features
                cost = compute_cost(currentXVal, y_val, theta)  # compute the cost
                if cost < bestCost:  # if current feature improves the cost save it
                    bestCost = cost
                    best = j
                selected_features.pop()
        selected_features.append(best)  # append the best feature found to the best feature array
        i = i + 1  # index for creating the ones vector (the initialized theta) in the correct size
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    new_collums = {}  # dictionary to hold all the new columns and add them together

    for i, col1 in enumerate(df.columns):
        for col2 in df_poly.columns[i:]:  # connect all colum with higher indexes to current colum
            if (col1 == col2):
                colName = f'{col1}^2'
            else:
                colName = f'{col1}*{col2}'
            new_collums[colName] = df_poly[col1] * df_poly[col2]
    newDf = pd.DataFrame(new_collums)  # turn the dictionary to a pandas data frame
    df_poly = pd.concat([df_poly, newDf], axis=1)  # concat the columns
    return df_poly
