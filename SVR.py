import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import r2_score # The example score uses R squared, but you can use other metrics as well
from sklearn.model_selection import KFold
from sklearn import preprocessing
# SVM
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#hyperopt
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope, stochastic
#Model saving
from joblib import dump, load

def seed_all():
  np.random.seed(123)

seed_all()

# Read train set
PF_train = pd.read_csv("path to train set sdf file")
PF_train_Y = PF_train[['ground truth']] # y is the output, the string is the corresponding column name
PF_train_X = PF_train
PF_train_X.drop(['ground truth'], axis=1, inplace=True) # X is the feature, need to remove the output
# Read test set
PF_test = pd.read_csv("path to test set sdf file")
PF_test_Y = PF_test[['ground truth']]
y_test = PF_test_Y.values
PF_test_X = PF_test
PF_test_X.drop(['ground truth'], axis=1, inplace=True)

# Next, we need to do some preprocessing to remove descriptors that cannot be input into the model
# First, merge the X of Train set and test set
PF = pd.concat([PF_train_X, PF_test_X])
PF = PF.reset_index(drop=True)
# Remove columns with infinity values
np.all(np.isfinite(PF.shape)) # whether contain infinity values
PF = PF.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
# Then split into train and test set
X_train = PF.loc[:(PF_train.shape[0]-1),:]
X_test = PF.loc[(PF_train.shape[0]):,:]
# Standardize the features, which will be used later
X_train_normalized = preprocessing.scale(X_train, with_mean=0, with_std=1)
X_test_normalized = preprocessing.scale(X_test, with_mean=0, with_std=1)

# Divide the train set into five parts for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle = True, random_state = 123)

# The following code uses hyperopt to optimize the model's hyperparameters; if not needed or if you have another optimization method, you can skip it
# Define the loss function to optimize
def SVM_loss_function(hyperparams):
    # Fill in the parameters to be optimized, the following is just an example
    c, ep, kn, gm, c0, tol, cs, dfs, cw = hyperparams['C'], hyperparams['epsilon'], hyperparams['kernel'], hyperparams['gamma'], hyperparams[
        'coef0'], hyperparams['tol'], hyperparams['cache_size'], hyperparams['decision_function_shape'], hyperparams[
                                          'class_weight']
    clf = make_pipeline(StandardScaler(), SVR(C=c,
                                              epsilon=ep,
                                              kernel=kn,
                                              gamma=gm,
                                              coef0=c0,
                                              tol=tol,
                                              cache_size=cs,
                                              decision_function_shape=dfs,
                                              class_weight=cw))
    CV_train_score = 0
    CV_valid_score = 0
    # Divide the train set into five parts for 5-fold cross-validation
    for train, valid in kf.split(X_train_normalized): # Here, the standardized dataset is used, if standardization is not needed, use X_train
        # Split X and y into train and valid set
        X_train, X_valid = X_train_normalized[train], X_train_normalized[valid]
        y_train, y_valid = PF_train_Y.loc[train].values, PF_train_Y.loc[valid].values
        clf.fit(X_train, y_train) # Train the model

        train_y_pred = clf.predict(X_train) # Predict the CV's train set
        train_score = r2_score(train_y_pred, y_train)
        CV_train_score += train_score

        valid_y_pred = clf.predict(X_valid) # Predict the CV's valid set
        valid_score = r2_score(valid_y_pred, y_valid)
        CV_valid_score += valid_score
    # Average the scores from the five folds
    CV_valid_score = CV_valid_score / 5
    CV_train_score = CV_train_score / 5

    print(str(hyperparams))
    print(CV_train_score)
    print(CV_valid_score)
    return 1 - CV_valid_score # Here, I'm using the valid score to calculate the loss. You can define the loss yourself; the goal of optimization is to minimize the loss value.

# Define the range of parameters to be optimized, below are examples, you can define your own
epsilon = [0.1,0.2,0.3,0.4,0.5]
kernel = ['linear', 'rbf','poly', 'sigmoid']
gamma = ['auto','scale']
tol = [0.001,0.002,0.004,0.006,0.008,0.01,0.02,0.04,0.06,0.08,0.1]
# The range of integer parameters does not need to be listed as above, you can use scope.int(hp.quniform(range, q=multiple)) like the range for C
hyperparams = {
                'C': scope.int(hp.quniform('C', 1, 10, q=1)),
                'epsilon': hp.choice('epsilon', epsilon),
                'kernel': hp.choice('kernel', kernel),
                'gamma': hp.choice('gamma', gamma),
                'coef0': scope.int(hp.quniform('coef0', 0, 5, q=1)),
                'tol': hp.choice('tol', tol),
                'cache_size': scope.int(hp.quniform('cache_size', 150, 250, q=1)),
              }

trials = Trials()
best = fmin(
              fn = SVM_loss_function,
              space = hyperparams,
              algo = tpe.suggest,
              max_evals = 10000, # Define the number of parameter combinations to try, you can set it yourself
              return_argmin = True,
              trials = trials)

print("best hyperparameter:")
print(best)

# Build the model using the best hyperparameters and evaluate using the test set
clf = make_pipeline(StandardScaler(), SVR(C=int(best['C']),
                                        epsilon= epsilon[best['epsilon']], # If the range is defined by a list, the value recorded in best will be the index of the list
                                        kernel= kernel[best['kernel']],
                                        gamma= gamma[best['gamma']],
                                        coef0= int(best['coef0']),
                                        tol= tol[best['tol']],
                                        cache_size = int(best['cache_size'])))
CV_train_score = 0
CV_valid_score = 0
for train, valid in kf.split(X_train_normalized):
  X_train, X_valid = X_train_normalized[train], X_train_normalized[valid]
  y_train, y_valid = PF_train_Y.loc[train].values, PF_train_Y.loc[valid].values
  clf.fit(X_train, y_train)

  train_y_pred = clf.predict(X_train)
  train_score = r2_score(train_y_pred, y_train)
  CV_train_score += train_score

  valid_y_pred = clf.predict(X_valid)
  valid_score = r2_score(valid_y_pred, y_valid)
  CV_valid_score += valid_score

CV_valid_score = CV_valid_score / 5
CV_train_score = CV_train_score / 5

# testset_score
clf.fit(X_train_normalized, PF_train_Y.values)

train_y_pred = clf.predict(X_train_normalized)
train_r2score = r2_score(train_y_pred, PF_train_Y.values)

test_y_pred = clf.predict(X_test_normalized)
test_r2score = r2_score(y_test, test_y_pred)

# Pack hyperparameters and prediction scores
Result = {
            'C': [],
            'epsilon': [],
            'kernel': [],
            'gamma': [],
            'coef0': [],
            'tol': [],
            'cache_size': [],
            'CV_train_r2score': [],
            'CV_valid_r2score':[],
            'train_r2score': [],
            'test_r2score': [],
        }
Result['C'].append(best['C'])
Result['epsilon'].append(epsilon[best['epsilon']])
Result['kernel'].append(kernel[best['kernel']])
Result['gamma'].append(gamma[best['gamma']])
Result['coef0'].append(best['coef0'])
Result['tol'].append(tol[best['tol']])
Result['cache_size'].append(best['cache_size'])
Result['CV_train_r2score'].append(CV_train_score)
Result['CV_valid_r2score'].append(CV_valid_score)
Result['train_r2score'].append(train_r2score)
Result['test_r2score'].append(test_r2score)

Result = pd.DataFrame.from_dict(Result)
Result.to_csv('path to output file')

# Save model
#dump(clf,'filename.joblib')
# Load model
#clf = load('filename.joblib')
