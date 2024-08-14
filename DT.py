import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import auc, roc_curve, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
#C5.0
from sklearn import tree
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
PF_train_Y = PF_train[['labels']] # y is the output, the string is the corresponding column name
PF_train_X = PF_train
PF_train_X.drop(['labels'], axis=1, inplace=True) # X is the feature, need to remove the output
# Read test set
PF_test = pd.read_csv("path to test set sdf file")
PF_test_Y = PF_test[['labels']]
y_test = PF_test_Y.values
PF_test_X = PF_test
PF_test_X.drop(['labels'], axis=1, inplace=True)

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

"""hyperopt"""
# Below is the code for optimizing the model's hyperparameters using hyperopt
# Optimizing function for C
def C_loss_function(hyperparams):
  crit, md, mss, msl, mwf, mf, cw = hyperparams['crit'], hyperparams['md'], hyperparams['mss'], hyperparams['msl'], hyperparams['mwf'],  hyperparams['mf'], hyperparams['cw']
  clf = tree.DecisionTreeClassifier(
                                criterion=crit,
                                max_depth=md,
                                min_samples_split=mss,
                                min_samples_leaf=msl,
                                min_weight_fraction_leaf=mwf,
                                max_features=mf,
                                class_weight=cw)

  CV_train_score = 0
  CV_valid_score = 0
  for train, valid in kf.split(X_train_normalized):
    X_train, X_valid = X_train_normalized[train], X_train_normalized[valid]
    y_train, y_valid = PF_train_Y.loc[train].values, PF_train_Y.loc[valid].values
    clf.fit(X_train, y_train)

    train_y_pred = clf.predict(X_train)
    train_y_pred_yc = train_y_pred.ravel().astype(bool)
    train_true = y_train.ravel().astype(bool)
    train_score = accuracy_score(train_true, train_y_pred_yc)
    CV_train_score += train_score

    valid_y_pred = clf.predict(X_valid)
    valid_y_pred_yc = valid_y_pred.ravel().astype(bool)
    valid_true = y_valid.ravel().astype(bool)
    valid_score = accuracy_score(valid_true, valid_y_pred_yc)
    CV_valid_score += valid_score

  CV_valid_score = CV_valid_score/5
  CV_train_score = CV_train_score/5
  print(str(hyperparams))
  print(CV_train_score)
  print(CV_valid_score)
  # Because decision tree generally performs poorly on valid score, the highest valid score may have a very low train score, meaning the model was not well trained, we need to ensure train score > 0.95
  if CV_train_score > 0.95:
    return 1-CV_valid_score
  else:
      return 1

# Define the range of parameters to be optimized, below are examples, you can define your own
criterion = ['gini', 'entropy']
min_weight_fraction_leaf = [0,0.1,0.2,0.3]
max_features = [0.1,0.3,0.5,0.7,0.9]
class_weight = ['balanced', None]
# The range of integer parameters does not need to be listed as above, you can use scope.int(hp.quniform(range, q=multiple)) like the range for criterion
hyperparams = {
                'crit': hp.choice('crit', criterion),
                'md': scope.int(hp.quniform('md', 10, 50, q=1)),
                'mss': scope.int(hp.quniform('mss', 2, 6, q=1)),
                'msl': scope.int(hp.quniform('msl', 1, 6, q=1)),
                'mwf': hp.choice('mwf', min_weight_fraction_leaf),
                'mf': hp.choice('mf', max_features),
                'cw': hp.choice('cw', class_weight)
              }

trials = Trials()
best = fmin(
              fn = C_loss_function,
              space = hyperparams,
              algo = tpe.suggest,
              max_evals = 10000, # Define the number of parameter combinations to try, you can set it yourself
              return_argmin = True,
              trials = trials)

print("best hyperparameter:")
print(best)

# Build the model using the best hyperparameters and evaluate using the test set
clf = tree.DecisionTreeClassifier(
                            criterion=criterion[best['crit']],
                            max_depth=int(best['md']),
                            min_samples_split=int(best['mss']),
                            min_samples_leaf=int(best['msl']),
                            min_weight_fraction_leaf=min_weight_fraction_leaf[best['mwf']],
                            max_features=max_features[best['mf']],
                            class_weight=class_weight[best['cw']])
CV_train_score = 0
CV_valid_score = 0
CV_train_sen = 0
CV_valid_sen = 0
CV_train_spe = 0
CV_valid_spe = 0
CV_train_MCC = 0
CV_valid_MCC = 0
for train, valid in kf.split(X_train_normalized):
    X_train, X_valid = X_train_normalized[train], X_train_normalized[valid]
    y_train, y_valid = PF_train_Y.loc[train].values, PF_train_Y.loc[valid].values
    clf.fit(X_train, y_train)

    train_y_pred = clf.predict(X_train)
    train_y_pred_yc = train_y_pred.ravel().astype(bool)
    train_true = y_train.ravel().astype(bool)
    train_score = accuracy_score(train_true, train_y_pred_yc)
    CV_train_score += train_score

    valid_y_pred = clf.predict(X_valid)
    valid_y_pred_yc = valid_y_pred.ravel().astype(bool)
    valid_true = y_valid.ravel().astype(bool)
    valid_score = accuracy_score(valid_true, valid_y_pred_yc)
    CV_valid_score += valid_score
    # sen_spe
    train_cf = confusion_matrix(y_train, train_y_pred_yc.astype(int))
    valid_cf = confusion_matrix(y_valid, valid_y_pred_yc.astype(int))
    CV_train_sen += train_cf[1, 1] / (train_cf[1, 1] + train_cf[1, 0])
    CV_valid_sen += valid_cf[1, 1] / (valid_cf[1, 1] + valid_cf[1, 0])
    CV_train_spe += train_cf[0, 0] / (train_cf[0, 0] + train_cf[0, 1])
    CV_valid_spe += valid_cf[0, 0] / (valid_cf[0, 0] + valid_cf[0, 1])
    # mcc
    CV_train_MCC += matthews_corrcoef(y_train, train_y_pred_yc.astype(int))
    CV_valid_MCC += matthews_corrcoef(y_valid, valid_y_pred_yc.astype(int))

# testset_score
clf.fit(X_train_normalized, PF_train_Y.values)

train_y_pred = clf.predict(X_train_normalized)
train_y_pred_yc = train_y_pred.ravel().astype(bool)
train_true = PF_train_Y.values.ravel().astype(bool)
train_score = accuracy_score(train_true, train_y_pred_yc)

test_y_pred = clf.predict(X_test_normalized)
test_y_pred_yc = test_y_pred.ravel().astype(bool)
test_true = y_test.ravel().astype(bool)
test_score = accuracy_score(test_true, test_y_pred_yc)
# sen_spe
train_cf = confusion_matrix(PF_train_Y.values, train_y_pred_yc.astype(int))
test_cf = confusion_matrix(y_test, test_y_pred_yc.astype(int))

# Pack hyperparameters and prediction scores
Result = {
            'CV_train_MCC': [],
            'CV_train_score': [],
            'CV_train_sen': [],
            'CV_train_spe': [],
            'CV_valid_MCC': [],
            'CV_valid_score': [],
            'CV_valid_sen': [],
            'CV_valid_spe': [],
            'criterion': [],
            'max_depth': [],
            'min_samples_split': [],
            'min_samples_leaf': [],
            'min_weight_fraction_leaf': [],
            'max_features': [],
            'class_weight': [],
            'train_MCC': [],
            'train_score': [],
            'train_sen': [],
            'train_spe': [],
            'test_MCC': [],
            'test_score': [],
            'test_sen': [],
            'test_spe': [],
            'train_cf_pro': [],
            'test_cf_pro': []
            }

Result['criterion'].append(criterion[best['crit']])
Result['max_depth'].append(int(best['md']))
Result['min_samples_split'].append(int(best['mss']))
Result['min_samples_leaf'].append(int(best['msl']))
Result['min_weight_fraction_leaf'].append(min_weight_fraction_leaf[best['mwf']])
Result['max_features'].append(max_features[best['mf']])
Result['class_weight'].append(class_weight[best['cw']])
Result['CV_train_MCC'].append(CV_train_MCC / 5),
Result['CV_valid_MCC'].append(CV_valid_MCC / 5),
Result['CV_train_score'].append(CV_train_score / 5),
Result['CV_valid_score'].append(CV_valid_score / 5),
Result['CV_train_sen'].append(CV_train_sen / 5),
Result['CV_valid_sen'].append(CV_valid_sen / 5),
Result['CV_train_spe'].append(CV_train_spe / 5),
Result['CV_valid_spe'].append(CV_valid_spe / 5),
Result['train_MCC'].append(matthews_corrcoef(PF_train_Y.values, train_y_pred_yc.astype(int))),
Result['train_score'].append(train_score),
Result['train_sen'].append(train_cf[1, 1] / (train_cf[1, 1] + train_cf[1, 0])),
Result['train_spe'].append(train_cf[0, 0] / (train_cf[0, 0] + train_cf[0, 1])),
Result['test_MCC'].append(matthews_corrcoef(y_test, test_y_pred_yc.astype(int))),
Result['test_score'].append(test_score),
Result['test_sen'].append(test_cf[1, 1] / (test_cf[1, 1] + test_cf[1, 0])),
Result['test_spe'].append(test_cf[0, 0] / (test_cf[0, 0] + test_cf[0, 1])),
Result['train_cf_pro'].append(train_cf),
Result['test_cf_pro'].append(test_cf)

print(Result)

Result = pd.DataFrame.from_dict(Result)
Result.to_csv('path to output file')
