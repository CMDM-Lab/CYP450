import numpy as np
import pandas as pd
import tensorflow as tf
import deepchem as dc
import os
import tempfile
import sys
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef
target_names = ['Neg', 'Pos']

# Use ConvMolFeaturizer to convert SMILES into input graph features
graph_featurizer = dc.feat.graph_features.ConvMolFeaturizer()
# Define csvloader, task corresponds to the column name of the label in the data, smiles_field corresponds to the column name of SMILES
loader = dc.data.data_loader.CSVLoader(tasks=['labels'], smiles_field="SMILES", featurizer=graph_featurizer)
# Load training and testing sets, the input is the path of the file
trainset = loader.featurize('path-to-CYP-trainingset.csv')
testset = loader.featurize('path-to-CYP-testingset.csv')

# Use the RandomSplitter from deepchem's splitter package to split the trainset into five parts, fold_datas is a list containing five sets of train and valid sets
splitter = dc.splits.splitters.RandomSplitter()
fold_datas = splitter.k_fold_split(trainset, 5)

# Hyperparameter optimization: Use the hyperopt package, first define the loss function, which includes the entire model's parameter definition and training, using the return value to select the parameter combination
def GCN_loss_function(hyperparams):
    gcl, dls, lr, do, bs, ep = hyperparams['gcl'], hyperparams['dls'], hyperparams['lr'], hyperparams['do'], hyperparams['bs'], hyperparams['ep']
    # deepchem2.0: dc.models.tensorgraph.models.graph_models.GraphConvTensorGraph
    model = dc.models.graph_models.GraphConvModel(n_tasks=1,
                                                  graph_conv_layers=[gcl, gcl],
                                                  dense_layer_size=dls,
                                                  learning_rate=lr,
                                                  dropout=do,
                                                  batch_size=bs,
                                                  mode='classification',
                                                  n_classes=2,
                                                  model_dir='path-to-CYP-project')

    CV_train_score = 0
    CV_valid_score = 0
    # loss = 0
    for train_set, val_set in fold_datas:
       model.fit(train_set, nb_epoch=ep, deterministic=True)
       # loss = model.fit(train_set, nb_epoch=ep, deterministic=True)
       # loss += loss
       train_predictions = model.predict(train_set)
       train_predictions_y = train_predictions[:len(train_set.y)]
       train_predictions_yc = train_predictions_y.ravel()[1::2] > 0.5
       train_true = train_set.y.ravel().astype(bool)
       train_score = accuracy_score(train_true, train_predictions_yc)
       CV_train_score += train_score

       valid_predictions = model.predict(val_set)
       valid_predictions_y = valid_predictions[:len(val_set.y)]
       valid_predictions_yc = valid_predictions_y.ravel()[1::2] > 0.5
       valid_true = val_set.y.ravel().astype(bool)
       valid_score = accuracy_score(valid_true, valid_predictions_yc)
       CV_valid_score += valid_score
    # Since I want to use the accuracy rate of the valid set to select parameters, if you want to use the model's loss for selection, you can directly return CV_loss
    CV_valid_score = CV_valid_score / 5
    CV_train_score = CV_train_score / 5
    # CV_loss = loss/5
    print(str(hyperparams))
    print(CV_train_score)
    print(CV_valid_score)
    return 1 - CV_valid_score

# Define the selection range of parameters and the number of trials in the fmin function max_evals
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope, stochastic
learn = [0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.008, 0.01]
drop = [0, 0.2, 0.3, 0.4]
hyperparams = {
   'gcl': scope.int(hp.quniform('gcl', 32, 128, q=1)), # Random integer from 32 to 128
   'dls': scope.int(hp.quniform('dls', 16, 256, q=1)),
   'lr': hp.choice('lr', learn), # Random value from the list
   'do': hp.choice('do', drop),
   'bs': scope.int(hp.quniform('bs', 32, 128, q=1)),
   'ep': scope.int(hp.quniform('ep', 0, 150, q=1))
}
trials = Trials() # Record all results of each parameter combination
best = fmin(
   fn=GCN_loss_function,
   space=hyperparams,
   algo=tpe.suggest,
   max_evals=100, # Number of parameter combinations
   return_argmin=False,
   trials=trials)
print("best:")
print(best)
print("bestpara:")
# Record the best parameter combinations
bestpara = {
   'gcl': [],
   'dls': [],
   'lr': [],
   'do': [],
   'bs': [],
   'ep': []
}
best_trials = sorted(trials.trials, key=lambda x: x['result']['loss'], reverse=False) # Sort all parameter combinations in trials by the result of GCN_loss_function in ascending order
for trial in best_trials[:10]: # Extract the top 10 parameters into bestpara
   print(trial['result'])
   bestpara['gcl'].append(trial['misc']['vals']['gcl'][0])
   bestpara['dls'].append(trial['misc']['vals']['dls'][0])
   bestpara['lr'].append(learn[(trial['misc']['vals']['lr'][0])]) # If using a list to select parameters, the index will be recorded
   bestpara['do'].append(drop[(trial['misc']['vals']['do'][0])])
   bestpara['bs'].append(trial['misc']['vals']['bs'][0])
   bestpara['ep'].append(trial['misc']['vals']['ep'][0])
print(bestpara)
# Add all the performance metrics you want to obtain
bestpara['CV_train_score'] = [] # CV training accuracy
bestpara['CV_valid_score'] = []
bestpara['CV_train_sen'] = [] # CV training sensitivity
bestpara['CV_valid_sen'] = []
bestpara['CV_train_spe'] = [] # CV training specificity
bestpara['CV_valid_spe'] = []
bestpara['CV_train_MCC'] = [] # CV training Matthew correlation coefficient
bestpara['CV_valid_MCC'] = []
bestpara['train_score'] = [] # Accuracy of the entire training set
bestpara['train_sen'] = []
bestpara['train_spe'] = []
bestpara['train_MCC'] = []
bestpara['test_score'] = [] # Accuracy of the testing set
bestpara['test_sen'] = []
bestpara['test_spe'] = []
bestpara['test_MCC'] = []
bestpara['train_cf_pro'] = [] # confusion_matrix: you can see the number of TP, FP, FN, TN
bestpara['test_cf_pro'] = []

# Run the CV and train_test performance of the top ten parameter combinations
for i in range(len(bestpara['gcl'])):
   gcl, dls, lr, do, bs, ep = int(bestpara['gcl'][i]), int(bestpara['dls'][i]), bestpara['lr'][i], bestpara['do'][i], \
                                  int(bestpara['bs'][i]), int(bestpara['ep'][i])
   model = dc.models.graph_models.GraphConvModel(n_tasks=1,
                                                 graph_conv_layers=[gcl, gcl],
                                                 dense_layer_size=dls,
                                                 learning_rate=lr,
                                                 dropout=do,
                                                 batch_size=bs,
                                                 mode='classification',
                                                 n_classes=2,
                                                 model_dir='/GCN_model/CV')
   CV_train_score = 0
   CV_valid_score = 0
   CV_train_sen = 0
   CV_valid_sen = 0
   CV_train_spe = 0
   CV_valid_spe = 0
   CV_train_MCC = 0
   CV_valid_MCC = 0
   for train_set, val_set in fold_datas:
       model.fit(train_set, nb_epoch=ep, deterministic=True)

       # acc
       train_predictions = model.predict(train_set)
       train_predictions_y = train_predictions[:len(train_set.y)]
       train_predictions_yc = train_predictions_y.ravel()[1::2] > 0.5
       train_true = train_set.y.ravel().astype(bool)
       train_score = accuracy_score(train_true, train_predictions_yc)
       CV_train_score += train_score

       valid_predictions = model.predict(val_set)
       valid_predictions_y = valid_predictions[:len(val_set.y)]
       valid_predictions_yc = valid_predictions_y.ravel()[1::2] > 0.5
       valid_true = val_set.y.ravel().astype(bool)
       valid_score = accuracy_score(valid_true, valid_predictions_yc)
       CV_valid_score += valid_score

       # sen_spe
       train_cf = confusion_matrix(train_set.y, train_predictions_yc.astype(int))
       valid_cf = confusion_matrix(val_set.y, valid_predictions_yc.astype(int))
       CV_train_sen += train_cf[1, 1] / (train_cf[1, 1] + train_cf[1, 0])
       CV_valid_sen += valid_cf[1, 1] / (valid_cf[1, 1] + valid_cf[1, 0])
       CV_train_spe += train_cf[0, 0] / (train_cf[0, 0] + train_cf[0, 1])
       CV_valid_spe += valid_cf[0, 0] / (valid_cf[0, 0] + valid_cf[0, 1])

       # mcc
       CV_train_MCC += matthews_corrcoef(train_set.y, train_predictions_yc.astype(int))
       CV_valid_MCC += matthews_corrcoef(val_set.y, valid_predictions_yc.astype(int))

   CV_valid_score = CV_valid_score / 5
   CV_train_score = CV_train_score / 5
   CV_train_sen = CV_train_sen / 5
   CV_valid_sen = CV_valid_sen / 5
   CV_train_spe = CV_train_spe / 5
   CV_valid_spe = CV_valid_spe / 5
   CV_train_MCC = CV_train_MCC / 5
   CV_valid_MCC = CV_valid_MCC / 5
   bestpara['CV_train_score'].append(CV_train_score / 5)
   bestpara['CV_valid_score'].append(CV_valid_score / 5)
   bestpara['CV_train_sen'].append(CV_train_sen / 5)
   bestpara['CV_valid_sen'].append(CV_valid_sen / 5)
   bestpara['CV_train_spe'].append(CV_train_spe / 5)
   bestpara['CV_valid_spe'].append(CV_valid_spe / 5)
   bestpara['CV_train_MCC'].append(CV_train_MCC / 5)
   bestpara['CV_valid_MCC'].append(CV_valid_MCC / 5)

   # After running CV, predict the complete training set and testing set
   model.fit(trainset, nb_epoch=ep, deterministic=True)

   train_predictions = model.predict(trainset)
   train_predictions_y = train_predictions[:len(trainset.y)]
   train_predictions_yc = train_predictions_y.ravel()[1::2] > 0.5
   train_true = trainset.y.ravel().astype(bool)
   train_cf = confusion_matrix(trainset.y, train_predictions_yc.astype(int))

   bestpara['train_score'].append(accuracy_score(train_true, train_predictions_yc))

   test_predictions = model.predict(testset)
   test_predictions_y = test_predictions[:len(testset.y)]
   test_predictions_yc = test_predictions_y.ravel()[1::2] > 0.5
   test_true = testset.y.ravel().astype(bool)
   test_cf = confusion_matrix(testset.y, test_predictions_yc.astype(int))

   bestpara['test_score'].append(accuracy_score(test_true, test_predictions_yc))

   # sen_spe
   bestpara['train_sen'].append(train_cf[1, 1] / (train_cf[1, 1] + train_cf[1, 0]))
   bestpara['train_spe'].append(train_cf[0, 0] / (train_cf[0, 0] + train_cf[0, 1]))
   bestpara['test_sen'].append(test_cf[1, 1] / (test_cf[1, 1] + test_cf[1, 0]))
   bestpara['test_spe'].append(test_cf[0, 0] / (test_cf[0, 0] + test_cf[0, 1]))

   # mcc
   bestpara['train_MCC'].append(matthews_corrcoef(trainset.y, train_predictions_yc.astype(int)))
   bestpara['test_MCC'].append(matthews_corrcoef(testset.y, test_predictions_yc.astype(int)))

   bestpara['train_cf_pro'].append(train_cf)
   bestpara['test_cf_pro'].append(test_cf)

CYP1A2_DC20_all_result = pd.DataFrame(data=bestpara)
CYP1A2_DC20_all_result.to_csv('/CYP1A2_CV_hyperpara.csv') # Output the results as a csv file
