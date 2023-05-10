from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# load the dataset and call it "data"
data = pd.read_csv(r'S:\Everything\new\spam.csv')
# remove instances with missing values
data = data.dropna()
print("Dataset Loaded...")
# select target and feature variables. X = variables, Y = target
dX = data.iloc[:, :-1].values
dY = data[['Class']].values
# split into first 1000 as training and remaining as test data, (dX and dY for decision tree, rX an rY for random forest)
dX_train = dX[:1001, :]
dX_test = dX[1001:, :]
dY_train = dY[:1001]
dY_test = dY[1001:]
# convert rY_train to a 1-dimensional array using ravel()
dY_train = dY_train.ravel()


# FINDING THE BEST MAX DEPTH OF DECISION TREE
maxdepthpool = {'max_depth': range(1, 10)}
# create a test decision tree classifier object
tclf = DecisionTreeClassifier(random_state=27)
# use grid search to find the best max_depth
m_grid_search = GridSearchCV(tclf, maxdepthpool, cv=5)
m_grid_search.fit(dX, dY)
# print the best max_depth
print('Best max_depth:', m_grid_search.best_params_['max_depth'], "\n")


# FINDING THE BEST N_ESTIMATOR FOR RANDOM FOREST
testestimators = {'n_estimators': [100, 150, 200]}
# create a test random forest classifier object
trfc = RandomForestClassifier()
# use grid search to find the best n_estimators value
n_grid_search = GridSearchCV(estimator=trfc, param_grid=testestimators, cv=5)
n_grid_search.fit(dX, dY.ravel())
# print the best n_estimators
print('Best n_estimators:', n_grid_search.best_params_['n_estimators'], "\n")


# create RandomForest Classifier
rfc = RandomForestClassifier(n_estimators=n_grid_search.best_params_['n_estimators'], random_state=42)
# train the random forest classifier on the training set
rfc.fit(dX_train, dY_train)
# make predictions on the random forest testing set
rfc_pred = rfc.predict(dX_test)


# define base models for Voting Ensemble
models = list()
models.append(('dtc', DecisionTreeClassifier(max_depth=m_grid_search.best_params_['max_depth'], random_state=27)))
models.append(('gnb', GaussianNB()))
models.append(('lgr', LogisticRegression(max_iter=5000)))
# create ensemble classifier with hard voting
ensemble = VotingClassifier(estimators=models, voting='hard')
# train the voting model on the training set
ensemble.fit(dX_train, dY_train)
# make predictions on the voting model testing set
vmod_pred = ensemble.predict(dX_test)


# create ADABoost ensemble with Decision Tree Base Learner
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m_grid_search.best_params_['max_depth'], random_state=27))
# train the ADABoost with training set
ada.fit(dX_train, dY_train)
# make predictions on testing set with ADABoost
ada_pred = ada.predict(dX_test)


# ADJUSTING TRAINING SAMPLE SIZE
# 20% test size
x2_train, x2_test, y2_train, y2_test = train_test_split(dX, dY, test_size=0.20, random_state=27)
# 30% test size
x3_train, x3_test, y3_train, y3_test = train_test_split(dX, dY, test_size=0.30, random_state=27)
# 40% test size
x4_train, x4_test, y4_train, y4_test = train_test_split(dX, dY, test_size=0.40, random_state=27)
# 50% test size
x5_train, x5_test, y5_train, y5_test = train_test_split(dX, dY, test_size=0.50, random_state=27)


# TESTING DIFFERENT TRAINING SIZE ON VOTING ENSEMBLE
# create a new test ensemble classifier with hard voting
ensembleV2 = VotingClassifier(estimators=models, voting='hard')
# 20%
ensembleV2.fit(x2_train, y2_train.ravel())
vmod20_pred = ensembleV2.predict(x2_test)
print("Voting Model 20% Test Size Accuracy Score", accuracy_score(y2_test, vmod20_pred))
# 30%
ensembleV2.fit(x3_train, y3_train.ravel())
vmod30_pred = ensembleV2.predict(x3_test)
print("Voting Model 30% Test Size Accuracy Score", accuracy_score(y3_test, vmod30_pred))
# 40%
ensembleV2.fit(x4_train, y4_train.ravel())
vmod40_pred = ensembleV2.predict(x4_test)
print("Voting Model 40% Test Size Accuracy Score", accuracy_score(y4_test, vmod40_pred))
# 50%
ensembleV2.fit(x5_train, y5_train.ravel())
vmod50_pred = ensembleV2.predict(x5_test)
print("Voting Model 50% Test Size Accuracy Score", accuracy_score(y5_test, vmod50_pred), "\n")


# TESTING DIFFERENT TRAINING SIZE ON ADABOOST
adaV2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=m_grid_search.best_params_['max_depth'], random_state=27))
# 20%
adaV2.fit(x2_train, y2_train.ravel())
ada20_pred = ada.predict(x2_test)
print("ADABoost 20% Test Size Accuracy Score", accuracy_score(y2_test, ada20_pred))
# 30%
adaV2.fit(x3_train, y3_train.ravel())
ada30_pred = ada.predict(x3_test)
print("ADABoost 30% Test Size Accuracy Score", accuracy_score(y3_test, ada30_pred))
# 40%
adaV2.fit(x4_train, y4_train.ravel())
ada40_pred = ada.predict(x4_test)
print("ADABoost 40% Test Size Accuracy Score", accuracy_score(y4_test, ada40_pred))
# 50%
adaV2.fit(x5_train, y5_train.ravel())
ada50_pred = ada.predict(x5_test)
print("ADABoost 50% Test Size Accuracy Score", accuracy_score(y5_test, ada50_pred), "\n")


# Accuracy Report of each model
print("Random Forest Accuracy Score", accuracy_score(dY_test, rfc_pred), "\n")
print("Voting Model Accuracy Score", accuracy_score(dY_test, vmod_pred), "\n")
print("AdaBoost Accuracy Score", accuracy_score(dY_test, ada_pred), "\n")
# Classification Report of each model
print("Random Forest Classification Report: \n", classification_report(dY_test, rfc_pred))
print("Voting Model Classification Report \n", classification_report(dY_test, vmod_pred))
print("AdaBoost Classification Report \n", classification_report(dY_test, ada_pred))
# Confusion Matrix of each model
print("Random Forest Confusion Matrix: \n", confusion_matrix(dY_test, rfc_pred))
print("Voting Model Confusion Matrix: \n", confusion_matrix(dY_test, vmod_pred))
print("AdaBoost Confusion Matrix: \n", confusion_matrix(dY_test, ada_pred))
