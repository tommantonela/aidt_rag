
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

L2RFILENAME = './models/gbrank.pkl'

### GRADIENT BOOST ###
### AKA GBRank ###
np.random.seed(0)

print("Loading datasets ...")
dataset_train = pd.read_csv("./data/dataSetPropuestaCSVTrainEval.txt")
dataset_train = dataset_train.fillna(0)
dataset_test = pd.read_csv("./data/dataSetPropuestaCSVTest.txt")
dataset_test = dataset_test.fillna(0)

scaler = MinMaxScaler()
X_train = dataset_train.drop(['label'], axis=1)
X_train = scaler.fit_transform(X_train)
y_train = dataset_train['label']

scaler = MinMaxScaler()
X_test = dataset_test.drop(['label'], axis=1)
X_test = scaler.fit_transform(X_test)
y_test = dataset_test['label']
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("Training learning2rank model (GradientBoosting) ...")
start_time = time.time()

gb = GradientBoostingClassifier(learning_rate=0.005, min_samples_split=50, min_samples_leaf=10, 
                                max_depth=50, max_features='log2', subsample=0.8, random_state=200)
param_test1 = {
    'n_estimators':[2375],
    'learning_rate':[0.004], 
    'max_depth':[50],
    'min_samples_split':[50], 
    'min_samples_leaf':[10]
    }
clf = GridSearchCV(estimator=gb, param_grid=param_test1, scoring='roc_auc',n_jobs=4,cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found:", clf.best_params_)
print("Best score:", clf.best_score_)
print()
    
y_true, y_pred = y_train, clf.predict(X_train)
print("Training stats:")
print(classification_report(y_true, y_pred))
#print("Training Accuracy - KFold:", accuracy_score(y_true, y_pred))
print()
y_true, y_pred = y_test, clf.predict(X_test)
print("Test stats:")
print(classification_report(y_true, y_pred))
#print("Test Accuracy - KFold:", accuracy_score(y_true, y_pred))
top_features = 15
print("Feature Importances:", top_features)
fi_dict = { 'features': dataset_train.drop(['label'], axis=1).columns,
           'importance': clf.best_estimator_.feature_importances_ }
fi_df = pd.DataFrame(fi_dict).sort_values(by='importance', ascending=False)
print(fi_df.head(top_features))
print()
print("--- %s seconds ---" % (time.time() - start_time))
print()

# Saving the best model for later use
pickle.dump(clf.best_estimator_, open(L2RFILENAME, 'wb'))

gb = pickle.load(open(L2RFILENAME, 'rb'))
y_true, y_pred = y_test, gb.predict(X_test)
print("Test stats (chosen model):")
print(classification_report(y_true, y_pred))
print()