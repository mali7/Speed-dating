
import sys
import numpy as np
import os.path
import os
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import sklearn.metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors, datasets
from sklearn.metrics import mean_squared_error
import warnings
from sklearn import preprocessing
# warnings.filterwarnings("ignore", ConvergenceWarning)

def file_len(fname):
    with open(fname) as f:
        for t, l in enumerate(f):
            pass
    return t + 1
	
is_after_index = 2
ratings_start_point = 114


print('-------Based on SHORE+OpenFace features 1st minute---------')
f = open ('S1ALL.csv','r')
n_line = file_len('S1ALL.csv')
features = []
ratings = []
SA_confidance_index = 4
top_row = f.readline().split(',')
# print(top_row[204:209])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' :
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[4:ratings_start_point-2]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings) 

features = preprocessing.scale(features)
feature_names = []
feature_names.append([j for j in top_row[4:ratings_start_point-2]])
feature_names = np.array(feature_names)

print(' ','\t',end='')
for i in feature_names[0]:
	print(i,'\t',end='')
print()

for i in range(23):
	min_error = np.inf
	best_alpha = -1
	print(top_row[ratings_start_point+i],'\t',end='')
	al=0.01
	for c in range(20):
		al = 0.01*c
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter = 100)
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
	list = best_clf.coef_.tolist()
	for val in list:
		print(val,'\t',end='')
	# print()
	print(min_error)
# exit(1)
	

print('-------Based on SHORE+OpenFace features last  minute---------')
f = open ('S3ALL.csv','r')
n_line = file_len('S3ALL.csv')
features = []
ratings = []
SA_confidance_index = 4
top_row = f.readline().split(',')
# print(top_row[204:209])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' :
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[4:ratings_start_point-2]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings) 

features = preprocessing.scale(features)
feature_names = []
feature_names.append([j for j in top_row[4:ratings_start_point-2]])
feature_names = np.array(feature_names)

print(' ','\t',end='')
for i in feature_names[0]:
	print(i,'\t',end='')
print()

for i in range(23):
	min_error = np.inf
	best_alpha = -1
	print(top_row[ratings_start_point+i],'\t',end='')
	al=0.01
	for c in range(20):
		al = 0.01*c
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter = 100)
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
	list = best_clf.coef_.tolist()
	for val in list:
		print(val,'\t',end='')
	# print()
	print(min_error)
# exit(1)