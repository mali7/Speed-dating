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


def file_len(fname):
    with open(fname) as f:
        for t, l in enumerate(f):
            pass
    return t + 1
	
is_after_index = 2
ratings_start_point = 446
SA_confidance_index = 298
	
print('-------Based on SHORE features 1st minute---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[396:408])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[396:408]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
not_feature_index = [4,10]
features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
				ans =  np.around(ans).astype(int)
				y_true = np.around(ratings[12:,i]).astype(int)
				y_true = [1 if j>3 else 0 for j in y_true]
				ans = [1 if j>3 else 0 for j in ans]
				c_report = classification_report(ans, y_true)
				c_matrix = confusion_matrix(ans, y_true)
				Accuracy = accuracy_score(np.around(ans), y_true)
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
				ans =  np.around(ans).astype(int)
				y_true = np.around(ratings[:,i]).astype(int)
				y_true = [1 if j>3 else 0 for j in y_true]
				ans = [1 if j>3 else 0 for j in ans]
				c_report = classification_report(ans, y_true)
				c_matrix = confusion_matrix(ans, y_true)
				Accuracy = accuracy_score(np.around(ans), y_true)
	print("Ratings = ",top_row[ratings_start_point+i])
	print('MSE = ', min_error)
	print("Accuracy = ", Accuracy)
	print(c_matrix)
	print(c_report)
	# print(r2)
exit(1)

print('-------Based on SHORE features of middle part---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[408:420])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[408:420]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
not_feature_index = [4,10]
features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)
	
	
print('-------Based on SHORE features of last minute---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[420:432])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[420:432]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
not_feature_index = [4,10]
features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)
	
	
print('-------Based on SHORE features avg features---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[432:444])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[432:444]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
not_feature_index = [4,10]
features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)

	
print('-------Openface AU features of 1st minute---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[18:102])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[18:102]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)


# not_feature_index = [range(len(features))]
# del_this = [71,92]
# not_feature_index = np.delete(not_feature_index, del_this, axis=1)
# features=np.delete(features, not_feature_index, axis=1)

features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)
	
	
print('-------Openface AU features of middle ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[116:200])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[116:200]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)
	
	
print('-------Openface AU features of last ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[214:298])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[214:298]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
	print(min_error)
	# print(r2)
# exit(1)
	
	
print('-------Openface AU features of avg feat ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[312:396])
for i in range (n_line-1):
	line = f.readline().split(',')
	if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[312:396]])
		ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	zero_count = ratings[-1].count(0)
	zero_feat_count = features[-1].count(0)
	if(zero_count>4 or zero_feat_count > 4):
		ratings.pop()
		features.pop()
features = np.array(features)
ratings = np.array(ratings)
features = preprocessing.scale(features)
for i in range(22):
	min_error = np.inf
	best_alpha = -1
	al=0.01
	for c in range(20):
		clf  = linear_model.SGDRegressor(penalty='l1',alpha=al,n_iter = 100)
		al = 0.01*c
		ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			if (min_error>mean_squared_error(ratings[12:,i], ans)):
				min_error = mean_squared_error(ratings[12:,i], ans)
				best_clf = clf.fit(features[12:,:], ratings[12:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[12:,i]))**2)
				sstot = np.sum((ratings[12:,i]- np.mean(ratings[12:,i]))**2)
				r2 = ssreg/sstot
				c_report = classification_report(np.around(ans), ratings[12:,i])
				c_matrix = confusion_matrix(np.around(ans), ratings[12:,i])
				Accuracy = accuracy_score(np.around(ans), ratings[12:,i])
		else:
			if (min_error>mean_squared_error(ratings[:,i], ans)):
				min_error = mean_squared_error(ratings[:,i], ans)
				best_clf = clf.fit(features, ratings[:,i])
				best_alpha = al
				ssreg = np.sum((ans- np.mean(ratings[:,i]))**2)
				sstot = np.sum((ratings[:,i]- np.mean(ratings[:,i]))**2)
				r2 = ssreg/sstot
				c_report = classification_report(np.around(ans), ratings[:,i])
				c_matrix = confusion_matrix(np.around(ans), ratings[:,i])
				Accuracy = accuracy_score(np.around(ans), ratings[:,i])

	print('MSE = ', min_error)
	print("Accuracy = ", Accuracy)
	print(c_matrix)
	print(c_report)
	# print(r2)
# exit(1)