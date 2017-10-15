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
from sklearn.metrics import r2_score
# warnings.filterwarnings("ignore", ConvergenceWarning)

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
feat_names = top_row[396:408]
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
feat_names = np.delete(feat_names, not_feature_index, axis=0)
features = preprocessing.scale(features)

for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)


print('-------Based on SHORE features of middle part---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[408:420])
feat_names = top_row[408:420]
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
feat_names = np.delete(feat_names, not_feature_index, axis=0)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)

print('-------Based on SHORE features of last minute---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[420:432])
feat_names = top_row[420:432]
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
feat_names = np.delete(feat_names, not_feature_index, axis=0)
features = preprocessing.scale(features)
gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4]
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)

print('-------Based on SHORE features avg features---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[432:444])
feat_names = top_row[432:444]
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
feat_names = np.delete(feat_names, not_feature_index, axis=0)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)

	
print('-------Openface AU features of 1st minute---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[18:102])
feat_names = top_row[18:102]
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
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)
	
	
print('-------Openface AU features of middle ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[116:200])
feat_names = top_row[116:200]
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
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)
	
print('-------Openface AU features of last ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[214:298])
feat_names = top_row[214:298]
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
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)
	
	
print('-------Openface AU features of avg feat ---------')
f = open ('LISSA_FEATURES11.csv','r')
n_line = file_len('LISSA_FEATURES11.csv')
features = []
ratings = []
top_row = f.readline().split(',')
print(top_row[312:396])
feat_names = top_row[312:396]
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
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
for i in range(22):
	lowest_mse_feat = []
	feat_list = []
	X = []
	current_best = 0
	min_mse_sofar = np.inf
	min_error = np.inf
	for k in range(features.shape[1]):
		for feat in range(features.shape[1]):
			if feat not in lowest_mse_feat:
				X.append(features[:,feat])
				al=0.01
				for c in range(10):
					al = 0.1*c
					clf  = linear_model.SGDRegressor(penalty='l1',alpha=al, n_iter=50 )						
					ans = cross_val_predict(clf, np.array(X).T, ratings[:,i], cv=5)
					if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
						ans = cross_val_predict(clf, np.array(X).T[12:,:], ratings[12:,i], cv=5)
						if (min_error>mean_squared_error(ratings[12:,i], ans)):
							min_error = mean_squared_error(ratings[12:,i], ans)
							r2 = abs(r2_score(ratings[12:,i], ans))
							current_best = feat
					else:
						if (min_error>mean_squared_error(ratings[:,i], ans)):
							min_error = mean_squared_error(ratings[:,i], ans)
							r2 = abs(r2_score(ratings[:,i], ans))
							current_best = feat
				X.pop()
		if (min_mse_sofar>min_error):
			lowest_mse_feat.append(current_best)
			X.append(features[:,current_best])
			feat_list.append(feat_names[current_best])
			min_mse_sofar = min_error
		else:
			break
		# print(min_error)
	print(min_mse_sofar,'\t',r2,'\t', feat_list)
# exit(1)