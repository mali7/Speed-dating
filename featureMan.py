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
# not_feature_index = [0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	50,	51,	52,	53,	54,	55,	56,	57,	58,	59,	60,	61,	12,	63,	100,	101,	102,	103,	104,	105,	106,	107,	108,	109,	110,	111,	112,	113,	150,	151,	152,	153,	154,	155,	156,	157,	158,	159,	160,	161,	112,	163,	205,	211,	217,	223]
# # not_feature_index = [0,1,50,51,99,100,149,150,220,221] #remember to subtract 4
# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=1)	
# # clf = linear_model.Lasso(alpha=5)

# print('-------Based on all features ---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# try:
			# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[4:ratings_start_point-2]])
			# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
		# except ValueError:
			# print(line,i)
	# zero_count = ratings[-1].count(0)
	# if(zero_count>4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# features=np.delete(features, not_feature_index, axis=1)

# #-------------preprocess - ------------------------#
# # min_max_scaler = preprocessing.MinMaxScaler()
# # features = min_max_scaler.fit_transform(features)
# features = preprocessing.scale(features)

# feature_names = []
# feature_names.append([j for j in top_row[4:ratings_start_point-2]])
# feature_names = np.array(feature_names)
# feature_names=np.delete(feature_names, not_feature_index, axis=1)
# print (feature_names)
# for i in range(22):
	# min_error = np.inf
	# for c in range(1,20):
		# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c)	
		# # clf = linear_model.Lasso(alpha=5)
		# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
		# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
			# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
			# if (min_error>mean_squared_error(ratings[12:,i], ans)):
				# min_error = mean_squared_error(ratings[12:,i], ans)
		# else:
			# if (min_error>mean_squared_error(ratings[:,i], ans)):
				# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)
	
# print('-------Based on SHORE features 1st minute---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[396:408])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[396:408]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)

# print('-------Based on SHORE features of middle part---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[408:420])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[408:420]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)
# print('-------Based on SHORE features of last minute---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[420:432])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[420:432]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)


# print('-------Based on SHORE features avg features---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[432:444])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[432:444]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)

	
# print('-------Openface AU features of 1st minute---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[18:102])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[18:102]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# # not_feature_index = [4,10]
# # features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)
	
	
# print('-------Openface AU features of middle ---------')
# f = open ('LISSA_FEATURES11.csv','r')
# n_line = file_len('LISSA_FEATURES11.csv')
# features = []
# ratings = []
# top_row = f.readline().split(',')
# print(top_row[116:200])
# for i in range (n_line-1):
	# line = f.readline().split(',')
	# if line[SA_confidance_index] != 'nan' and float(line[SA_confidance_index])>0.70:
		# features.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[116:200]])
		# ratings.append([0 if j == 'nan' or j == 'nan\n' else float(j) for j in line[ratings_start_point:]])
	# zero_count = ratings[-1].count(0)
	# zero_feat_count = features[-1].count(0)
	# if(zero_count>4 or zero_feat_count > 4):
		# ratings.pop()
		# features.pop()
# features = np.array(features)
# ratings = np.array(ratings)
# # not_feature_index = [4,10]
# # features=np.delete(features, not_feature_index, axis=1)
# features = preprocessing.scale(features)
# gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
# for i in range(22):
	# min_error = np.inf
	# for g in gamma_arr:
		# for c in range(1,20):
			# clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			# ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			# if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				# ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				# if (min_error>mean_squared_error(ratings[12:,i], ans)):
					# min_error = mean_squared_error(ratings[12:,i], ans)
			# else:
				# if (min_error>mean_squared_error(ratings[:,i], ans)):
					# min_error = mean_squared_error(ratings[:,i], ans)
	# print(min_error)
# # exit(1)
	
	
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
not_feature_index = [range(len(features))]
del_this = [71,92]
not_feature_index = np.delete(not_feature_index, del_this, axis=1)
features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
print (features.shape)
gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
for i in range(22):
	min_error = np.inf
	for g in gamma_arr:
		for c in range(1,20):
			clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				if (min_error>mean_squared_error(ratings[12:,i], ans)):
					min_error = mean_squared_error(ratings[12:,i], ans)
			else:
				if (min_error>mean_squared_error(ratings[:,i], ans)):
					min_error = mean_squared_error(ratings[:,i], ans)
	print(min_error)
exit(1)
	
	
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
# not_feature_index = [4,10]
# features=np.delete(features, not_feature_index, axis=1)
features = preprocessing.scale(features)
gamma_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1,2,3,4,5,6,7,8,9]
for i in range(22):
	min_error = np.inf
	for g in gamma_arr:
		for c in range(1,20):
			clf = svm.SVR(kernel='rbf',max_iter = 3000,C=c,gamma=g)	
			ans = cross_val_predict(clf, features, ratings[:,i], cv=5)
			if top_row[ratings_start_point+i] == 'cspostur' or  top_row[ratings_start_point+i] == 'cseyecon':
				ans = cross_val_predict(clf, features[12:,:], ratings[12:,i], cv=5)
				if (min_error>mean_squared_error(ratings[12:,i], ans)):
					min_error = mean_squared_error(ratings[12:,i], ans)
			else:
				if (min_error>mean_squared_error(ratings[:,i], ans)):
					min_error = mean_squared_error(ratings[:,i], ans)
	print(min_error)
# exit(1)