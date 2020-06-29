# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import warnings
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
import gc
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# train_path = '/content/drive/My Drive/pa_task_data'
# 设置文件读取路径
train_path = '.'

df_data = pd.read_csv(os.path.join(train_path, 'pa-task-data.csv'))
df_data['type'] = 'train'
# df_data.to_pickle('/content/drive/My Drive/data.plk')

df_data.head()

df_data['scr'].hist()

# 数据特征猜测： 学历
df_data['diploma'].value_counts()

# 数据特征猜测： 是否拥有房产
df_data['home_ownership'].value_counts()

# 数据特征猜测： 是否拥有车
df_data['car_ownership'].value_counts()

df_data['location'].value_counts()

# 数据特征猜测： 星座
df_data['constellation'].value_counts()

# 牡羊座/白羊座 (3/21 - 4/20)的英文名：Aries
# 金牛座 (4/21 - 5/20)的英文名： Taurus
# 双子座 (5/21 - 6/21)的英文名： Gemini
# 巨蟹座 (6/22 - 7/22)的英文名： Cancer
# 狮子座 (7/23 - 8/22)的英文名： Leo
# 处女座/室女座 (8/23 - 9/22)的英文名： Virgo
# 天秤座 (9/23 - 10/22)的英文名： Libra
# 天蝎座 (10/23 - 11/21)的英文名： Scorpio
# 射手座/人马座 (11/22 - 12/21)的英文名： Sagittarius
# 魔羯座/山羊座 (12/22 - 1/19)的英文名： Capricorn
# 水瓶座 (1/20 - 2/18)的英文名： Aquarius
# 双鱼座 (2/19 - 3/20)的英文名： Pisces
# https://www.8s8s.com/xingzuo/xingzuozhishi/18624.html

# 数据特征猜测： 用户等级（分段）
df_data['grade'].value_counts()

df_data['index'].value_counts()

# 数据特征猜测： 贷款数量
df_data['dk_cnt'].value_counts()

# 数据特征猜测： 总金额？
df_data['tot_amnt'].value_counts()

# 数据特征猜测： 收入（分段）
df_data['income'].value_counts()

# 数据特征猜测： 年龄
df_data['gender'].value_counts()

# 数据特征猜测： 职业领域
df_data['occupation'].value_counts()

# 数据特征猜测： 贷款金额
df_data['dk_amnt(k)'].hist()

# 数据特征猜测： 工作时长
df_data['emp_length'].value_counts()

# 数据特征猜测： dq？数量
df_data['dq_cnt'].value_counts()

# label 分布
df_data['y'].value_counts()

# from feature_selector import FeatureSelector
# https://github.com/WillKoehrsen/feature-selector

# fs = FeatureSelector(data = df_data.drop(columns = ['y', 'type']), labels = df_data['y'])

# fs.identify_missing(missing_threshold=0.01)

# fs.identify_single_unique()

# fs.identify_collinear(correlation_threshold=0.9)

# fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
#                             n_iterations = 10, early_stopping = True)

df_feature = df_data.copy()

# df_feature.to_pickle('/content/drive/My Drive/feature.plk')

# df_feature = pd.read_pickle('/content/drive/My Drive/feature.plk')

for f in df_feature.select_dtypes('object'):
    if f not in ['date', 'type']:
        print(f)
        lbl = LabelEncoder()
        df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

df_feature['home_car_ownership'] = df_feature[['home_ownership', 'car_ownership']].apply(lambda x: x[0]+x[1]*2, axis=1)

df_feature['dk_dq_cnt_ratio'] = df_feature['dk_cnt'] / df_feature['dq_cnt']
df_feature['dk_dq_cnt_sum'] = df_feature['dk_cnt'] + df_feature['dq_cnt']

df_feature['dk_amnt_per_cnt'] = df_feature['dk_amnt(k)'] / df_feature['dk_cnt']

df_feature['dk_amnt_income_ratio'] = df_feature['dk_amnt(k)'] / df_feature['income']

df_feature['dk_amnt_emp_length_ratio'] = df_feature['dk_amnt(k)'] / df_feature['emp_length']

df_feature['income_occupation'] = df_feature[['income', 'occupation']].apply(lambda x: x[0]*7+x[1], axis=1)

# df_feature.to_pickle('/content/drive/My Drive/feature_1.plk')

# df_feature = pd.read_pickle('/content/drive/My Drive/feature_1.plk')

def group_base_mean(df_feature, group, base):
  group_array = df_feature[group].unique()
  f_dict = {}
  for group_element in group_array:
    f_dict[group_element] = df_feature[df_feature[group] == group_element][base].mean()
  df_feature[group+"_"+base+"_mean_diff"] = df_feature[[base, group]].apply(lambda x: x[0]-f_dict[x[1]], axis=1)

def group_base_median(df_feature, group, base):
  group_array = df_feature[group].unique()
  f_dict = {}
  for group_element in group_array:
    f_dict[group_element] = df_feature[df_feature[group] == group_element][base].median()
  df_feature[group+"_"+base+"_median_diff"] = df_feature[[base, group]].apply(lambda x: x[0]-f_dict[x[1]], axis=1)

def group_base_max(df_feature, group, base):
  group_array = df_feature[group].unique()
  f_dict = {}
  for group_element in group_array:
    f_dict[group_element] = df_feature[df_feature[group] == group_element][base].max()
  df_feature[group+"_"+base+"_max_diff"] = df_feature[[base, group]].apply(lambda x: x[0]-f_dict[x[1]], axis=1)

def group_base_min(df_feature, group, base):
  group_array = df_feature[group].unique()
  f_dict = {}
  for group_element in group_array:
    f_dict[group_element] = df_feature[df_feature[group] == group_element][base].min()
  df_feature[group+"_"+base+"_min_diff"] = df_feature[[base, group]].apply(lambda x: x[0]-f_dict[x[1]], axis=1)

for group in ['constellation', 'grade', 'index', 'occupation', 'diploma']:
  for base in ['scr', 'dk_cnt', 'tot_amnt', 'income', 'dk_amnt(k)', 'emp_length', 'dq_cnt']:
    for f in [group_base_mean, group_base_median, group_base_max, group_base_min]:
      f(df_feature, group, base)

# df_feature.to_pickle('/content/drive/My Drive/feature_2.plk')

# df_feature = pd.read_pickle('/content/drive/My Drive/feature_2.plk')

df_feature['scr_bin'] = pd.cut(df_feature['scr'], bins=5, labels=['scr_bin_0', 'scr_bin_1', 'scr_bin_2', 'scr_bin_3', 'scr_bin_4'])

for group in ['scr_bin']:
  for base in ['scr', 'diploma', 'constellation', 'grade', 'index', 'dk_cnt', 'tot_amnt', 'income', 'occupation', 'dk_amnt(k)', 'emp_length', 'dq_cnt']:
    for f in [group_base_mean, group_base_median, group_base_max, group_base_min]:
      f(df_feature, group, base)

df_feature['scr_bin'] = lbl.fit_transform(df_feature['scr_bin'])

# df_feature.to_pickle('/content/drive/My Drive/feature_3.plk')

# df_feature = pd.read_pickle('/content/drive/My Drive/feature_3.plk')

df_feature['dk_amnt(k)_bin'] = pd.cut(df_feature['dk_amnt(k)'], bins=10, labels=['dk_amnt(k)_bin_0', 'dk_amnt(k)_bin_1', 'dk_amnt(k)_bin_2', 'dk_amnt(k)_bin_3', 'dk_amnt(k)_bin_4', 'dk_amnt(k)_bin_5', 'dk_amnt(k)_bin_6', 'dk_amnt(k)_bin_7', 'dk_amnt(k)_bin_8', 'dk_amnt(k)_bin_9'])
for group in ['dk_amnt(k)_bin']:
  for base in ['scr', 'diploma', 'constellation', 'grade', 'index', 'dk_cnt', 'tot_amnt', 'income', 'occupation', 'dk_amnt(k)', 'emp_length', 'dq_cnt']:
    for f in [group_base_mean, group_base_median, group_base_max, group_base_min]:
      f(df_feature, group, base)

df_feature['dk_amnt(k)_bin'] = lbl.fit_transform(df_feature['dk_amnt(k)_bin'])

# df_feature.to_pickle('/content/drive/My Drive/feature_4.plk')

# df_feature = pd.read_pickle('/content/drive/My Drive/feature_4.plk')

label_0_idxs = df_feature.index[df_feature['y'] == 0].to_list()
label_1_idxs = df_feature.index[df_feature['y'] == 1].to_list()
test_size_0 = 19200
test_size_1 = 800
import random
random.seed(2020)
label_0_test_idxs = random.sample(label_0_idxs, test_size_0)
label_1_test_idxs = random.sample(label_1_idxs, test_size_1)

df_feature.head()

for f in df_feature.select_dtypes('object'):
    if f not in ['date', 'type']:
        print(f)
        lbl = LabelEncoder()
        df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

for idx in label_0_test_idxs + label_1_test_idxs:
  df_feature.iloc[idx, df_feature.columns.get_loc('type')] = 'test'

seed = 2020
df_train = df_feature[df_feature['type'] == 'train'].copy()
df_train = shuffle(df_train, random_state=seed)

delete_list = []

ycol = 'y'

feature_names = list(
    filter(lambda x: x not in ([ycol, 'type'] + delete_list), df_train.columns))

model = lgb.LGBMClassifier(num_leaves=64,
                           max_depth=10,
                           learning_rate=0.1,
                           n_estimators=10000000,
                           subsample=0.8,
                           feature_fraction=0.8,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=seed,
                           metric=None
                           )

# # SMOTE 算法处理样本不平衡
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=2020)
# X_res, y_res = sm.fit_resample(df_train[feature_names], df_train[ycol])
# df_train = pd.DataFrame(X_res)
# df_train.columns = feature_names
# df_train[ycol] = y_res
# df_train = shuffle(df_train, random_state=seed)

oof = []
df_importance_list = []
prediction_list = []
df_test = df_feature[df_feature['type'] == 'test'].copy()
prediction = pd.DataFrame([0 for i in range(len(df_test))])
prediction.columns = ['target']

prediction_train_all_list = []
df_train_all = df_feature.copy()
prediction_train_all = pd.DataFrame([0 for i in range(len(df_train_all))])
prediction_train_all.columns = ['target']

kfold = KFold(n_splits=10)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train[feature_names], df_train[ycol])):
    X_train = df_train.iloc[trn_idx][feature_names]
    Y_train = df_train.iloc[trn_idx][ycol]
    
    X_val = df_train.iloc[val_idx][feature_names]
    Y_val = df_train.iloc[val_idx][ycol]

    print('\nFold_{} Training ================================\n'.format(fold_id+1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

    pred_val = lgb_model.predict_proba(
        X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
    df_oof = df_train.iloc[val_idx][[ycol]].copy()
    df_oof['pred'] = pred_val
    oof.append(df_oof)

    pred_test = lgb_model.predict_proba(
        df_test[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
    prediction['target'] += pred_test / 10
    prediction_list.append(pred_test.copy())

    pred_train_all = lgb_model.predict_proba(
        df_train_all[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
    prediction_train_all['target'] += pred_train_all / 10
    prediction_train_all_list.append(pred_train_all.copy())


    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
df_importance

df_oof_train = pd.concat(oof, sort=False)

df_oof_train.head()

def pre2label(pred_list, threshold):
  return [0 if pred < threshold else 1 for pred in pred_list]

def score(labels, pres, threshold=0.5):
  print("roc_auc_score: "+str(roc_auc_score(labels, pres)))
  print("accuracy_score: "+str(accuracy_score(labels, pre2label(pres, threshold))))
  print("recall_score: "+str(recall_score(labels, pre2label(pres, threshold))))
  print("precision_score: "+str(precision_score(labels, pre2label(pres, threshold))))
  print("f1_score: "+str(f1_score(labels, pre2label(pres, threshold))))

"""# After Data mining (dk amnt Group)"""

score(df_oof_train['y'], df_oof_train['pred'], 0.5)

score(df_oof_train['y'], df_oof_train['pred'], 0.3)

score(df_test[ycol], prediction['target'], 0.5)

score(df_test[ycol], prediction['target'], 0.3)

score(df_train_all[ycol], prediction_train_all['target'], 0.5)

score(df_train_all[ycol], prediction_train_all['target'], 0.3)

"""# After Data mining (scr Group)"""

score(df_oof_train['y'], df_oof_train['pred'], 0.5)

score(df_oof_train['y'], df_oof_train['pred'], 0.3)

score(df_test[ycol], prediction['target'], 0.5)

score(df_test[ycol], prediction['target'], 0.3)

score(df_train_all[ycol], prediction_train_all['target'], 0.5)

score(df_train_all[ycol], prediction_train_all['target'], 0.3)

"""# After Data mining (Basic Group)"""

score(df_oof_train['y'], df_oof_train['pred'], 0.5)

score(df_oof_train['y'], df_oof_train['pred'], 0.3)

score(df_test[ycol], prediction['target'], 0.5)

score(df_test[ycol], prediction['target'], 0.3)

score(df_train_all[ycol], prediction_train_all['target'], 0.5)

score(df_train_all[ycol], prediction_train_all['target'], 0.3)

"""# Basic Features"""

score(df_oof_train['y'], df_oof_train['pred'], 0.5)

score(df_oof_train['y'], df_oof_train['pred'], 0.3)

score(df_test[ycol], prediction['target'], 0.5)

score(df_test[ycol], prediction['target'], 0.3)

score(df_train_all[ycol], prediction_train_all['target'], 0.5)

score(df_train_all[ycol], prediction_train_all['target'], 0.3)

"""# Baseline"""

score(df_oof_train['y'], df_oof_train['pred'], 0.5)

score(df_oof_train['y'], df_oof_train['pred'], 0.3)

score(df_test[ycol], prediction['target'], 0.5)

score(df_test[ycol], prediction['target'], 0.3)

score(df_train_all[ycol], prediction_train_all['target'], 0.5)

score(df_train_all[ycol], prediction_train_all['target'], 0.3)

"""# Results"""

res = [
       [0.9631984268450245, 0.972175, 0.6453154875717017, 0.9625846354166665, 0.97285, 0.6561114629512349, 0.9822388431272205, 0.97841, 0.7272267845862286],
       [0.9628956910589479, 0.9723875, 0.6439967767929091, 0.9621458984375001, 0.97325, 0.6568313021167416, 0.9849011994665718, 0.97992, 0.7460799190692969],
       [0.9623005151955559, 0.9725, 0.6490108487555839, 0.9621179687500001, 0.97255, 0.6500956022944551, 0.9875234775195488, 0.9817, 0.7711355677838919],
       [0.962000514207119, 0.9724375, 0.6474820143884892, 0.9618130859374998, 0.97225, 0.6458200382897256, 0.9874500288932938, 0.98189, 0.7730860794386667],
       [0.9626019212550064, 0.9721875, 0.6446254591918222, 0.9621425130208333, 0.97235, 0.6466453674121405, 0.9882003885443795, 0.98264, 0.7827284105131415]
]

import matplotlib.pyplot as plt

plt.plot([row[0] for row in res], label='dev')
plt.plot([row[3] for row in res], label='test')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('roc_auc_score')
plt.title('Dev vs Test on roc_auc_score')
plt.legend()

plt.plot([row[1] for row in res], label='dev')
plt.plot([row[4] for row in res], label='test')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('accuracy_score')
plt.title('Dev vs Test on accuracy_score (threshold=0.3)')
plt.legend()

plt.plot([row[2] for row in res], label='dev')
plt.plot([row[5] for row in res], label='test')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('f1_score')
plt.title('Dev vs Test on f1_score (threshold=0.3)')
plt.legend()

plt.plot([row[0] for row in res], label='dev')
plt.plot([row[3] for row in res], label='test')
plt.plot([row[6] for row in res], label='full')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('roc_auc_score')
plt.title('Full vs (Dev / Test) on roc_auc_score')
plt.legend()

plt.plot([row[1] for row in res], label='dev')
plt.plot([row[4] for row in res], label='test')
plt.plot([row[7] for row in res], label='full')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('accuracy_score')
plt.title('Full vs (Dev / Test) on accuracy_score (threshold=0.3)')
plt.legend()

plt.plot([row[2] for row in res], label='dev')
plt.plot([row[5] for row in res], label='test')
plt.plot([row[8] for row in res], label='full')
plt.xlabel('Experiment index: 0-Basseline; 1-Basic Features; 2-Basic Group; 3-scr Group; 4-dk amnt Group')
plt.ylabel('f1_score')
plt.title('Full vs (Dev / Test) on f1_score (threshold=0.3)')
plt.legend()