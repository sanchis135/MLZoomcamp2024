import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy.stats.mstats import winsorize
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


df = pd.read_csv('C:/Users/Sandra/Documents/Cursos/MLZoomcamp2024/Capstone_1_Project/dataset/heart_data.csv')
output_RF = f'C:/Users/Sandra/Documents/Cursos/MLZoomcamp2024/Capstone_1_Project/models/model_RF.bin'
output_GB = f'C:/Users/Sandra/Documents/Cursos/MLZoomcamp2024/Capstone_1_Project/models/model_GB.bin'
output_XGB = f'C:/Users/Sandra/Documents/Cursos/MLZoomcamp2024/Capstone_1_Project/models/model_XGB.bin'

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

def detect_outliers_zscore(data):
    outliers = []
    thres = 3
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers
num_cols_outliers = []
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    sample_outliers = detect_outliers_zscore(df[col])
    if len(sample_outliers)>0:
        num_cols_outliers.append(col)

for col in num_cols_outliers:
    df[col] = winsorize(df[col],limits = [0.10, 0.10], inplace = True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

del df_train['target']
del df_val['target']
del df_test['target']

def train_RF(df, y):
    cat = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = RandomForestClassifier()
    model.fit(X, y)

    return dv, model

def train_GB(df, y):
    cat = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = GradientBoostingClassifier()
    model.fit(X, y)

    return dv, model

def train_XGB(df, y):
    cat = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = XGBClassifier()
    model.fit(X, y)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

#Training Random Forest Model
dv_RF, model_RF = train_RF(df_train, y_train)
y_pred_RF = predict(df_test, dv_RF, model_RF)
auc_RF = roc_auc_score(y_test, y_pred_RF)

print(f'auc={auc_RF}')

with open(output_RF, 'wb') as f_out:
    pickle.dump((dv_RF, model_RF), f_out)

print(f'the model is saved to {output_RF}')

#Training GBClassifier Model
dv_GB, model_GB = train_GB(df_train, y_train)
y_pred_GB = predict(df_test, dv_GB, model_GB)
auc_GB = roc_auc_score(y_test, y_pred_GB)

print(f'auc={auc_GB}')

with open(output_GB, 'wb') as f_out:
    pickle.dump((dv_GB, model_GB), f_out)

print(f'the model is saved to {output_GB}')

#Training XGBClassifier Model
dv_XGB, model_XGB = train_XGB(df_train, y_train)
y_pred_XGB = predict(df_test, dv_XGB, model_XGB)
auc_XGB = roc_auc_score(y_test, y_pred_XGB)

print(f'auc={auc_XGB}')

with open(output_XGB, 'wb') as f_out:
    pickle.dump((dv_XGB, model_XGB), f_out)

print(f'the model is saved to {output_XGB}')
