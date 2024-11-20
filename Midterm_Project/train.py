import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.mstats import winsorize
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


df = pd.read_csv('Midterm_Project\\Customer_Churn.csv')
output_file = f'Midterm_Project\\model.bin'

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

df.drop(["Age Group"], axis=1, inplace=True)
df.drop(["Seconds of Use"], axis=1, inplace=True)
df.drop(["FN"], axis=1, inplace=True)
df.drop(["FP"], axis=1, inplace=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.Churn.values
y_val = df_val.Churn.values
y_test = df_test.Churn.values

del df_train['Churn']
del df_val['Churn']
del df_test['Churn']

def train(df, y):
    cat = df.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = RandomForestClassifier()
    model.fit(X, y)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

dv, model = train(df_train, y_train)

y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')

