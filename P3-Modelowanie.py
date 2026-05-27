################################################################################################
#importy

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

#ignorowanie warningów - biblioteka generuje dużo FutureWarningów, które nie mają wpływu na działanie kodu, a jedynie zaśmiecają output
warnings.filterwarnings('ignore')

################################################################################################
#wczytanie danych i przygotowanie do modelowania

data = pd.read_csv('preprocessed_data.csv')
X = data.drop(columns=['stroke']).values
y = data['stroke'].values

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

clf = GaussianNB()
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2)
undersample = RandomUnderSampler()

bac_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

################################################################################################
#modelowanie

for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    #podział danych na zbiór treningowy i testowy
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    #uzupełnienie braków w bmi
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    #skalowanie cech
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #undersampling danych treningowych
    X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)
    
    #trenowanie modelu i ocena
    clf.fit(X_train_resampled, y_train_resampled)
    predict = clf.predict(X_test)

    #obliczanie metryk    
    bac_scores.append(balanced_accuracy_score(y_test, predict))
    recall_scores.append(recall_score(y_test, predict))
    precision_scores.append(precision_score(y_test, predict))
    f1_scores.append(f1_score(y_test, predict))

################################################################################################
#podsumowanie wyników
print(f'BAC: {np.mean(bac_scores):.3f} ± {np.std(bac_scores):.3f}')
print(f'Recall: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}')
print(f'Precision: {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}')
print(f'F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}')
