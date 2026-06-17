################################################################################################
#importy

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

gnb = GaussianNB()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10)
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)
undersample = RandomUnderSampler(random_state=42)
smote = SMOTE(random_state=42)

results = {
    'Undersampling': {
        'GaussianNB': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'KNN': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'DecisionTree': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'RandomForest': {'bac': [], 'recall': [], 'precision': [], 'f1': []}
    },
    'SMOTE': {
        'GaussianNB': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'KNN': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'DecisionTree': {'bac': [], 'recall': [], 'precision': [], 'f1': []},
        'RandomForest': {'bac': [], 'recall': [], 'precision': [], 'f1': []}
    }
}

################################################################################################
#modelowanie

# definicja metod resamplingu
resampling_methods = {
    'Undersampling': undersample,
    'SMOTE': smote
}

for method_name, resampler in resampling_methods.items():  
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        # podział danych na zbiór treningowy i testowy
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # uzupełnienie braków w bmi
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # skalowanie cech
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # resampling danych treningowych
        #X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        #bez resamplingu, aby sprawdzić, jak modele radzą sobie na niezbalansowanych danych
        X_train_resampled, y_train_resampled = X_train, y_train
        
        # ===== GaussianNB =====
        gnb.fit(X_train_resampled, y_train_resampled)
        predict_gnb = gnb.predict(X_test)
        results[method_name]['GaussianNB']['bac'].append(balanced_accuracy_score(y_test, predict_gnb))
        results[method_name]['GaussianNB']['recall'].append(recall_score(y_test, predict_gnb))
        results[method_name]['GaussianNB']['precision'].append(precision_score(y_test, predict_gnb))
        results[method_name]['GaussianNB']['f1'].append(f1_score(y_test, predict_gnb))
        
        # ===== KNN =====
        knn.fit(X_train_resampled, y_train_resampled)
        predict_knn = knn.predict(X_test)
        results[method_name]['KNN']['bac'].append(balanced_accuracy_score(y_test, predict_knn))
        results[method_name]['KNN']['recall'].append(recall_score(y_test, predict_knn))
        results[method_name]['KNN']['precision'].append(precision_score(y_test, predict_knn))
        results[method_name]['KNN']['f1'].append(f1_score(y_test, predict_knn))
        
        # ===== Decision Tree =====
        dt.fit(X_train_resampled, y_train_resampled)
        predict_dt = dt.predict(X_test)
        results[method_name]['DecisionTree']['bac'].append(balanced_accuracy_score(y_test, predict_dt))
        results[method_name]['DecisionTree']['recall'].append(recall_score(y_test, predict_dt))
        results[method_name]['DecisionTree']['precision'].append(precision_score(y_test, predict_dt))
        results[method_name]['DecisionTree']['f1'].append(f1_score(y_test, predict_dt))

        # ===== Random Forest =====
        rf.fit(X_train_resampled, y_train_resampled)
        predict_rf = rf.predict(X_test)
        results[method_name]['RandomForest']['bac'].append(balanced_accuracy_score(y_test, predict_rf))
        results[method_name]['RandomForest']['recall'].append(recall_score(y_test, predict_rf))
        results[method_name]['RandomForest']['precision'].append(precision_score(y_test, predict_rf))
        results[method_name]['RandomForest']['f1'].append(f1_score(y_test, predict_rf))

################################################################################################
#podsumowanie wyników
print("\n" + "=" * 80)
print("PODSUMOWANIE WYNIKÓW")
print("=" * 80)

for method_name in results.keys():
    print(f"\n{'='*60}")
    print(f"METODA RESAMPLINGU: {method_name}")
    print(f"{'='*60}")
    for clf_name in results[method_name].keys():
        print(f"\n--- {clf_name} ---")
        print(f"  BAC: {np.mean(results[method_name][clf_name]['bac']):.3f} ± {np.std(results[method_name][clf_name]['bac']):.3f}")
        print(f"  Recall: {np.mean(results[method_name][clf_name]['recall']):.3f} ± {np.std(results[method_name][clf_name]['recall']):.3f}")
        print(f"  Precision: {np.mean(results[method_name][clf_name]['precision']):.3f} ± {np.std(results[method_name][clf_name]['precision']):.3f}")
        print(f"  F1 Score: {np.mean(results[method_name][clf_name]['f1']):.3f} ± {np.std(results[method_name][clf_name]['f1']):.3f}")

print("\n" + "=" * 80)
print("PORÓWNANIE - ŚREDNIE BAC (Undersampling vs SMOTE)")
print("=" * 80)
for clf_name in results['Undersampling'].keys():
    under_bac = np.mean(results['Undersampling'][clf_name]['bac'])
    smote_bac = np.mean(results['SMOTE'][clf_name]['bac'])
    print(f"{clf_name}: Undersampling = {under_bac:.3f} | SMOTE = {smote_bac:.3f}")