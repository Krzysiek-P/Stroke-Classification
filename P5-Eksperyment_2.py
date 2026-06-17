################################################################################################
# EKSPERYMENT 2: THRESHOLD MOVING (ZMIANA PROGU DECYZYJNEGO)
# 
# Cel: Sprawdzenie, czy obniżenie progu decyzyjnego (domyślnie 0.5) 
#      pozwoli lepiej wykrywać rzadką klasę (udar) bez konieczności 
#      sztucznego balansowania zbioru danych (resamplingu).
#
# Testowane modele:
#   1. RandomForestClassifier (z parametrami regularyzacji zapobiegającymi przeuczeniu)
#   2. GaussianNB
#   3. KNeighborsClassifier (n_neighbors=20, dla płynnych prawdopodobieństw)
#
# Metryka główna do optymalizacji: BAC (Balanced Accuracy)
# Walidacja: RepeatedStratifiedKFold (5 powtórzeń, 2 foldy)
################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

################################################################################################
# 1. Wczytanie danych
print("Wczytywanie danych i przygotowanie do eksperymentu...")
data = pd.read_csv('preprocessed_data.csv')
X = data.drop(columns=['stroke']).values
y = data['stroke'].values

################################################################################################
# 2. Ustawienia eksperymentu

# Progi decyzyjne do przetestowania (od 1% do 80% co 2 punkty procentowe)
thresholds = np.arange(0.01, 0.80, 0.02) 

# Definicja modeli
models = {
    'RandomForest': RandomForestClassifier(
        random_state=42, 
        max_depth=5,           # Zapobiega zbyt głębokim drzewom
        min_samples_split=20,  # Zapobiega podziałom na zbyt małych grupach
        min_samples_leaf=10    # Wymusza minimum 10 pacjentów w liściu końcowym
    ),
    'GaussianNB': GaussianNB(),
    'KNN_20': KNeighborsClassifier(n_neighbors=20) 
}

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Struktura na wyniki: dodano klucz 'train_bac' służący do bezpiecznego wyboru progu
results = {
    model_name: {thresh: {'train_bac': [], 'bac': [], 'recall': [], 'precision': [], 'f1': []} for thresh in thresholds}
    for model_name in models.keys()
}

################################################################################################
# 3. Modelowanie i walidacja krzyżowa

print("Rozpoczęcie walidacji krzyżowej... \n")

for train_index, test_index in rskf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    # Preprocessing (imputacja braków + skalowanie)
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    # Iteracja po każdym modelu
    for model_name, model in models.items():
        # Trenowanie na niezbalansowanych danych
        model.fit(X_train, y_train)
        
        # Używamy wewnętrznej walidacji krzyżowej (cross_val_predict) do znalezienia 
        # realistycznych (Out-Of-Fold) prawdopodobieństw na zbiorze treningowym.
        # Eliminuje to overfitting i sztuczne wyniki BAC równe 1.000.
        y_train_proba = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]
        
        # Klasyczne predykcje prawdopodobieństw dla niewidzianego zbioru testowego
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Testowanie każdego progu dla danego modelu
        for thresh in thresholds:
            # Tworzenie predykcji na podstawie niestandardowego progu
            y_train_pred_custom = (y_train_proba >= thresh).astype(int)
            y_pred_custom = (y_pred_proba >= thresh).astype(int)

            # Zapisanie wyników dla danego foldu
            results[model_name][thresh]['train_bac'].append(balanced_accuracy_score(y_train, y_train_pred_custom))
            results[model_name][thresh]['bac'].append(balanced_accuracy_score(y_test, y_pred_custom))
            results[model_name][thresh]['recall'].append(recall_score(y_test, y_pred_custom))
            results[model_name][thresh]['precision'].append(precision_score(y_test, y_pred_custom, zero_division=0))
            results[model_name][thresh]['f1'].append(f1_score(y_test, y_pred_custom))

################################################################################################
# 4. Podsumowanie i wyświetlenie wyników

print("=" * 80)
print("WYNIKI EKSPERYMENTU: ZMIANA PROGU DECYZYJNEGO (THRESHOLD MOVING)")
print("=" * 80)

# Słownik do przechowywania najlepszych wyników dla podsumowania
best_results = {}

# Wypisywanie szczegółowych danych z każdego thresholdu
for model_name in models.keys():
    print(f"\n{'='*75}")
    print(f"KLASYFIKATOR: {model_name} - WYNIKI SZCZEGÓŁOWE")
    print(f"{'='*75}")
    
    best_thresh = None
    best_bac = 0
    best_train_bac = 0  # Nowa zmienna do decydowania o najlepszym progu
    best_metrics_at_thresh = {}

    print(f"{'Próg':<8} | {'Train BAC':<10} | {'Test BAC':<8} | {'Recall':<8} | {'Precision':<10} | {'F1 Score':<8}")
    print("-" * 75)

    for thresh in thresholds:
        mean_train_bac = np.mean(results[model_name][thresh]['train_bac']) # BAC na zbiorze treningowym (OOF)
        mean_bac = np.mean(results[model_name][thresh]['bac'])             # BAC na zbiorze testowym
        mean_recall = np.mean(results[model_name][thresh]['recall'])
        mean_precision = np.mean(results[model_name][thresh]['precision'])
        mean_f1 = np.mean(results[model_name][thresh]['f1'])
        
        # Wyświetlanie wyników dla każdego progu
        print(f"{thresh:.2f}     | {mean_train_bac:.3f}      | {mean_bac:.3f}    | {mean_recall:.3f}    | {mean_precision:.3f}       | {mean_f1:.3f}")
        
        # Zapamiętywanie najlepszego wyniku
        # Decyzję o wyborze progu podejmujemy bezpiecznie na podstawie Train BAC
        if mean_train_bac > best_train_bac:
            best_train_bac = mean_train_bac
            best_bac = mean_bac  # Do raportu zapisujemy to, jak ten próg zachował się na testach
            best_thresh = thresh
            best_metrics_at_thresh = {'recall': mean_recall, 'precision': mean_precision, 'f1': mean_f1}
            
    # Zapis najlepszego wyniku do późniejszego podsumowania
    best_results[model_name] = {
        'best_thresh': best_thresh,
        'best_bac': best_bac,
        'metrics': best_metrics_at_thresh
    }

# Wyświetlanie zbiorczego podsumowania na samym końcu
print("\n" + "=" * 80)
print("PODSUMOWANIE NAJLEPSZYCH WYNIKÓW (DLA PROGU WYBRANEGO NA ZBIORZE TRENINGOWYM)")
print("=" * 80)

for model_name, res in best_results.items():
    print(f"\nPODSUMOWANIE DLA {model_name}:")
    print(f"  Najlepszy próg decyzyjny: {res['best_thresh']:.2f}")
    print(f"  BAC przy tym progu:       {res['best_bac']:.3f}")
    print(f"  Recall przy tym progu:    {res['metrics']['recall']:.3f}")
    print(f"  Precision przy tym progu: {res['metrics']['precision']:.3f}")