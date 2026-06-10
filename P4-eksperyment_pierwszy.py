################################################################################################
# EKSPERYMENT: LEKKI SMOTE + POŁĄCZENIE Z UNDERSAMPLINGIEM
# 
# Cel: Sprawdzenie, czy umiarkowane dodanie sztucznych próbek klasy udaru (klasa 1)
#      da lepsze wyniki niż pełny SMOTE (wyrównanie 1:1) lub undersampling.
#      Dodatkowo sprawdzenie połączenia SMOTE + Undersampling (pełne wyrównanie 1:1
#      przy różnych docelowych liczebnościach).
#
# Testowane warianty:
#   1. Undersampling (baseline) - usuwanie zdrowych próbek (249:249)
#   2. SMOTE_500/750/1000/1500 - tylko oversampling (nadal niezbalansowane)
#   3. SMOTE_full - pełny oversampling do 4860:4860
#   4. SMOTE+Under_500/750/1000/1500 - najpierw SMOTE, potem Undersampling (pełne wyrównanie 1:1)
#
# Klasyfikator: Decision Tree (najlepszy z poprzednich eksperymentów)
# Metryka główna: BAC (Balanced Accuracy)
# Walidacja: RepeatedStratifiedKFold (5 powtórzeń, 2 foldy)
################################################################################################

################################################################################################
#importy

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

################################################################################################
#wczytanie danych i przygotowanie do modelowania

print("=" * 80)
print("EKSPERYMENT: LEKKI SMOTE + POŁĄCZENIE Z UNDERSAMPLINGIEM")
print("=" * 80)

data = pd.read_csv('preprocessed_data.csv')
X = data.drop(columns=['stroke']).values
y = data['stroke'].values

# Sprawdzenie liczności klas
counts = pd.Series(y).value_counts()
n_majority = counts[0]  # klasa 0 (brak udaru)
n_minority = counts[1]   # klasa 1 (udar)

print(f"\nLiczność klas w oryginalnym zbiorze:")
print(f"  Klasa 0 (brak udaru): {n_majority}")
print(f"  Klasa 1 (udar): {n_minority}")
print(f"  Proporcja: {n_minority/n_majority*100:.2f}%")

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=42)

# Tylko Decision Tree (najlepszy z poprzednich eksperymentów)
dt = DecisionTreeClassifier(random_state=42)

################################################################################################
#definicja metod resamplingu

# Undersampling (baseline) - wyrównuje do najmniejszej klasy (249)
undersample = RandomUnderSampler(random_state=42)

# SMOTE z różnymi strategiami (docelowa liczba próbek klasy 1)
smote_targets = {
    'SMOTE_500': 500,
    'SMOTE_750': 750,
    'SMOTE_1000': 1000,
    'SMOTE_1500': 1500,
    'SMOTE_full': n_majority  # pełne wyrównanie do klasy 0
}

# Docelowe liczby dla połączenia SMOTE + Undersampling (pełne wyrównanie 1:1)
combined_targets = {
    'SMOTE+Under_500': 500,
    'SMOTE+Under_750': 750,
    'SMOTE+Under_1000': 1000,
    'SMOTE+Under_1500': 1500
}

# Słownik przechowujący wyniki
results = {}

# 1. Undersampling
results['Undersampling'] = {'bac': [], 'recall': [], 'precision': [], 'f1': [], 'resampler': undersample}

# 2. SMOTE (samo)
for target_name, target_count in smote_targets.items():
    results[target_name] = {'bac': [], 'recall': [], 'precision': [], 'f1': []}
    if target_name == 'SMOTE_full':
        results[target_name]['resampler'] = SMOTE(random_state=42)
    else:
        results[target_name]['resampler'] = SMOTE(sampling_strategy={1: target_count}, random_state=42)

# 3. SMOTE + Undersampling (połączenie)
for target_name, target_count in combined_targets.items():
    results[target_name] = {'bac': [], 'recall': [], 'precision': [], 'f1': []}
    # Pipeline: najpierw SMOTE (do target_count), potem Undersampling (do target_count)
    results[target_name]['resampler'] = ImbPipeline([
        ('smote', SMOTE(sampling_strategy={1: target_count}, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy={0: target_count}, random_state=42))
    ])

################################################################################################
#modelowanie

print("\n" + "=" * 80)
print("ROZPOCZĘCIE EKSPERYMENTU")
print("=" * 80)

# Iteracja po metodach resamplingu
for method_name in results.keys():
    resampler = results[method_name]['resampler']
    
    print(f"\n--- {method_name} ---")
    
    # Wyświetlenie informacji o metodzie
    if method_name == 'Undersampling':
        print(f"  Opis: Tylko undersampling (wyrównanie do {n_minority}:{n_minority})")
    elif method_name.startswith('SMOTE+Under'):
        target = combined_targets[method_name]
        print(f"  Opis: SMOTE -> Undersampling (pełne wyrównanie {target}:{target})")
        print(f"    Krok 1: SMOTE zwiększa klasę 1 z {n_minority} do {target} (+{target - n_minority})")
        print(f"    Krok 2: Undersampling zmniejsza klasę 0 z {n_majority} do {target} (-{n_majority - target})")
    elif method_name == 'SMOTE_full':
        print(f"  Opis: Tylko SMOTE (pełne wyrównanie do {n_majority}:{n_majority})")
        print(f"    Dodaje {n_majority - n_minority} sztucznych próbek klasy 1")
    elif method_name.startswith('SMOTE_'):
        target = smote_targets[method_name]
        print(f"  Opis: Tylko SMOTE (zwiększa klasę 1 do {target})")
        print(f"    Dodaje {target - n_minority} sztucznych próbek klasy 1")
        print(f"    UWAGA: Dalej niezbalansowane ({n_majority}:{target})")
    
    bac_scores = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    for train_index, test_index in rskf.split(X, y):
        # podział danych
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        
        # uzupełnienie braków
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        # skalowanie cech
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # resampling
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        
        # trenowanie Decision Tree
        dt.fit(X_train_resampled, y_train_resampled)
        y_pred = dt.predict(X_test)
        
        # zapis metryk
        bac_scores.append(balanced_accuracy_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # zapis wyników
    results[method_name]['bac'] = bac_scores
    results[method_name]['recall'] = recall_scores
    results[method_name]['precision'] = precision_scores
    results[method_name]['f1'] = f1_scores
    
    # wyświetlenie
    print(f"  Wyniki:")
    print(f"    BAC: {np.mean(bac_scores):.3f} ± {np.std(bac_scores):.3f}")
    print(f"    Recall: {np.mean(recall_scores):.3f} ± {np.std(recall_scores):.3f}")
    print(f"    Precision: {np.mean(precision_scores):.3f} ± {np.std(precision_scores):.3f}")
    print(f"    F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")

################################################################################################
#podsumowanie wyników (tabela porównawcza)

print("\n" + "=" * 80)
print("PODSUMOWANIE - PORÓWNANIE WSZYSTKICH METOD RESAMPLINGU")
print("=" * 80)

print("\n" + "-" * 100)
print(f"{'Metoda':<20} {'BAC':<18} {'Recall':<18} {'Precision':<18} {'F1 Score':<18}")
print("-" * 100)

for method_name in results.keys():
    bac_mean = np.mean(results[method_name]['bac'])
    bac_std = np.std(results[method_name]['bac'])
    recall_mean = np.mean(results[method_name]['recall'])
    recall_std = np.std(results[method_name]['recall'])
    precision_mean = np.mean(results[method_name]['precision'])
    precision_std = np.std(results[method_name]['precision'])
    f1_mean = np.mean(results[method_name]['f1'])
    f1_std = np.std(results[method_name]['f1'])
    
    print(f"{method_name:<20} {bac_mean:.3f}±{bac_std:.3f}   {recall_mean:.3f}±{recall_std:.3f}   {precision_mean:.3f}±{precision_std:.3f}   {f1_mean:.3f}±{f1_std:.3f}")

print("-" * 100)

################################################################################################
#wnioski - wskazanie najlepszej metody

print("\n" + "=" * 80)
print("WNIOSKI")
print("=" * 80)

# Znalezienie najlepszej metody (najwyższe BAC)
best_method = max(results.keys(), key=lambda x: np.mean(results[x]['bac']))
best_bac = np.mean(results[best_method]['bac'])

print(f"\nNajlepsza metoda resamplingu: {best_method}")
print(f"Najwyższe średnie BAC: {best_bac:.3f}")

# Porównanie z Undersampling (baseline)
baseline_bac = np.mean(results['Undersampling']['bac'])
if best_method == 'Undersampling':
    print(f"\nUndersampling (BAC={baseline_bac:.3f}) okazał się najlepszą metodą.")
    print("Żaden poziom oversamplingu ani połączenie z undersamplingiem nie poprawiły wyników.")
else:
    print(f"\n{best_method} (BAC={best_bac:.3f}) jest lepszy od Undersamplingu (BAC={baseline_bac:.3f})")
    print(f"Różnica: {best_bac - baseline_bac:.3f}")

# Ranking metod według BAC
print("\nRanking metod według BAC (od najlepszej do najgorszej):")
ranking = sorted(results.keys(), key=lambda x: np.mean(results[x]['bac']), reverse=True)
for i, method in enumerate(ranking, 1):
    bac_mean = np.mean(results[method]['bac'])
    print(f"  {i}. {method}: {bac_mean:.3f}")

# Dodatkowe podsumowanie - najlepsza metoda z pełnym wyrównaniem 1:1
print("\n" + "-" * 80)
print("Porównanie metod dających pełne wyrównanie 1:1:")
print("-" * 80)

balanced_methods = ['Undersampling', 'SMOTE_full'] + list(combined_targets.keys())
for method in balanced_methods:
    if method in results:
        bac_mean = np.mean(results[method]['bac'])
        n_samples = None
        if method == 'Undersampling':
            n_samples = n_minority
        elif method == 'SMOTE_full':
            n_samples = n_majority
        elif method.startswith('SMOTE+Under'):
            n_samples = combined_targets[method]
        print(f"  {method}: BAC = {bac_mean:.3f} (zbiór zbalansowany, {n_samples}:{n_samples} próbek)")

print("\n" + "=" * 80)