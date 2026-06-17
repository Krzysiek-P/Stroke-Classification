################################################################################################
# EKSPERYMENT: LEKKI SMOTE + POŁĄCZENIE Z UNDERSAMPLINGIEM (Z ROZSZERZENIEM O INNE MODELE)
# 
# Cel: Sprawdzenie, czy umiarkowane dodanie sztucznych próbek klasy udaru (klasa 1)
#      da lepsze wyniki niż pełny SMOTE (wyrównanie 1:1) lub undersampling.
#      Dodatkowo sprawdzenie połączenia SMOTE + Undersampling (pełne wyrównanie 1:1
#      przy różnych docelowych liczebnościach).
#      Dodano również obliczanie wyników dla innych klasyfikatorów.
#
# Testowane warianty:
#   1. Undersampling (baseline) - usuwanie zdrowych próbek (249:249)
#   2. SMOTE_500/750/1000/1500 - tylko oversampling (nadal niezbalansowane)
#   3. SMOTE_full - pełny oversampling do 4860:4860
#   4. SMOTE+Under_500/750/1000/1500 - najpierw SMOTE, potem Undersampling (pełne wyrównanie 1:1)
#
# Klasyfikatory: Decision Tree (główny opis) + Random Forest, KNN, GaussianNB (w tle)
# Metryka główna: BAC (Balanced Accuracy)
# Walidacja: RepeatedStratifiedKFold (5 powtórzeń, 2 foldy)
################################################################################################

################################################################################################
# importy

import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

################################################################################################
# wczytanie danych i przygotowanie do modelowania

print("=" * 80)
print("EKSPERYMENT: LEKKI SMOTE + POŁĄCZENIE Z UNDERSAMPLINGIEM (ORAZ INNE MODELE)")
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

################################################################################################
# definicja metod resamplingu i modeli

models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'GaussianNB': GaussianNB()
}

# Słownik na obiekty resamplingu
resamplers = {}

# 1. Undersampling (baseline) - wyrównuje do najmniejszej klasy (249)
resamplers['Undersampling'] = RandomUnderSampler(random_state=42)

# 2. SMOTE z różnymi strategiami (docelowa liczba próbek klasy 1)
smote_targets = {
    'SMOTE_500': 500,
    'SMOTE_750': 750,
    'SMOTE_1000': 1000,
    'SMOTE_1500': 1500,
    'SMOTE_full': n_majority  # pełne wyrównanie do klasy 0
}

for target_name, target_count in smote_targets.items():
    if target_name == 'SMOTE_full':
        resamplers[target_name] = SMOTE(random_state=42)
    else:
        resamplers[target_name] = SMOTE(sampling_strategy={1: target_count}, random_state=42)

# 3. SMOTE + Undersampling (połączenie)
combined_targets = {
    'SMOTE+Under_500': 500,
    'SMOTE+Under_750': 750,
    'SMOTE+Under_1000': 1000,
    'SMOTE+Under_1500': 1500
}

for target_name, target_count in combined_targets.items():
    resamplers[target_name] = ImbPipeline([
        ('smote', SMOTE(sampling_strategy={1: target_count}, random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy={0: target_count}, random_state=42))
    ])

# Słownik przechowujący wyniki dla wszystkich klasyfikatorów
results = {
    clf_name: {method: {'bac': [], 'recall': [], 'precision': [], 'f1': []} for method in resamplers.keys()}
    for clf_name in models.keys()
}

################################################################################################
# modelowanie

print("\n" + "=" * 80)
print("ROZPOCZĘCIE EKSPERYMENTU")
print("=" * 80)

# Iteracja po metodach resamplingu
for method_name, resampler in resamplers.items():
    
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
        
        # trenowanie wszystkich modeli
        for clf_name, model in models.items():
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test)
            
            # zapis metryk
            results[clf_name][method_name]['bac'].append(balanced_accuracy_score(y_test, y_pred))
            results[clf_name][method_name]['recall'].append(recall_score(y_test, y_pred))
            results[clf_name][method_name]['precision'].append(precision_score(y_test, y_pred))
            results[clf_name][method_name]['f1'].append(f1_score(y_test, y_pred))
    
    # wyświetlenie bieżących wyników TYLKO dla Decision Tree
    dt_bac = results['DecisionTree'][method_name]['bac']
    dt_recall = results['DecisionTree'][method_name]['recall']
    dt_precision = results['DecisionTree'][method_name]['precision']
    dt_f1 = results['DecisionTree'][method_name]['f1']
    
    print(f"  Wyniki (Decision Tree):")
    print(f"    BAC: {np.mean(dt_bac):.3f} ± {np.std(dt_bac):.3f}")
    print(f"    Recall: {np.mean(dt_recall):.3f} ± {np.std(dt_recall):.3f}")
    print(f"    Precision: {np.mean(dt_precision):.3f} ± {np.std(dt_precision):.3f}")
    print(f"    F1 Score: {np.mean(dt_f1):.3f} ± {np.std(dt_f1):.3f}")

################################################################################################
# podsumowanie wyników (tabela porównawcza dla Decision Tree)

print("\n" + "=" * 80)
print("PODSUMOWANIE - PORÓWNANIE WSZYSTKICH METOD RESAMPLINGU (DLA DECISION TREE)")
print("=" * 80)

print("\n" + "-" * 100)
print(f"{'Metoda':<20} {'BAC':<18} {'Recall':<18} {'Precision':<18} {'F1 Score':<18}")
print("-" * 100)

for method_name in resamplers.keys():
    bac_mean = np.mean(results['DecisionTree'][method_name]['bac'])
    bac_std = np.std(results['DecisionTree'][method_name]['bac'])
    recall_mean = np.mean(results['DecisionTree'][method_name]['recall'])
    recall_std = np.std(results['DecisionTree'][method_name]['recall'])
    precision_mean = np.mean(results['DecisionTree'][method_name]['precision'])
    precision_std = np.std(results['DecisionTree'][method_name]['precision'])
    f1_mean = np.mean(results['DecisionTree'][method_name]['f1'])
    f1_std = np.std(results['DecisionTree'][method_name]['f1'])
    
    print(f"{method_name:<20} {bac_mean:.3f}±{bac_std:.3f}   {recall_mean:.3f}±{recall_std:.3f}   {precision_mean:.3f}±{precision_std:.3f}   {f1_mean:.3f}±{f1_std:.3f}")

print("-" * 100)

################################################################################################
# wnioski - wskazanie najlepszej metody (dla Decision Tree)

print("\n" + "=" * 80)
print("WNIOSKI (DECISION TREE)")
print("=" * 80)

# Znalezienie najlepszej metody dla DT (najwyższe BAC)
best_dt_method = max(resamplers.keys(), key=lambda x: np.mean(results['DecisionTree'][x]['bac']))
best_dt_bac = np.mean(results['DecisionTree'][best_dt_method]['bac'])

print(f"\nNajlepsza metoda resamplingu: {best_dt_method}")
print(f"Najwyższe średnie BAC: {best_dt_bac:.3f}")

# Porównanie z Undersampling (baseline) dla DT
baseline_bac = np.mean(results['DecisionTree']['Undersampling']['bac'])
if best_dt_method == 'Undersampling':
    print(f"\nUndersampling (BAC={baseline_bac:.3f}) okazał się najlepszą metodą.")
    print("Żaden poziom oversamplingu ani połączenie z undersamplingiem nie poprawiły wyników.")
else:
    print(f"\n{best_dt_method} (BAC={best_dt_bac:.3f}) jest lepszy od Undersamplingu (BAC={baseline_bac:.3f})")
    print(f"Różnica: {best_dt_bac - baseline_bac:.3f}")

# Ranking metod według BAC dla DT
print("\nRanking metod według BAC (od najlepszej do najgorszej):")
ranking = sorted(resamplers.keys(), key=lambda x: np.mean(results['DecisionTree'][x]['bac']), reverse=True)
for i, method in enumerate(ranking, 1):
    bac_mean = np.mean(results['DecisionTree'][method]['bac'])
    print(f"  {i}. {method}: {bac_mean:.3f}")

# Dodatkowe podsumowanie - najlepsza metoda z pełnym wyrównaniem 1:1 dla DT
print("\n" + "-" * 80)
print("Porównanie metod dających pełne wyrównanie 1:1:")
print("-" * 80)

balanced_methods = ['Undersampling', 'SMOTE_full'] + list(combined_targets.keys())
for method in balanced_methods:
    if method in resamplers:
        bac_mean = np.mean(results['DecisionTree'][method]['bac'])
        n_samples = None
        if method == 'Undersampling':
            n_samples = n_minority
        elif method == 'SMOTE_full':
            n_samples = n_majority
        elif method.startswith('SMOTE+Under'):
            n_samples = combined_targets[method]
        print(f"  {method}: BAC = {bac_mean:.3f} (zbiór zbalansowany, {n_samples}:{n_samples} próbek)")

################################################################################################
# NAJLEPSZE METODY I BAC DLA WSZYSTKICH KLASYFIKATORÓW

print("\n" + "=" * 80)
print("NAJLEPSZE WYNIKI (BAC) DLA WSZYSTKICH TESTOWANYCH KLASYFIKATORÓW")
print("=" * 80)

for clf_name in models.keys():
    best_method = max(resamplers.keys(), key=lambda m: np.mean(results[clf_name][m]['bac']))
    best_bac = np.mean(results[clf_name][best_method]['bac'])
    
    print(f"\nKlasyfikator: {clf_name.upper()}")
    print(f"  Najlepsza metoda: {best_method}")
    print(f"  Najwyższe BAC:    {best_bac:.3f}")

print("\n" + "=" * 80)