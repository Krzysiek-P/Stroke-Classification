################################################################################################
#importy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################################################################
#wczytanie danych

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

################################################################################################
#przygotowanie danych (preprocessing)

#usunięcie kolumny 'id', która nie jest potrzebna do analizy
data = data.drop(columns=['id'])

#usunięcie pojedynczego wiersza "Other" w kolumnie 'gender'
data = data[data['gender'] != 'Other']

#uzupełnienie brakujących wartości w kolumnie 'bmi' medianą - możliwy Wyciek Danych (Data Leakage), jest niewielki, w kolejnych etapach projektu zostanie zaimplementowana bardziej poprawna wersja, bez wycieku danych 
#median_bmi = data['bmi'].median()
#data['bmi'] = data['bmi'].fillna(median_bmi)

#LUB usunięcie wierszy z brakującymi wartościami w kolumnie 'bmi' - procent osób z udarem bez wartości w "bmi" to 16% już stanowiącej tylko ~5% klasy mniejszościowej, więc usuwanie tych wierszy może nie być najlepszym rozwiązaniem
#data = data.dropna(subset=['bmi'])

#kodowanie zmiennych kategorycznych (one-hot encoding)
categorical_vars = data.select_dtypes(include=['object', 'string']).columns.tolist()
data = pd.get_dummies(data, columns=categorical_vars, dtype=int, drop_first=False) #drop_first=False, aby zachować wszystkie kategorie (może być przydatne do analizy i modelowania) - w kolejnych etapach projektu można zdecydować, czy usunąć jedną kategorię (drop_first=True) w celu uniknięcia pułapki zmiennych fikcyjnych (dummy variable trap) w modelach liniowych

#zapisanie przetworzonych danych do nowego pliku CSV
data.to_csv('preprocessed_data.csv', index=False)

################################################################################################
#Dodatkowe wymagane czynności na danych - takie jak: 
# - podział na zbiór treningowy i testowy, 
# - standaryzacja/normalizacja zmiennych numerycznych,
# - uzupelnianie brakujących wartości w kolumnie 'bmi'
#zostaną wykonane w kolejnych etapach projektu