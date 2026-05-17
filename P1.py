################################################################################################
#importy

import numpy as np
import matplotlib as plt
import pandas as pd

################################################################################################
#wczytanie danych

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

################################################################################################
#typy zmiennych

print("=" * 80)
print("1. TYPY ZMIENNYCH (LICZBOWE vs KATEGORYCZNE)")
print("=" * 80)
print("\n--- Podsumowanie typów ---")
print(data.dtypes)
print("\n--- Zmienne liczbowe (numeric) ---")
numeric_vars = data.select_dtypes(include=[np.number]).columns.tolist()
print(f"Liczba zmiennych numerycznych: {len(numeric_vars)}")
print(f"Lista: {numeric_vars}")

print("\n--- Zmienne kategoryczne (object/categorical) ---")
categorical_vars = data.select_dtypes(include=['object', 'string']).columns.tolist()
print(f"Liczba zmiennych kategorycznych: {len(categorical_vars)}")
print(f"Lista: {categorical_vars}")
print("\n")

################################################################################################
#braki danych

print("=" * 80)
print("2. BRAKI DANYCH (MISSING VALUES)")
print("=" * 80)

missing_total = data.isnull().sum()
missing_percent = (missing_total / len(data)) * 100

missing_data = pd.DataFrame({
    'Brakujące wartości': missing_total,
    'Procent (%)': missing_percent
})
missing_data = missing_data[missing_data['Brakujące wartości'] > 0].sort_values('Brakujące wartości', ascending=False)

print("\n--- Wszystkie kolumny z brakami danych ---")
print(missing_data)

################################################################################################
#statystyki opisowe

print("=" * 80)
print("3. STATYSTYKI OPISOWE (ZMIENNE NUMERYCZNE)")
print("=" * 80)
print(data[numeric_vars].describe().round(2))
print("\n")

################################################################################################
#udar wyniki

print("=" * 80)
print("4. BALANS KLAS (zmienna docelowa: stroke)")
print("=" * 80)
stroke_counts = data['stroke'].value_counts()
stroke_percent = (stroke_counts / len(data)) * 100
for val in stroke_counts.index:
    label = "Udar" if val == 1 else "Brak udaru"
    print(f"{label} ({val}): {stroke_counts[val]} ({stroke_percent[val]:.2f}%)")
print("\n")

