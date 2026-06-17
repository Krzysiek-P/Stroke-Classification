import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Przepisane dane z wynikami do wykresu
data = {
    'Klasyfikator': ['RandomForest', 'RandomForest', 'RandomForest', 
                     'KNN', 'KNN', 'KNN', 
                     'GaussianNB', 'GaussianNB', 'GaussianNB'],
    'Eksperyment': ['Baseline', 'Eksperyment 1', 'Eksperyment 2', 
                    'Baseline', 'Eksperyment 1', 'Eksperyment 2', 
                    'Baseline', 'Eksperyment 1', 'Eksperyment 2'],
    'BAC': [0.500, 0.760, 0.768, 
            0.509, 0.651, 0.659, 
            0.622, 0.641, 0.623]
}

df = pd.DataFrame(data)

# Tworzenie wykresu
plt.figure(figsize=(12, 7))
chart = sns.barplot(data=df, x='Klasyfikator', y='BAC', hue='Eksperyment', palette='viridis')

# Dodanie etykiet wartości na słupkach
for container in chart.containers:
    chart.bar_label(container, fmt='%.3f', padding=3)

plt.title('Porównanie skuteczności (BAC) klasyfikatorów w różnych eksperymentach', fontsize=16)
plt.ylabel('Balanced Accuracy (BAC)', fontsize=12)
plt.ylim(0, 1.0)
plt.legend(title='Etap badania')
plt.tight_layout()

plt.savefig('wykresy/wykres_Podsumowanie.png', dpi=300)
plt.close()