################################################################################################
# WIZUALIZACJA ZALEŻNOŚCI
################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# usunięcie wiersza other z gender
data = data[data['gender'] != 'Other']

print("generowanie lepszych wykresów zależności...")
print("=" * 60)

################################################################################################
# WYKRES 1: MACIERZ KORELACJI Z OZNACZENIAMI
################################################################################################
fig, ax = plt.subplots(figsize=(10, 8))
numeric_for_corr = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
corr_matrix = data[numeric_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={'size': 12})
ax.set_title('macierz korelacji między cechami numerycznymi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres_korelacji.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres korelacji zapisany")

################################################################################################
# WYKRES 2: WIEK A UDAR - GĘSTOŚĆ ROZKŁADU (KDE) ZAMIAST BOXPLOT
################################################################################################
fig, ax = plt.subplots(figsize=(12, 6))
for stroke_val, label, color in [(0, 'brak udaru', '#2ecc71'), (1, 'udar', '#e74c3c')]:
    subset = data[data['stroke'] == stroke_val]['age'].dropna()
    sns.kdeplot(subset, label=label, color=color, linewidth=2.5, ax=ax, shade=True, alpha=0.4)
ax.set_xlabel('wiek (lata)', fontsize=12)
ax.set_ylabel('gęstość prawdopodobieństwa', fontsize=12)
ax.set_title('rozkład wieku pacjentów z udarem i bez udaru', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_wiek_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres wieku (kde) zapisany")

################################################################################################
# WYKRES 3: POZIOM GLUKOZY A UDAR - GĘSTOŚĆ ROZKŁADU
################################################################################################
fig, ax = plt.subplots(figsize=(12, 6))
for stroke_val, label, color in [(0, 'brak udaru', '#2ecc71'), (1, 'udar', '#e74c3c')]:
    subset = data[data['stroke'] == stroke_val]['avg_glucose_level'].dropna()
    sns.kdeplot(subset, label=label, color=color, linewidth=2.5, ax=ax, shade=True, alpha=0.4)
ax.set_xlabel('średni poziom glukozy (mg/dl)', fontsize=12)
ax.set_ylabel('gęstość prawdopodobieństwa', fontsize=12)
ax.set_title('rozkład poziomu glukozy u pacjentów z udarem i bez udaru', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_glukoza_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres glukozy (kde) zapisany")

################################################################################################
# WYKRES 4: BMI A UDAR - GĘSTOŚĆ ROZKŁADU
################################################################################################
fig, ax = plt.subplots(figsize=(12, 6))
data_bmi = data.dropna(subset=['bmi'])
for stroke_val, label, color in [(0, 'brak udaru', '#2ecc71'), (1, 'udar', '#e74c3c')]:
    subset = data_bmi[data_bmi['stroke'] == stroke_val]['bmi']
    sns.kdeplot(subset, label=label, color=color, linewidth=2.5, ax=ax, shade=True, alpha=0.4)
ax.set_xlabel('wskaźnik bmi (kg/m²)', fontsize=12)
ax.set_ylabel('gęstość prawdopodobieństwa', fontsize=12)
ax.set_title('rozkład wskaźnika bmi u pacjentów z udarem i bez udaru', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_bmi_kde.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres bmi (kde) zapisany")

################################################################################################
# WYKRES 5: WIEK A GLUKOZA Z PODZIAŁEM NA UDAR (scatter z linią trendu)
################################################################################################
fig, ax = plt.subplots(figsize=(12, 8))
data_clean = data.dropna(subset=['age', 'avg_glucose_level'])
colors = {0: '#2ecc71', 1: '#e74c3c'}
labels = {0: 'brak udaru', 1: 'udar'}
for stroke_val in [0, 1]:
    subset = data_clean[data_clean['stroke'] == stroke_val]
    ax.scatter(subset['age'], subset['avg_glucose_level'], 
               c=colors[stroke_val], label=labels[stroke_val], 
               alpha=0.5, s=40, edgecolors='black', linewidth=0.3)
    # dodanie linii trendu
    z = np.polyfit(subset['age'], subset['avg_glucose_level'], 1)
    p = np.poly1d(z)
    ax.plot(subset['age'].sort_values(), p(subset['age'].sort_values()), 
            color=colors[stroke_val], linewidth=2, linestyle='--')
ax.set_xlabel('wiek (lata)', fontsize=12)
ax.set_ylabel('poziom glukozy (mg/dl)', fontsize=12)
ax.set_title('zależność między wiekiem a poziomem glukozy (kropki = pacjenci, linie = trend)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_wiek_glukoza_scatter_z_trendem.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres wiek vs glukoza z trendem zapisany")

################################################################################################
# WYKRES 6: WIEK A BMI Z PODZIAŁEM NA UDAR
################################################################################################
fig, ax = plt.subplots(figsize=(12, 8))
data_bmi_clean = data.dropna(subset=['age', 'bmi'])
for stroke_val in [0, 1]:
    subset = data_bmi_clean[data_bmi_clean['stroke'] == stroke_val]
    ax.scatter(subset['age'], subset['bmi'], 
               c=colors[stroke_val], label=labels[stroke_val], 
               alpha=0.5, s=40, edgecolors='black', linewidth=0.3)
    z = np.polyfit(subset['age'], subset['bmi'], 1)
    p = np.poly1d(z)
    ax.plot(subset['age'].sort_values(), p(subset['age'].sort_values()), 
            color=colors[stroke_val], linewidth=2, linestyle='--')
ax.set_xlabel('wiek (lata)', fontsize=12)
ax.set_ylabel('wskaźnik bmi (kg/m²)', fontsize=12)
ax.set_title('zależność między wiekiem a bmi (kropki = pacjenci, linie = trend)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_wiek_bmi_scatter_z_trendem.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres wiek vs bmi z trendem zapisany")

################################################################################################
# WYKRES 7: STOS NADCIŚNIENIA I CHORÓB SERCA (wykresy obok siebie z odsetkami)
################################################################################################
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# nadciśnienie
hyper_by_stroke = data.groupby(['hypertension', 'stroke']).size().unstack()
hyper_percent = hyper_by_stroke.div(hyper_by_stroke.sum(axis=0), axis=1) * 100
hyper_percent.T.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[0].set_title('odsetek pacjentów z nadciśnieniem w zależności od udaru', fontsize=12, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('procent pacjentów (%)', fontsize=11)
axes[0].legend(['brak nadciśnienia', 'nadciśnienie'], fontsize=10)
axes[0].set_ylim(0, 100)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f%%', fontsize=9)
# choroby serca
heart_by_stroke = data.groupby(['heart_disease', 'stroke']).size().unstack()
heart_percent = heart_by_stroke.div(heart_by_stroke.sum(axis=0), axis=1) * 100
heart_percent.T.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1].set_title('odsetek pacjentów z chorobą serca w zależności od udaru', fontsize=12, fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('procent pacjentów (%)', fontsize=11)
axes[1].legend(['brak choroby serca', 'choroba serca'], fontsize=10)
axes[1].set_ylim(0, 100)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f%%', fontsize=9)
plt.suptitle('wpływ chorób współistniejących na ryzyko udaru', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres_choroby_odsetki.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres chorób (odsetki) zapisany")

################################################################################################
# WYKRES 8: STATUS PALENIA A UDAR (odsetki)
################################################################################################
fig, ax = plt.subplots(figsize=(10, 6))
smoke_by_stroke = data.groupby(['smoking_status', 'stroke']).size().unstack()
smoke_percent = smoke_by_stroke.div(smoke_by_stroke.sum(axis=0), axis=1) * 100
smoke_order = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
smoke_percent = smoke_percent.reindex(smoke_order)
smoke_percent.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_title('odsetek pacjentów z danym statusem palenia w zależności od udaru', fontsize=14, fontweight='bold')
ax.set_xlabel('status palenia', fontsize=12)
ax.set_ylabel('procent pacjentów (%)', fontsize=12)
ax.legend(['brak udaru', 'udar'], fontsize=11)
ax.tick_params(axis='x', rotation=15)
ax.set_ylim(0, 100)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('wykres_palenie_odsetki.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres palenia (odsetki) zapisany")

################################################################################################
# WYKRES 9: VIOLIN PLOT - LEPSZA WERSJA BOXPLOTA (pokazuje pełny rozkład)
################################################################################################
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
features = [('age', 'wiek (lata)'), ('avg_glucose_level', 'poziom glukozy (mg/dl)'), ('bmi', 'wskaźnik bmi')]
for idx, (feature, label) in enumerate(features):
    data_f = data.dropna(subset=[feature])
    data_to_plot = [data_f[data_f['stroke']==0][feature], data_f[data_f['stroke']==1][feature]]
    parts = axes[idx].violinplot(data_to_plot, positions=[0, 1], showmeans=False, showmedians=True)
    parts['bodies'][0].set_facecolor('#2ecc71')
    parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor('#e74c3c')
    parts['bodies'][1].set_alpha(0.6)
    axes[idx].set_xticks([0, 1])
    axes[idx].set_xticklabels(['brak udaru', 'udar'])
    axes[idx].set_ylabel(label, fontsize=11)
    axes[idx].set_title(f'{label} - pełny rozkład', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
plt.suptitle('porównanie pełnych rozkładów cech numerycznych między klasami (violin plot)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('wykres_violin_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykresy violin zapisane")

################################################################################################
# WYKRES 10: WIEK A CHOROBY SERCA - ZALEŻNOŚĆ (boxplot z podziałem na choroby serca)
################################################################################################
fig, ax = plt.subplots(figsize=(10, 6))
data_hd = data.dropna(subset=['age'])
positions = [1, 2, 4, 5]
labels = ['brak choroby\nbrak udaru', 'brak choroby\nudar', 'choroba serca\nbrak udaru', 'choroba serca\nudar']
colors_hd = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
boxes = []
for i, (heart, stroke) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
    subset = data_hd[(data_hd['heart_disease']==heart) & (data_hd['stroke']==stroke)]['age']
    if len(subset) > 0:
        bp = ax.boxplot([subset], positions=[positions[i]], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors_hd[i])
        boxes.append(bp)
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel('wiek (lata)', fontsize=12)
ax.set_title('wiek a choroby serca i udar', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('wykres_wiek_choroby_serca.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wykres wiek vs choroby serca zapisany")

print("=" * 60)
print("wszystkie wykresy zostały wygenerowane")
print("")
print("lista plików:")
print("  wykres_korelacji.png - macierz korelacji")
print("  wykres_wiek_kde.png - rozkład wieku (gęstość)")
print("  wykres_glukoza_kde.png - rozkład glukozy (gęstość)")
print("  wykres_bmi_kde.png - rozkład bmi (gęstość)")
print("  wykres_wiek_glukoza_scatter_z_trendem.png - wiek vs glukoza z linią trendu")
print("  wykres_wiek_bmi_scatter_z_trendem.png - wiek vs bmi z linią trendu")
print("  wykres_choroby_odsetki.png - nadciśnienie i choroby serca (odsetki)")
print("  wykres_palenie_odsetki.png - status palenia (odsetki)")
print("  wykres_violin_plots.png - violin plot dla wszystkich cech numerycznych")
print("  wykres_wiek_choroby_serca.png - wiek a choroby serca")