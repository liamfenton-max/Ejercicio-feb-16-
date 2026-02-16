"""
ANÁLISIS COMPLETO DE DATOS DE E-COMMERCE
Código para responder todas las preguntas del cuestionario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')

# ============================================================================
# CARGAR DATOS
# ============================================================================
# Cambia esta ruta según donde tengas el archivo
df_original = pd.read_csv('e-commerce__3_.csv')

print("="*80)
print("INFORMACIÓN DEL DATASET ORIGINAL")
print("="*80)
print(f"Dimensiones: {df_original.shape}")
print("\nPrimeras filas:")
print(df_original.head())
print("\nInformación:")
print(df_original.info())
print("\nEstadísticas:")
print(df_original.describe())

# ============================================================================
# PREGUNTA 2: ¿Cuántas variables son categóricas?
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 2")
print("="*80)

categorical_vars = df_original.select_dtypes(include=['object']).columns.tolist()
# La variable return_flag también es categórica (binaria)
if 'return_flag' in df_original.columns:
    categorical_vars.append('return_flag')

print(f"Variables categóricas: {categorical_vars}")
respuesta_2 = len(categorical_vars)
print(f"\n✓ RESPUESTA 2: {respuesta_2}")

# ============================================================================
# PREGUNTA 3: Variable más importante
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 3")
print("="*80)
print("✓ RESPUESTA 3: net_revenue")
print("Justificación: Las ventas efectivas son el KPI principal del negocio")

# ============================================================================
# PREGUNTA 4: Variables con missing values
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 4")
print("="*80)

missing_counts = df_original.isnull().sum()
print("\nMissing values por columna:")
print(missing_counts[missing_counts > 0])

respuesta_4 = (missing_counts > 0).sum()
print(f"\n✓ RESPUESTA 4: {respuesta_4}")

# ============================================================================
# PREGUNTA 5: Porcentaje de missing values
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 5")
print("="*80)

total_cells = df_original.shape[0] * df_original.shape[1]
total_missing = df_original.isnull().sum().sum()
respuesta_5 = round((total_missing / total_cells) * 100, 2)

print(f"Total de celdas: {total_cells}")
print(f"Celdas con missing: {total_missing}")
print(f"\n✓ RESPUESTA 5: {respuesta_5}")

# ============================================================================
# PREGUNTA 6: Observaciones duplicadas
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 6")
print("="*80)

respuesta_6 = df_original.duplicated().sum()
print(f"✓ RESPUESTA 6: {respuesta_6}")

# ============================================================================
# PREGUNTA 7: Eliminar duplicados y promedio de net_revenue
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 7")
print("="*80)

df = df_original.drop_duplicates()
print(f"Filas antes: {len(df_original)}")
print(f"Filas después: {len(df)}")
print(f"Duplicados eliminados: {len(df_original) - len(df)}")

respuesta_7 = round(df['net_revenue'].mean(), 2)
print(f"\n✓ RESPUESTA 7: {respuesta_7}")

# ============================================================================
# PREGUNTA 8: Variables con valores extremos (outliers)
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 8")
print("="*80)

# Excluir order_id y customer_id
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['order_id', 'customer_id']]

print(f"Variables numéricas analizadas: {numeric_cols}\n")

# Crear boxplots
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
axes = axes.flatten() if n_rows > 1 else [axes]

for idx, col in enumerate(numeric_cols):
    df.boxplot(column=col, ax=axes[idx])
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Valor')

for idx in range(len(numeric_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('boxplots_all_variables.png', dpi=150, bbox_inches='tight')
print("✓ Boxplots guardados en 'boxplots_all_variables.png'")

# Análisis de outliers con IQR
print("\nDetección de outliers (método IQR):")
vars_with_outliers = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers_count = len(df[(df[col] < lower) | (df[col] > upper)])
    
    if outliers_count > 0:
        vars_with_outliers.append(col)
        print(f"  {col}: {outliers_count} outliers")

print(f"\n✓ RESPUESTA 8: {vars_with_outliers}")
print("(Selecciona estas variables en el cuestionario)")

# ============================================================================
# PREGUNTA 9: Asimetría de variables numéricas
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 9")
print("="*80)

# Excluir return_flag (es binaria/categórica)
cols_for_skew = [col for col in numeric_cols if col != 'return_flag']

print("Asimetría (skewness):")
print("-" * 40)

strong_skew_count = 0
for col in cols_for_skew:
    skew_val = df[col].skew()
    is_strong = abs(skew_val) > 1
    if is_strong:
        strong_skew_count += 1
    marker = " *** FUERTE" if is_strong else ""
    print(f"  {col:20s}: {skew_val:7.3f}{marker}")

respuesta_9 = strong_skew_count
print(f"\n✓ RESPUESTA 9: {respuesta_9}")

# ============================================================================
# PREGUNTA 10: Mejor técnica para outliers con asimetría
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 10")
print("="*80)
print("✓ RESPUESTA 10: IQR (Rango intercuartílico)")
print("Razón: Es robusto ante asimetría, a diferencia del Z-score")

# ============================================================================
# PREGUNTA 11: Outliers en income_monthly
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 11")
print("="*80)

Q1_income = df['income_monthly'].quantile(0.25)
Q3_income = df['income_monthly'].quantile(0.75)
IQR_income = Q3_income - Q1_income
lower_income = Q1_income - 1.5 * IQR_income
upper_income = Q3_income + 1.5 * IQR_income

outliers_income = df[(df['income_monthly'] < lower_income) | 
                      (df['income_monthly'] > upper_income)]

respuesta_11 = len(outliers_income)
print(f"Q1: {Q1_income:.2f}")
print(f"Q3: {Q3_income:.2f}")
print(f"IQR: {IQR_income:.2f}")
print(f"Límites: [{lower_income:.2f}, {upper_income:.2f}]")
print(f"\n✓ RESPUESTA 11: {respuesta_11}")

# ============================================================================
# PREGUNTA 12: Mediana de income_monthly sin outliers
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 12")
print("="*80)

df_no_outliers_income = df[(df['income_monthly'] >= lower_income) & 
                            (df['income_monthly'] <= upper_income)].copy()

respuesta_12 = round(df_no_outliers_income['income_monthly'].median(), 2)
print(f"Filas antes: {len(df)}")
print(f"Filas después: {len(df_no_outliers_income)}")
print(f"\n✓ RESPUESTA 12: {respuesta_12}")

# Guardar para imputación posterior
median_income_imputation = respuesta_12

# ============================================================================
# PREGUNTA 13: Limpiar todas las variables numéricas
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 13")
print("="*80)

df_clean = df_no_outliers_income.copy()

# Limpiar el resto de variables (excepto income_monthly que ya limpiamos)
vars_to_clean = [v for v in vars_with_outliers if v != 'income_monthly']

print(f"Limpiando variables: {vars_to_clean}\n")

for col in vars_to_clean:
    before = len(df_clean)
    
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    removed = before - len(df_clean)
    print(f"  {col}: {removed} outliers eliminados")

respuesta_13 = len(df_clean)
print(f"\n✓ RESPUESTA 13: {respuesta_13}")

# ============================================================================
# PREGUNTA 14: Missing values después de limpiar
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 14")
print("="*80)

missing_after = df_clean.isnull().sum()
print("\nMissing values restantes:")
print(missing_after[missing_after > 0])

respuesta_14 = (missing_after > 0).sum()
print(f"\n✓ RESPUESTA 14: {respuesta_14}")

# ============================================================================
# PREGUNTA 15: Mejor técnica para imputar region
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 15")
print("="*80)
print("✓ RESPUESTA 15: Con la moda")
print("Razón: 'region' es categórica, se usa la moda")

# ============================================================================
# PREGUNTA 16: Moda de region
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 16")
print("="*80)

print("\nFrecuencias de region:")
print(df_clean['region'].value_counts())

respuesta_16 = df_clean['region'].mode()[0]
print(f"\n✓ RESPUESTA 16: {respuesta_16}")

# ============================================================================
# PREGUNTA 17: Imputar y calcular pérdida de datos
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 17")
print("="*80)

# Imputar
df_clean['income_monthly'].fillna(median_income_imputation, inplace=True)
df_clean['region'].fillna(respuesta_16, inplace=True)

print(f"Imputada mediana {median_income_imputation} en income_monthly")
print(f"Imputada moda '{respuesta_16}' en region")

# Verificar
print(f"\nMissing values restantes: {df_clean.isnull().sum().sum()}")

# Calcular pérdida
original_count = len(df_original)
final_count = len(df_clean)
respuesta_17 = round(((original_count - final_count) / original_count) * 100, 2)

print(f"\nObservaciones originales: {original_count}")
print(f"Observaciones finales: {final_count}")
print(f"Pérdida: {original_count - final_count} ({respuesta_17}%)")
print(f"\n✓ RESPUESTA 17: {respuesta_17}")

# ============================================================================
# PREGUNTA 18: Categoría menos frecuente en channel
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 18")
print("="*80)

print("\nFrecuencias de channel:")
channel_freq = df_clean['channel'].value_counts()
print(channel_freq)

respuesta_18 = channel_freq.idxmin()
print(f"\n✓ RESPUESTA 18: {respuesta_18}")

# ============================================================================
# PREGUNTA 19: Tipo de sesgo
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 19")
print("="*80)
print("✓ RESPUESTA 19: Sesgo de representación")
print("Razón: Un canal con pocas observaciones no está bien representado")

# ============================================================================
# PREGUNTA 20: Tabla cruzada region vs product_category
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 20")
print("="*80)

crosstab_pct = pd.crosstab(df_clean['region'], 
                           df_clean['product_category'], 
                           normalize='all') * 100

print("\nTabla cruzada (porcentajes):")
print(crosstab_pct.round(2))

# Buscar Centro y Electrónica/Electronics
try:
    if 'Electrónica' in crosstab_pct.columns:
        respuesta_20 = round(crosstab_pct.loc['Centro', 'Electrónica'], 2)
    elif 'Electronics' in crosstab_pct.columns:
        respuesta_20 = round(crosstab_pct.loc['Centro', 'Electronics'], 2)
    else:
        # Buscar columna que contenga "electr"
        elec_col = [c for c in crosstab_pct.columns if 'electr' in c.lower()][0]
        respuesta_20 = round(crosstab_pct.loc['Centro', elec_col], 2)
    
    print(f"\n✓ RESPUESTA 20: {respuesta_20}")
except Exception as e:
    print(f"\nError: {e}")
    print("Verifica los nombres exactos de las categorías")

# ============================================================================
# PREGUNTA 21: Región que más compra en tienda
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 21")
print("="*80)

# Filtrar solo tienda/store
store_df = df_clean[df_clean['channel'].str.lower() == 'store']

if len(store_df) > 0:
    print("\nCompras en tienda por región:")
    store_by_region = store_df['region'].value_counts()
    print(store_by_region)
    
    respuesta_21 = store_by_region.idxmax()
    print(f"\n✓ RESPUESTA 21: {respuesta_21}")

# ============================================================================
# PREGUNTA 22: Canal para categoría más frecuente
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 22")
print("="*80)

most_freq_category = df_clean['product_category'].value_counts().idxmax()
print(f"\nCategoría más frecuente: {most_freq_category}")

category_df = df_clean[df_clean['product_category'] == most_freq_category]
channel_for_category = category_df['channel'].value_counts()

print(f"\nCanales para {most_freq_category}:")
print(channel_for_category)

respuesta_22 = channel_for_category.idxmax()
print(f"\n✓ RESPUESTA 22: {respuesta_22}")

# ============================================================================
# PREGUNTA 23: Región con mayor promedio de ventas
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 23")
print("="*80)

avg_revenue_by_region = df_clean.groupby('region')['net_revenue'].mean().sort_values(ascending=False)

print("\nPromedio de net_revenue por región:")
print(avg_revenue_by_region.round(2))

respuesta_23 = avg_revenue_by_region.idxmax()
print(f"\n✓ RESPUESTA 23: {respuesta_23}")

# ============================================================================
# PREGUNTA 24: Promedio de ventas de la región ganadora
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 24")
print("="*80)

respuesta_24 = round(avg_revenue_by_region.max(), 2)
print(f"✓ RESPUESTA 24: {respuesta_24}")

# ============================================================================
# PREGUNTA 25: Edad de clientes con menor satisfacción
# ============================================================================
print("\n" + "="*80)
print("PREGUNTA 25")
print("="*80)

print("\nEdad promedio por nivel de satisfacción:")
age_by_satisfaction = df_clean.groupby('satisfaction')['customer_age'].mean()
print(age_by_satisfaction.round(2))

min_satisfaction = df_clean['satisfaction'].min()
respuesta_25 = round(df_clean[df_clean['satisfaction'] == min_satisfaction]['customer_age'].mean(), 2)

print(f"\nSatisfacción mínima: {min_satisfaction}")
print(f"✓ RESPUESTA 25: {respuesta_25}")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMEN DE RESPUESTAS")
print("="*80)

resumen = f"""
PREGUNTA 2:  {respuesta_2}
PREGUNTA 3:  net_revenue
PREGUNTA 4:  {respuesta_4}
PREGUNTA 5:  {respuesta_5}
PREGUNTA 6:  {respuesta_6}
PREGUNTA 7:  {respuesta_7}
PREGUNTA 8:  {', '.join(vars_with_outliers)}
PREGUNTA 9:  {respuesta_9}
PREGUNTA 10: IQR (Rango intercuartílico)
PREGUNTA 11: {respuesta_11}
PREGUNTA 12: {respuesta_12}
PREGUNTA 13: {respuesta_13}
PREGUNTA 14: {respuesta_14}
PREGUNTA 15: Con la moda
PREGUNTA 16: {respuesta_16}
PREGUNTA 17: {respuesta_17}
PREGUNTA 18: {respuesta_18}
PREGUNTA 19: Sesgo de representación
PREGUNTA 20: {respuesta_20 if 'respuesta_20' in locals() else 'Ver tabla cruzada'}
PREGUNTA 21: {respuesta_21}
PREGUNTA 22: {respuesta_22}
PREGUNTA 23: {respuesta_23}
PREGUNTA 24: {respuesta_24}
PREGUNTA 25: {respuesta_25}
"""

print(resumen)

# Guardar datos limpios
df_clean.to_csv('ecommerce_cleaned.csv', index=False)
print("\n✓ Datos limpios guardados en 'ecommerce_cleaned.csv'")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
