# ==========================================
# E-COMMERCE DATA ANALYSIS PIPELINE
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. LOAD DATA
# ------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset loaded successfully\n")
    return df


# ------------------------------------------
# 2. BASIC DESCRIPTION
# ------------------------------------------

def describe_data(df, title="DATASET"):
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    
    print("\nStatistical summary:\n")
    print(df.describe(include='all'))
    
    print("\nSkewness:\n")
    print(df.select_dtypes(include=np.number).skew().sort_values(ascending=False))


# ------------------------------------------
# 3. DETECT OUTLIERS (IQR)
# ------------------------------------------

def detect_outliers_iqr(df, columns):
    outliers_summary = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outliers_summary[col] = len(outliers)
    
    print("\nOutliers detected:")
    for k, v in outliers_summary.items():
        print(f"{k}: {v}")
    
    return outliers_summary


# ------------------------------------------
# 4. CLEAN DATASET
# ------------------------------------------

def clean_data(df):
    df_clean = df.copy()
    
    # ---- Missing values ----
    df_clean['region'] = df_clean['region'].fillna("Desconocido")
    
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # ---- Fix inconsistencies ----
    if 'discount_rate' in df_clean.columns:
        df_clean['discount_rate'] = df_clean['discount_rate'].clip(0, 1)
    
    if 'net_revenue' in df_clean.columns:
        df_clean.loc[df_clean['net_revenue'] < 0, 'net_revenue'] = 0
    
    # ---- Remove outliers using IQR ----
    columns_to_clean = ['income_monthly', 'unit_price', 'shipping_days']
    
    for col in columns_to_clean:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    
    return df_clean


# ------------------------------------------
# 5. VISUALIZATION
# ------------------------------------------

def plot_distribution(df, column):
    plt.figure()
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_boxplot(df, column):
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


# ------------------------------------------
# 6. BUSINESS ANALYSIS
# ------------------------------------------

def business_analysis(df):
    print("\nTop 5 Product Categories by Revenue:")
    print(df.groupby('product_category')['net_revenue']
          .sum()
          .sort_values(ascending=False)
          .head())
    
    print("\nRevenue by Channel:")
    print(df.groupby('channel')['net_revenue']
          .sum()
          .sort_values(ascending=False))
    
    print("\nReturn Rate:")
    print(df['return_flag'].mean())


# ------------------------------------------
# 7. MAIN PIPELINE
# ------------------------------------------

def main():
    path = "data/e-commerce (3).csv"
    
    df = load_data(path)
    
    # BEFORE
    describe_data(df, title="BEFORE CLEANING")
    detect_outliers_iqr(df, ['income_monthly', 'unit_price', 'shipping_days'])
    
    # CLEAN
    df_clean = clean_data(df)
    
    # AFTER
    describe_data(df_clean, title="AFTER CLEANING")
    
    # Business insights
    business_analysis(df_clean)
    
    # Example plots
    plot_distribution(df_clean, 'net_revenue')
    plot_boxplot(df_clean, 'income_monthly')


if __name__ == "__main__":
    main()
