"""
Exploratory Data Analysis - AIMAI Project
Author: Mirnes
Date: 2026-02-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 50)
    print("🤖 AIMAI - Exploratory Data Analysis")
    print("=" * 50)
    
    # TODO: Dodaj kod za učitavanje i analizu podataka
    print("\n📝 Status: Notebook je spreman za analizu!")
    print("📁 Putanja za podatke: data/raw/")
    print("💾 Sačuvaj rezultate u: data/processed/")
    
    # Primer: Kreiraj dummy podatke za testiranje
    print("\n📊 Kreiram test podatke...")
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    print("\n📋 Info o podacima:")
    print(df.info())
    print("\n📈 Opis statistike:")
    print(df.describe())
    
    # ===== DODAJ OVO ISPOD =====
    print("\n🎨 Kreiram vizuelizaciju...")
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='feature1', y='feature2', hue='target', palette='viridis')
    plt.title('Feature1 vs Feature2 po Targetu')
    
    # Distribution plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x='feature1', hue='target', multiple='stack', bins=20)
    plt.title('Distribucija Feature1')
    
    plt.tight_layout()
    plt.savefig('notebooks/exploratory/analysis_plot.png')
    print("✅ Vizuelizacija sačuvana u notebooks/exploratory/analysis_plot.png")
    # ===== KRAJ DODATKA =====
    
    print("\n✅ Analiza završena!")

if __name__ == "__main__":
    main()