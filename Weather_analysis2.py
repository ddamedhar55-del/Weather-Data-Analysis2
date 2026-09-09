import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# WEATHER DATA ANALYSIS PROJECT
# =====================================================

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = "dataset/weatherHistory.csv"

df = pd.read_csv(file_path)

print("\n========== WEATHER DATA ANALYSIS ==========\n")

print("Dataset loaded successfully!")

# =====================================================
# 2. BASIC INFORMATION
# =====================================================

print("\n----- First 5 Rows -----")
print(df.head())

print("\n----- Dataset Shape -----")
print(df.shape)

print("\n----- Column Names -----")
print(df.columns.tolist())

print("\n----- Dataset Information -----")
print(df.info())

print("\n----- Statistical Summary -----")
print(df.describe())

# =====================================================
# 3. CHECK MISSING VALUES
# =====================================================

print("\n----- Missing Values -----")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDuplicates removed successfully.")

# =====================================================
# 4. IDENTIFY NUMERICAL COLUMNS
# =====================================================

numeric_columns = df.select_dtypes(include=np.number).columns

print("\n----- Numerical Columns -----")
print(numeric_columns.tolist())

# =====================================================
# 5. TEMPERATURE DISTRIBUTION
# =====================================================

if len(numeric_columns) > 0:

    temperature_column = None

    for column in numeric_columns:
        if "temp" in column.lower():
            temperature_column = column
            break

    if temperature_column:

        plt.figure(figsize=(10, 6))

        sns.histplot(
            df[temperature_column].dropna(),
            bins=20,
            kde=True
        )

        plt.title("Temperature Distribution")
        plt.xlabel("Temperature")
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig("graphs/temperature_distribution.png")
        plt.show()

# =====================================================
# 6. NUMERICAL CORRELATION
# =====================================================

if len(numeric_columns) >= 2:

    plt.figure(figsize=(10, 7))

    correlation = df[numeric_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Weather Data Correlation")
    plt.tight_layout()

    plt.savefig("graphs/correlation_heatmap.png")
    plt.show()

# =====================================================
# 7. TEMPERATURE TREND
# =====================================================

temperature_column = None

for column in df.columns:

    if "temp" in column.lower():

        if pd.api.types.is_numeric_dtype(df[column]):

            temperature_column = column
            break

if temperature_column:

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.index,
        df[temperature_column]
    )

    plt.title("Temperature Trend")
    plt.xlabel("Days / Records")
    plt.ylabel("Temperature")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("graphs/temperature_trend.png")
    plt.show()

# =====================================================
# 8. RAINFALL ANALYSIS
# =====================================================

rainfall_column = None

for column in df.columns:

    if "rain" in column.lower():

        if pd.api.types.is_numeric_dtype(df[column]):

            rainfall_column = column
            break

if rainfall_column:

    plt.figure(figsize=(12, 6))

    plt.bar(
        df.index,
        df[rainfall_column]
    )

    plt.title("Rainfall Analysis")
    plt.xlabel("Days / Records")
    plt.ylabel("Rainfall")

    plt.tight_layout()

    plt.savefig("graphs/rainfall_analysis.png")
    plt.show()

# =====================================================
# 9. HUMIDITY ANALYSIS
# =====================================================

humidity_column = None

for column in df.columns:

    if "humid" in column.lower():

        if pd.api.types.is_numeric_dtype(df[column]):

            humidity_column = column
            break

if humidity_column:

    plt.figure(figsize=(12, 6))

    plt.plot(
        df.index,
        df[humidity_column]
    )

    plt.title("Humidity Trend")
    plt.xlabel("Days / Records")
    plt.ylabel("Humidity")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("graphs/humidity_trend.png")
    plt.show()

# =====================================================
# 10. WEATHER STATISTICS
# =====================================================

print("\n========== IMPORTANT STATISTICS ==========")

if temperature_column:

    print(
        "\nAverage Temperature:",
        round(df[temperature_column].mean(), 2)
    )

    print(
        "Maximum Temperature:",
        round(df[temperature_column].max(), 2)
    )

    print(
        "Minimum Temperature:",
        round(df[temperature_column].min(), 2)
    )

if rainfall_column:

    print(
        "\nTotal Rainfall:",
        round(df[rainfall_column].sum(), 2)
    )

if humidity_column:

    print(
        "\nAverage Humidity:",
        round(df[humidity_column].mean(), 2)
    )

# =====================================================
# 11. SAVE CLEAN DATASET
# =====================================================

df.to_csv(
    "dataset/cleaned_weather.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

print("\n========== PROJECT COMPLETED ==========")