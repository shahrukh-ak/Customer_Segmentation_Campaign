"""
Customer Segmentation for Campaign Optimisation
=================================================
Applies KMeans clustering to a retail marketing dataset to identify
distinct customer segments. Includes feature engineering, outlier
handling, preprocessing, and cluster profiling to support targeted
campaign design.

Dataset: Campaign_data.csv
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load campaign data and display basic shape information."""
    df = pd.read_csv(filepath)
    pd.options.display.max_columns = None
    print(f"Shape: {df.shape}")
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing income and parse Dt_Customer as datetime."""
    initial_len = len(df)
    df = df.dropna()
    print(f"Removed {initial_len - len(df)} rows with missing values. Remaining: {len(df)}")

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, reference_year: int = 2022) -> pd.DataFrame:
    """
    Create derived features:
      - Age and Age_Range from Year_Birth
      - Customer_For (tenure in days)
      - WebConversionRate
      - Total_Spent, Total_Purchases
      - Adults, Dependents, Household_size, Is_Parent
      - Offers_Accepted
    """
    df["Age"] = reference_year - df["Year_Birth"]

    def age_group(age: int) -> str:
        if age <= 24:   return "Young Adults"
        elif age <= 34: return "Early Adulthood"
        elif age <= 44: return "Midlife"
        elif age <= 54: return "Middle Age"
        elif age <= 64: return "Pre-retirement"
        else:           return "Retirement Age"

    df["Age_Range"] = df["Age"].apply(age_group)

    dataset_date = datetime(reference_year, 12, 31)
    df["Customer_For"] = (dataset_date - df["Dt_Customer"]) / np.timedelta64(1, "D")

    df["WebConversionRate"] = np.where(
        df["NumWebVisitsMonth"] != 0,
        df["NumWebPurchases"] / df["NumWebVisitsMonth"],
        0,
    )

    df["Total_Spent"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    df["Total_Purchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )

    df["Adults"] = df["Marital_Status"].replace(
        {"Married": 2, "Together": 2, "Single": 1, "Divorced": 1,
         "Widow": 1, "Alone": 1, "Absurd": 1, "YOLO": 1}
    )
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]
    df["Household_size"] = df["Adults"] + df["Dependents"]
    df["Is_Parent"] = np.where(df["Dependents"] > 0, 1, 0)

    df["Education"] = df["Education"].replace(
        {"Graduation": "Postgraduate", "Master": "Postgraduate"}
    )

    df["Offers_Accepted"] = (
        df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"]
        + df["AcceptedCmp4"] + df["AcceptedCmp5"] + df["Response"]
    )

    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are redundant after feature engineering."""
    cols_to_drop = [
        "Marital_Status", "Dt_Customer", "Year_Birth",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5", "Response", "ID",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


# ── Preprocessing for Clustering ─────────────────────────────────────────────

def preprocess_for_clustering(df: pd.DataFrame) -> np.ndarray:
    """Encode categorical columns and scale all features."""
    df_model = df.copy()
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = LabelEncoder().fit_transform(df_model[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)
    return X_scaled, df_model.columns.tolist()


# ── Optimal Cluster Selection ─────────────────────────────────────────────────

def plot_elbow(X: np.ndarray, max_k: int = 10):
    """Plot the within-cluster sum of squares (WCSS) to aid in choosing k."""
    wcss = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=150)
    plt.show()
    print("Saved: elbow_plot.png")


# ── Clustering and Profiling ──────────────────────────────────────────────────

def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Fit KMeans and return cluster labels."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    print(f"KMeans fitted with k={n_clusters}. Inertia: {km.inertia_:.2f}")
    return labels


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, numeric_cols: list) -> pd.DataFrame:
    """Compute mean values per cluster for selected numeric columns."""
    df_profile = df.copy()
    df_profile["Cluster"] = labels
    profile = df_profile.groupby("Cluster")[numeric_cols].mean().round(2)
    print("\nCluster Profiles:")
    print(profile.to_string())
    return profile


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray):
    """Reduce to 2D via PCA and plot cluster assignments."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=20)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Clusters (PCA 2D Projection)")
    plt.tight_layout()
    plt.savefig("cluster_plot.png", dpi=150)
    plt.show()
    print("Saved: cluster_plot.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "Campaign_data.csv"
    N_CLUSTERS = 4

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df = drop_redundant_columns(df)

    X_scaled, feature_names = preprocess_for_clustering(df)

    plot_elbow(X_scaled)

    labels = fit_kmeans(X_scaled, n_clusters=N_CLUSTERS)

    profile_cols = ["Age", "Income", "Total_Spent", "Total_Purchases",
                    "Customer_For", "Household_size", "Offers_Accepted"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profile_clusters(df, labels, profile_cols)

    plot_clusters_2d(X_scaled, labels)
