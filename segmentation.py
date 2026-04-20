import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openTSNE import TSNE

import os, sys
os.makedirs("images", exist_ok=True)
os.makedirs("outs", exist_ok=True)
np.random.seed(42)

_log = open("outs/segmentation.log", "w", buffering=1)
sys.stdout = _log
sys.stderr = _log

# ── 1. Load ────────────────────────────────────────────────────────────────────
with open("census-bureau.columns", "r", encoding="utf-8") as f:
    cols = [line.strip() for line in f if line.strip()]

df = pd.read_csv("census-bureau.data", header=None, names=cols, sep=",", skipinitialspace=True)
print(f"Loaded: {df.shape}")

# ── 2. Feature selection ───────────────────────────────────────────────────────
numerical_cols = ["age", "capital gains", "capital losses", "dividends from stocks", "weeks worked in year", "wage per hour"]

categorical_cols = [
    "major occupation code", "major industry code", "education",
    "detailed household summary in household", "detailed household and family stat",
    "tax filer stat", "full or part time employment stat", "race", "marital stat",
    "veterans benefits", "country of birth self", "citizenship",
]

df["veterans benefits"] = df["veterans benefits"].astype(str)

# ── 3. Encode ──────────────────────────────────────────────────────────────────
df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
df_final = pd.concat([df[numerical_cols], df_encoded], axis=1)
print(f"Encoded shape: {df_final.shape}")

# ── 4. Scale numerical features ────────────────────────────────────────────────
scaler = StandardScaler()
df_final[numerical_cols] = scaler.fit_transform(df_final[numerical_cols])

# ── 5. PCA – explained variance to choose n_components ────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(df_final.values.astype(np.float32))
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar)
plt.axhline(0.80, color="r", linestyle="--", label="80%")
plt.axhline(0.90, color="g", linestyle="--", label="90%")
plt.axhline(0.95, color="b", linestyle="--", label="95%")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("images/pca_explained_variance.png", dpi=150)

for threshold in [0.80, 0.90, 0.95]:
    n = np.argmax(cumvar >= threshold) + 1
    print(f"{threshold*100:.0f}% variance: {n} components")

# ── 6. PCA 2D – visualization only ────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=42)
pca_result_2d = pca_2d.fit_transform(df_final.values.astype(np.float32))
df_pca_2d = pd.DataFrame(pca_result_2d, columns=["PCA1", "PCA2"])

plt.figure(figsize=(10, 6))
sns.scatterplot(x="PCA1", y="PCA2", data=df_pca_2d, alpha=0.3, s=5)
plt.title("PCA of Census Data (2D)")
plt.tight_layout()
plt.savefig("images/pca_2d.png", dpi=150)

# ── 7. t-SNE – visualization only ─────────────────────────────────────────────
print("Running t-SNE…")
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, negative_gradient_method="fft", verbose=True)
tsne_result = tsne.fit(df_final.values.astype(np.float32))
df_tsne = pd.DataFrame(tsne_result, columns=["tSNE1", "tSNE2"])

plt.figure(figsize=(10, 6))
sns.scatterplot(x="tSNE1", y="tSNE2", data=df_tsne, alpha=0.3, s=5)
plt.title("t-SNE of Census Data")
plt.tight_layout()
plt.savefig("images/tsne_raw.png", dpi=150)

# ── 8. PCA to 28 components (90% variance) for clustering ─────────────────────
N_COMPONENTS = 28  # chosen from explained variance plot
pca_28 = PCA(n_components=N_COMPONENTS, random_state=42)
df_pca_28 = pca_28.fit_transform(df_final.values.astype(np.float32))

# ── 9. Elbow + Silhouette to choose K ─────────────────────────────────────────
print("Running elbow/silhouette search…")
inertias, silhouettes = [], []
K_range = range(2, 12)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(df_pca_28)
    inertias.append(km.inertia_)
    sample_idx = np.random.default_rng(42).choice(len(df_pca_28), size=10000, replace=False)
    silhouettes.append(silhouette_score(df_pca_28[sample_idx], labels[sample_idx]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-")
ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Plot")
ax2.plot(K_range, silhouettes, "ro-")
ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Score")
plt.tight_layout()
plt.savefig("images/kmeans_elbow_silhouette.png", dpi=150)

print("\nSilhouette scores:")
for k, s in zip(K_range, silhouettes):
    print(f"  K={k}: {s:.4f}")

# ── 10. Final KMeans clustering ────────────────────────────────────────────────
K = 6  # update based on elbow/silhouette plots
print(f"\nFitting KMeans with K={K}…")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_pca_28)

plt.figure(figsize=(10, 6))
sns.scatterplot(x="tSNE1", y="tSNE2", data=df_tsne, hue=clusters, palette="Set2", legend="full", alpha=0.4, s=5)
plt.title(f"t-SNE with KMeans Clusters (K={K})")
plt.tight_layout()
plt.savefig("images/tsne_clusters.png", dpi=150)

# ── 11. Cluster profiling ──────────────────────────────────────────────────────
df_profile = df[numerical_cols + categorical_cols].copy()
df_profile["cluster"] = clusters
df_profile["high_income"] = (df["label"] == "50000+.").astype(int)

print("\n── Cluster sizes ──")
print(df_profile["cluster"].value_counts().sort_index())

print("\n── Numerical means per cluster ──")
print(df_profile.groupby("cluster")[numerical_cols].mean().round(1).to_string())

print("\n── Categorical mode per cluster ──")
cat_mode = df_profile.groupby("cluster")[categorical_cols].agg(lambda x: x.value_counts().index[0])
print(cat_mode.to_string())

print("\n── % earning >50K per cluster ──")
print(df_profile.groupby("cluster")["high_income"].mean().mul(100).round(1))

# ── 12. Cluster heatmaps ───────────────────────────────────────────────────────
cluster_num = df_profile.groupby("cluster")[numerical_cols].mean()
cluster_num_norm = (cluster_num - cluster_num.mean()) / cluster_num.std()

plt.figure(figsize=(8, 4))
sns.heatmap(cluster_num_norm.T, annot=cluster_num.T.round(1), fmt="g",
            cmap="RdBu_r", center=0, linewidths=0.5)
plt.title("Cluster Profiles — Numerical Features (color=normalized, label=raw mean)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("images/cluster_profile_numerical.png", dpi=150)

cat_pct = {}
for col in categorical_cols:
    mode_val = df_profile[col].value_counts().index[0]
    cat_pct[f"{col}\n[{mode_val}]"] = (
        df_profile.groupby("cluster")[col]
        .apply(lambda x: (x == mode_val).mean() * 100)
    )

cat_pct_df = pd.DataFrame(cat_pct).T

plt.figure(figsize=(8, 8))
sns.heatmap(cat_pct_df, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, cbar_kws={"label": "% with modal value"})
plt.title("Cluster Profiles — Categorical Features\n(% of cluster with most common overall value)")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("images/cluster_profile_categorical.png", dpi=150)
