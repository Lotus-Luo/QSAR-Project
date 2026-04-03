# %%
# 1. Import Libraries and Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_theme(style="whitegrid", font='Times New Roman')

print("Libraries loaded. Matplotlib configured to Times New Roman.")
print(f"Current working directory: {os.getcwd()}")

# %% 
# 2. Load Data and Process Fingerprints
file_path = '../Data/JAK3/JAK3_final_dataset.csv' ##### Adjust path if necessary
df = pd.read_csv(file_path)

# Identify fingerprint columns
fp_cols = [c for c in df.columns if c.startswith('morgan_')]

if len(fp_cols) >= 2048:
    print(f"Detected {len(fp_cols)} existing fingerprint columns.")
    X = df[fp_cols].values
else:
    print("Incomplete fingerprints detected. Generating 2048-bit Morgan FPs...")
    def get_fp(smile):
        mol = Chem.MolFromSmiles(smile)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    X = np.array([get_fp(s) for s in df['smiles']])

X = X.astype(int)
print(f"Data matrix shape: {X.shape}")
# Display first 5 rows of the dataframe to check data integrity
df.head()
# %%
df.to_csv("../Data/JAK3/JAK3_final_dataset_with_fps.csv", index=False)
print(f"Data with fingerprints saved to ../Data/JAK3/JAK3_final_dataset_with_fps.csv")

# %% 
# 3. Dimension Reduction via t-SNE
print("Running t-SNE reduction (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X)

# Store results
df['tsne_1'] = X_tsne[:, 0]
df['tsne_2'] = X_tsne[:, 1]

print("t-SNE completed.")
df[['smiles', 'tsne_1', 'tsne_2']].head()

# %% 
# 4. Determine Optimal Clusters (Elbow Method & Silhouette)
inertia = []
silhouette_avg = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_tsne)
    inertia.append(kmeans.inertia_)
    silhouette_avg.append(silhouette_score(X_tsne, labels))

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)

# Configure AX1 (Inertia - Elbow)
color_elbow = '#1f77b4' # Academic Blue
ax1.set_xlabel('Number of Clusters ($k$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Inertia (In-cluster SSE)', color=color_elbow, fontsize=14, fontweight='bold')
line1 = ax1.plot(k_range, inertia, marker='o', markersize=8, color=color_elbow, 
                 linewidth=2.5, label='Inertia', linestyle='-', alpha=0.9)
ax1.tick_params(axis='y', labelcolor=color_elbow, labelsize=12, direction='in', width=1.5)
ax1.tick_params(axis='x', labelsize=12, direction='in', width=1.5)

# Configure AX2 (Silhouette Score)
ax2 = ax1.twinx()
color_sil = '#d62728' # Academic Red
ax2.set_ylabel('Silhouette Score', color=color_sil, fontsize=14, fontweight='bold')
line2 = ax2.plot(k_range, silhouette_avg, marker='s', markersize=8, color=color_sil, 
                 linewidth=2.5, label='Silhouette Score', linestyle='--', alpha=0.9)
ax2.tick_params(axis='y', labelcolor=color_sil, labelsize=12, direction='in', width=1.5)

# High-end Aesthetic Adjustments
# 1. Force the frame to be visible and thick
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_visible(True)

# 2. Refined Grid
ax1.grid(True, linestyle='--', alpha=0.3, which='major')

# 3. Combined Legend (Optional, but clean)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', frameon=True, fontsize=10, edgecolor='black')

plt.title('Clustering Optimization: Elbow & Silhouette Analysis', fontsize=16, pad=20, fontweight='bold')
plt.tight_layout()

# Save as SVG for infinite scalability in papers
plt.savefig('clustering_optimization_journal_style.svg', format='svg', bbox_inches='tight')
plt.show()

# Print metrics table
eval_df = pd.DataFrame({'k': k_range, 'Inertia': [f"{x:.2f}" for x in inertia], 
                        'Silhouette': [f"{x:.4f}" for x in silhouette_avg]})
print("       CLUSTERING METRICS SUMMARY")
print(eval_df.to_string(index=False))


# %%
# 5. Final K-Means Clustering (Manual Adjustment)

# --- USER INPUT REQUIRED ---
# Based on the previous plot (Elbow/Silhouette), you might want to change K value:
manual_k = 6  # <--- Change this value manually after checking the evaluation plot

print(f"Executing K-Means with user-defined k={manual_k}...")

# Execute final clustering
kmeans_final = KMeans(n_clusters=manual_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_tsne)
centroids = kmeans_final.cluster_centers_

# Display distribution summary
cluster_counts = df['cluster'].value_counts().sort_index()
print("\n" + "-"*30)
print(f"{'Cluster':<10} | {'Molecules':<10}")
print("-"*30)
for c, count in cluster_counts.items():
    print(f"{c:<10} | {count:<10}")
print("-"*30)
# %% 
# 6. Visualization and Export 
output_base = f"../Data/JAK3/jak3_cluster_k{manual_k}"
# Set up the figure with high resolution
plt.figure(figsize=(10, 8), dpi=150)
ax = plt.gca()

# 1. Define high-quality color palette and markers
palette = sns.color_palette("husl", manual_k) # Academic-friendly vibrant palette
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '8', 'X']

# 2. Plot each cluster with refined aesthetics
for i, cluster in enumerate(sorted(df['cluster'].unique())):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(
        cluster_data['tsne_1'], 
        cluster_data['tsne_2'], 
        label=f'Cluster {i}',
        color=palette[i],
        alpha=0.65,           # Slight transparency for overlapping points
        edgecolors='white',   # White edge to separate points
        linewidths=0.5,
        marker=markers[i % len(markers)],
        s=70                  # Increased point size
    )

# 3. Plot Centroids - Minimalist yet distinctive
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=180, 
    c='none',             # Hollow center
    edgecolors='black',   # Solid black edge
    marker='o',           # Circle for centroid area
    linewidths=2,
    alpha=0.8,
    zorder=5
)
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    s=100, 
    c='black', 
    marker='x',           # Cross for center point
    linewidths=2,
    label='Centroids',
    zorder=6
)

# 4. Journal Formatting
plt.title(f'Chemical Space Distribution of NSD2 Inhibitors ($k$={manual_k})', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')

# Configure Legend: Multi-column if many clusters
num_cols = 1 if manual_k <= 5 else 2
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
           fontsize=11, title="Clusters", title_fontsize=12, ncol=num_cols)

# Tighten the frame and set tick direction
ax.tick_params(direction='in', length=6, width=1.5, labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.grid(True, linestyle='--', alpha=0.3)

# 5. Export Files

plt.savefig(f"{output_base}.png", dpi=600, bbox_inches='tight')
plt.savefig(f"{output_base}.svg", format='svg', bbox_inches='tight')

plt.show()
print(f"Success! Journal-style plots saved as {output_base}.png and {output_base}.svg")
# %%
