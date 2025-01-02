import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import altair as alt
from scipy.spatial.distance import pdist, squareform

# Read the parquet file
df = pd.read_parquet('output/description_embeddings.parquet')

# Filter to embeddings with more than 0 dimensions
df = df[df['embeddings'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

# Run PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df['embeddings'].tolist())

# Create a new dataframe with PCA results
pca_df = pd.DataFrame(
    data=pca_result,
    columns=['PC1', 'PC2']
)

# Add the title column from original dataframe
pca_df['title'] = df['title']

# Calculate pairwise distances
distances = pdist(pca_result)
distance_matrix = squareform(distances)

# Find min and max distances
min_dist_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(distance_matrix)) * np.inf), distance_matrix.shape)
max_dist_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

# Get episode pairs
min_dist_pair = {
    'distance': distance_matrix[min_dist_idx],
    'episode1': df['title'].iloc[min_dist_idx[0]],
    'episode2': df['title'].iloc[min_dist_idx[1]]
}

max_dist_pair = {
    'distance': distance_matrix[max_dist_idx],
    'episode1': df['title'].iloc[max_dist_idx[0]],
    'episode2': df['title'].iloc[max_dist_idx[1]]
}

# Print results
print("\nMost Similar Episodes:")
print(f"Distance: {min_dist_pair['distance']:.4f}")
print(f"Episode 1: {min_dist_pair['episode1']}")
print(f"Episode 2: {min_dist_pair['episode2']}")

print("\nMost Different Episodes:")
print(f"Distance: {max_dist_pair['distance']:.4f}")
print(f"Episode 1: {max_dist_pair['episode1']}")
print(f"Episode 2: {max_dist_pair['episode2']}")

# Create interactive scatter plot
chart = alt.Chart(pca_df).mark_circle().encode(
    x='PC1:Q',
    y='PC2:Q',
    tooltip=['title:N']
).properties(
    width=600,
    height=400,
    title='PCA Visualization of Embeddings'
).interactive()

# Save the chart
chart.save('docs/description_embeddings_viz.html')