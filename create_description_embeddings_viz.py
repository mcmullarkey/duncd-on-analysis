import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import altair as alt

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