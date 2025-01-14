import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import altair as alt
from scipy.spatial.distance import pdist, squareform

def main():
    df = read_filter_embeddings()
    df_pca, min_dist_pair, max_dist_pair = create_pca_df_results(df)
    print_results(min_dist_pair, max_dist_pair)
    create_interactive_chart(df_pca)

def read_filter_embeddings():

    # Read the parquet file
    df = pl.read_parquet('data/labeling-app/description_embeddings.parquet')
    
    print(df.glimpse())
    
    df_filtered = df.filter(pl.col("embedding_dimensions") > 0)
    
    return df_filtered

def create_pca_df_results(df):

    # Assuming 'df' is your Polars DataFrame with 'embeddings' as a column
    embeddings = df['embeddings'].to_list()  # Extract the embeddings as a list

    # Convert to numpy array for PCA
    embeddings_array = np.array(embeddings)

    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings_array)

    # Create a new Polars DataFrame with PCA results
    pca_df = pl.DataFrame({
        'title': df['title'],
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1]
    })

    # Calculate pairwise distances
    distances = pdist(pca_result)
    distance_matrix = squareform(distances)

    # Find min and max distances
    min_dist_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(distance_matrix)) * np.inf), distance_matrix.shape)
    max_dist_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

    # Get episode pairs
    min_dist_pair = {
        'distance': distance_matrix[min_dist_idx],
        'episode1': df['title'].to_list()[min_dist_idx[0]],  # Using .to_list() for access
        'episode2': df['title'].to_list()[min_dist_idx[1]]   # Using .to_list() for access
    }

    max_dist_pair = {
        'distance': distance_matrix[max_dist_idx],
        'episode1': df['title'].to_list()[max_dist_idx[0]],  # Using .to_list() for access
        'episode2': df['title'].to_list()[max_dist_idx[1]]   # Using .to_list() for access
    }
    
    return pca_df, min_dist_pair, max_dist_pair

def print_results(min_dist_pair, max_dist_pair):

    # Print results
    print("\nMost Similar Episodes:")
    print(f"Distance: {min_dist_pair['distance']:.4f}")
    print(f"Episode 1: {min_dist_pair['episode1']}")
    print(f"Episode 2: {min_dist_pair['episode2']}")

    print("\nMost Different Episodes:")
    print(f"Distance: {max_dist_pair['distance']:.4f}")
    print(f"Episode 1: {max_dist_pair['episode1']}")
    print(f"Episode 2: {max_dist_pair['episode2']}")

def create_interactive_chart(pca_df):

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

if __name__ == "__main__":
    main()