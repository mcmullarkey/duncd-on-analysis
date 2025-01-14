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
    df_embedding = pl.read_parquet('data/labeling-app/description_embeddings.parquet')
    print(df_embedding)
    
    df_episode_types = pl.read_csv("data/labeling-app/episode_types.csv")
    print(df_episode_types)
    
    df_full = df_episode_types.join(df_embedding, left_on = "episode", right_on = "title", how = "left")
    print("Join succeeds!")
    print(df_full)
    
    df_filtered = df_full.filter(pl.col("embedding_dimensions") > 0)
    
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
        'title': df['episode'],
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'episode_type': df['episode_type'].str.replace("_"," ").str.to_titlecase()
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
        'episode1': df['episode'].to_list()[min_dist_idx[0]],  # Using .to_list() for access
        'episode2': df['episode'].to_list()[min_dist_idx[1]]   # Using .to_list() for access
    }

    max_dist_pair = {
        'distance': distance_matrix[max_dist_idx],
        'episode1': df['episode'].to_list()[max_dist_idx[0]],  # Using .to_list() for access
        'episode2': df['episode'].to_list()[max_dist_idx[1]]   # Using .to_list() for access
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
    
    interval = alt.selection_interval()

    # Create interactive scatter plot
    scatter = alt.Chart(pca_df).mark_circle().encode(
        x= alt.X('PC1:Q', axis = alt.Axis(title = "Approximated Daily Duncs -> Main Episodes Embedding")),
        y= alt.Y('PC2:Q', axis = alt.Axis(title = "Approximated Big Picture -> Gamer Embedding")),
        color = alt.Color("episode_type:N", legend = None),
        tooltip=[alt.Tooltip('title:N', title = "Title"),
                 alt.Tooltip('episode_type:N', title = "Episode Type")]
    ).properties(
        width=800,
        height=600,
        title= alt.TitleParams("Principal Components of Dunc'd On Episode Description Embeddings",
        subtitle= ["The first two PCA components appear to differentiate between episode types",
                   "Drap and drop any section of the chart to see how many of each episode type are in the area"])
    ).add_params(
        interval
    )
    
    hist = alt.Chart(pca_df).mark_bar().encode(
        x= alt.X("count()", axis = alt.Axis(title = "Count of Episode Types in Selected Area")),
        y = alt.Y("episode_type:N", axis = alt.Axis(title = "")),
        color = "episode_type:N"
    ).properties(
        width=800,
        height=80
    ).transform_filter(
        interval
    )
    
    chart = scatter & hist

    # Save the chart
    chart.save('docs/description_embeddings_viz.html')

if __name__ == "__main__":
    main()