import pandas as pd
import altair as alt
from datetime import datetime
import os

def create_type_banger_viz(df: pd.DataFrame, podcast_name: str = "Dunc'd On") -> alt.Chart:
    """
    Create an interactive visualization of whether the episode is a banger or not 
    by episode type with human-readable time labels, with flipped axes.
    
    Args:
        df: DataFrame with columns 'episode_type' and 'banger'
        podcast_name: Name of the podcast for the title
    
    Returns:
        alt.Chart: Altair chart object
    """
    # Ensure 'count' is aggregated if not precomputed
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y(
            'episode_type:O',
            axis=alt.Axis(title='Type of Episode'),
        ),
        x=alt.X(
            'count()',
            axis=alt.Axis(title='Number of Episodes')
        ),
        color=alt.Color(
            'banger:N',
            legend=alt.Legend(title='Is Banger?')
        ),
        tooltip=[
            alt.Tooltip('episode_type:N', title='Type of Episode'),
            alt.Tooltip('count()', title='Episodes'),
            alt.Tooltip('banger:N', title="Is Banger?")
        ]
    ).properties(
        width=800,
        height=400,
        title=f"{podcast_name} Episode Types by Banger Status"
    ).configure_axis(
        grid=True
    ).configure_view(
        stroke=None
    ).interactive()
    
    return chart



def generate_html_visualization(csv_file: str, output_file: str = "podcast_viz.html"):
    """
    Generate an interactive HTML visualization from a podcast episodes CSV file.
    
    Args:
        csv_file: Path to the CSV file containing episode data
        output_file: Name of the output HTML file
    """
    try:
        # Read CSV file with explicit UTF-8 encoding and error handling
        df_init = pd.read_csv(csv_file, encoding='utf-8', encoding_errors='ignore')
        
        df = (
            df_init
            .loc[~(df_init["episode_type"] == "daily_duncs")]
            .assign(episode_type = lambda x: x['episode_type'].str.replace('_', ' ').str.title(),
                    banger = lambda x: x['banger'].str.replace('_', ' ').str.title())
            )
        
        if 'episode_type' not in df.columns:
            print("Error: CSV file must contain a 'episode_type' column")
            return
        
        # Create visualization
        chart = create_type_banger_viz(df)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Save as HTML file
        chart.save(output_file)
        print(f"Visualization saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file: {csv_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Does file exist? {os.path.exists(csv_file)}")
    except Exception as e:
        print(f"Error generating visualization: {e}")

def main():
    # Use the correct path to the CSV file
    csv_file = "data/labeling-app/episode_types.csv"
    output_file = "docs/episode_type_by_banger_status.html"
    
    # Generate visualization
    generate_html_visualization(csv_file, output_file)

if __name__ == "__main__":
    main()