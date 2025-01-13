import pandas as pd
import altair as alt
from datetime import datetime
import os

def create_type_banger_viz(df: pd.DataFrame, podcast_name: str = "Dunc'd On") -> alt.Chart:
    """
    Create an interactive visualization of whether the episode is a banger or not 
    by episode type with human-readable time labels.
    
    Args:
        df: DataFrame with episode types and banger status
        podcast_name: Name of the podcast for the title
    Returns:
        alt.Chart: Altair chart object
    """
    
    # Create bar chart of release times by hour
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('hour:O', 
                axis=alt.Axis(title='Hour of Day',
                              format='d',
                              labelExpr=f"datum.value + ':00'"),
                scale=alt.Scale(domain=list(range(24)))),
        y=alt.Y('count:Q',
                axis=alt.Axis(title='Number of Episodes')),
        tooltip=[
            alt.Tooltip('time_label:N', title='Time'),  # Updated tooltip
            alt.Tooltip('count:Q', title='Episodes')
        ]
    ).properties(
        width=800,
        height=400,
        title=f"{podcast_name} Episode Release Times by Hour of the Day"
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
        df = pd.read_csv(csv_file, encoding='utf-8', encoding_errors='ignore')
        
        if 'date' not in df.columns:
            print("Error: CSV file must contain a 'date' column")
            return
        
        # Create visualization
        chart = create_type_banger_viz(hour_counts)
        
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
    csv_file = "../data/labeling-app/podcast_episodes.csv"
    output_file = "../docs/episode_release_times.html"
    
    # Generate visualization
    generate_html_visualization(csv_file, output_file)

if __name__ == "__main__":
    main()