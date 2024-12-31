import pandas as pd
import altair as alt
from datetime import datetime
import os

def prepare_time_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare episode data for time-based visualization.
    
    Args:
        df: DataFrame with episode data including 'date' column
    Returns:
        pd.DataFrame: Processed data frame with time information
    """
    # Convert date string to datetime if it's not already
    if df['date'].dtype == 'object':
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract hour
    df['hour'] = df['date'].dt.hour
    
    # Create time string for display
    df['release_time'] = df['date'].dt.strftime('%I:%M %p')
    
    # Clean any problematic Unicode in title
    df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')
    
    # Count episodes per hour
    hour_counts = df.groupby('hour').size().reset_index(name='count')
    
    return hour_counts

def create_release_time_viz(df: pd.DataFrame, podcast_name: str = "Dunc'd On") -> alt.Chart:
    """
    Create an interactive visualization of episode release times by hour with human-readable time labels.
    
    Args:
        df: DataFrame with hour counts
        podcast_name: Name of the podcast for the title
    Returns:
        alt.Chart: Altair chart object
    """
    # Map hour to human-readable time labels
    time_labels = {
        0: "12:00 am", 1: "1:00 am", 2: "2:00 am", 3: "3:00 am", 4: "4:00 am", 5: "5:00 am",
        6: "6:00 am", 7: "7:00 am", 8: "8:00 am", 9: "9:00 am", 10: "10:00 am", 11: "11:00 am",
        12: "12:00 pm", 13: "1:00 pm", 14: "2:00 pm", 15: "3:00 pm", 16: "4:00 pm", 17: "5:00 pm",
        18: "6:00 pm", 19: "7:00 pm", 20: "8:00 pm", 21: "9:00 pm", 22: "10:00 pm", 23: "11:00 pm"
    }
    
    df['time_label'] = df['hour'].map(time_labels)
    
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
            
        # Prepare data
        hour_counts = prepare_time_data(df)
        
        # Create visualization
        chart = create_release_time_viz(hour_counts)
        
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
    csv_file = "data/podcast_episodes.csv"
    output_file = "docs/episode_release_times.html"
    
    # Generate visualization
    generate_html_visualization(csv_file, output_file)

if __name__ == "__main__":
    main()