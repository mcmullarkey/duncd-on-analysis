import polars as pl
import altair as alt
from datetime import datetime
import os

def prepare_time_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare episode data for time-based visualization.
    
    Args:
        df: DataFrame with episode data including 'date' column
    
    Returns:
        pl.DataFrame: Processed data frame with time information
    """
    # Check if 'date' column exists and is a string
    if 'date' in df.schema and df.schema['date'] == pl.Utf8:
        try:
            # Adjusted to use `pl.Datetime` and corrected format for datetime
            df = df.with_columns(
                pl.col('date').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False)
            )
        except Exception as e:
            raise ValueError(f"Error converting 'date' column: {e}")

    # Check for successful conversion
    if df.schema.get('date') != pl.Datetime:
        raise ValueError("'date' column is not in a valid datetime format.")

    # Add and group by hour column
    try:
        hour_counts = df.with_columns(
            hour = pl.col("date").dt.hour(),
            release_time = pl.col("date").dt.strftime('%I:%M %p'),
            title = pl.col("title").str.replace_all(r'[^\x00-\x7F]+', '')  # Remove non-ASCII characters
        ).group_by(
            "hour",
            maintain_order=True
        ).agg(
            count = pl.len()
        )
    except Exception as e:
        raise RuntimeError(f"Error processing the DataFrame: {e}")

    return hour_counts


def create_release_time_viz(df: pl.DataFrame, podcast_name: str = "Dunc'd On") -> alt.Chart:
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
    
    df = df.with_columns(
        pl.col("hour").replace_strict(time_labels).alias("time_label")
    )
    
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
        df = pl.read_csv(csv_file, encoding='utf8', ignore_errors=True)
        print("Reading df succeeded!")
        
        if 'date' not in df.columns:
            print("Error: CSV file must contain a 'date' column")
            return
            
        # Prepare data
        print("Preparing time data")
        hour_counts = prepare_time_data(df)
        print("Time data prepared")
        print(hour_counts)
        
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
    csv_file = "data/labeling-app/podcast_episodes.csv"
    output_file = "docs/episode_release_times.html"
    
    # Generate visualization
    generate_html_visualization(csv_file, output_file)

if __name__ == "__main__":
    main()