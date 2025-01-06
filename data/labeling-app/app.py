from pathlib import Path
import pandas as pd
from shiny import reactive
from shiny.express import input, ui, render
# Set up application directory and load data
app_dir = Path(__file__).parent
episodes_csv = app_dir / "podcast_episodes.csv"
df = pd.read_csv(episodes_csv)
print(df)
episode_titles = df["title"].to_list()

# Configure page options and CSS
ui.page_opts(title="Dunc'd On Episode Type")
ui.include_css(app_dir / "styles.css")

with ui.card():
    ui.card_header("Episode Info")
    ui.input_select(
        "episode",
        "Episode",
        choices=episode_titles
    )
        
    @render.data_frame
    def episodes_df():
        filtered_df = df.loc[df["title"] == input.episode(),["title", "description"]]
        return render.DataGrid(filtered_df)

# Episode Type Card
with ui.card():
    ui.card_header("Episode Type")
    ui.input_radio_buttons(
        "episode_type",
        "What episode type is this episode?",
        choices=["daily_duncs", "hollinger_duncan", "gamer", "big_picture"],
        selected=[],
        inline=True,
    )
    
    ui.div(
        ui.input_action_button("submit", "Submit", class_="btn btn-primary"),
        class_="d-flex justify-content-end",
    )

# Save to CSV effect
@reactive.effect
@reactive.event(input.submit)
def save_to_csv():
    fields = {"episode": input.episode(),
            "episode_type": input.episode_type()
            }
    df = pd.DataFrame([fields])
    responses = app_dir / "episode_types.csv"
    
    if not responses.exists():
        df.to_csv(responses, mode="a", header=True)
    else:
        df.to_csv(responses, mode="a", header=False)
    
    ui.modal_show(ui.modal("Form submitted, thank you!"))
