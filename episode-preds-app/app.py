from flask import Flask, render_template, jsonify
import polars as pl
import onnxruntime
import os
from datetime import datetime, timedelta, timezone
import json
import logging
import requests
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the ONNX model
logger.info("Loading ONNX model...")
session = onnxruntime.InferenceSession("episode_banger_model.onnx")
logger.info("ONNX model loaded successfully")

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

def get_recent_episodes():
    rss_url = os.environ.get('RSS_FEED_URL')
    if not rss_url:
        raise ValueError("RSS_FEED_URL environment variable not set")
    
    logger.info(f"Fetching RSS feed from {rss_url}...")
    
    # HTTP headers for RSS feed request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(rss_url, headers=headers)
        response.raise_for_status()
        logger.info("RSS feed fetched successfully")
    except requests.RequestException as e:
        logger.error(f"Error fetching RSS feed: {e}", exc_info=True)
        raise RuntimeError(f"Error fetching RSS feed: {e}")
    
    try:
        root = ET.fromstring(response.content)
        channel = root.find('channel')
        if channel is None:
            raise ValueError("No channel element found in RSS feed")
        
        episodes = []
        # Make cutoff_date timezone-aware (UTC)
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=7))
        
        for item in channel.findall('item'):
            pub_date = item.find('pubDate')
            if pub_date is not None:
                # pub_date_dt is offset-aware
                pub_date_dt = datetime.strptime(pub_date.text, '%a, %d %b %Y %H:%M:%S %z')
                if pub_date_dt > cutoff_date:
                    episodes.append({
                        'episode': item.find('title').text if item.find('title') is not None else 'Unknown Title',
                        'description': item.find('description').text if item.find('description') is not None else '',
                        'date': pub_date_dt.isoformat(),
                        'duration': item.find('.//{http://www.itunes.com/dtds/podcast-1.0.dtd}duration').text
                            if item.find('.//{http://www.itunes.com/dtds/podcast-1.0.dtd}duration') is not None else ''
                    })
        
        logger.info(f"Filtered to {len(episodes)} recent episodes")
        return pl.DataFrame(episodes).lazy().collect()
    
    except ET.ParseError as e:
        logger.error(f"Error parsing RSS feed XML: {e}", exc_info=True)
        raise RuntimeError(f"Error parsing RSS feed XML: {e}")



def engineer_features(df):
    logger.info("Starting feature engineering...")
    df_types = df.select(
        pl.col("episode"),
        pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z"),
        pl.col("description"),
        pl.col("duration").str.strip_chars().cast(pl.Float32).alias("duration_secs")
    )
    
    df_features = df_types.select(
        pl.col("episode").str.contains(r"Game [1-7]").alias("about_playoff_game").cast(pl.Float32),
        pl.col("episode").str.contains(r"H&D").alias("is_hollinger_duncan").cast(pl.Float32),
        pl.col("episode").str.contains(r"Daily Duncs").alias("is_daily_dunc").cast(pl.Float32),
        pl.col("episode").str.contains(r"Mock").alias("is_mock_episode").cast(pl.Float32),
        pl.col("episode").str.contains(r"Awards").alias("is_awards_episode").cast(pl.Float32),
        pl.col("date").dt.year().alias("year").cast(pl.Float32),
        pl.col("date").dt.month().alias("month").cast(pl.Float32),
        pl.col("date").dt.weekday().alias("weekday").cast(pl.Float32),
        pl.col("date").dt.hour().alias("hour").cast(pl.Float32),
        (pl.col("duration_secs") > 1800).alias("longer_thirty_min").cast(pl.Float32),
        pl.col("duration_secs").cast(pl.Float32),
        pl.col("episode").str.contains(r"Celtics").alias("description_contains_celtics").cast(pl.Float32),
    ).drop_nulls()
    
    logger.info("Feature engineering completed")
    return df_features

def predict_bangers(features):
    logger.info("Making predictions...")
    # Prepare features for ONNX model
    input_name = session.get_inputs()[0].name
    pred_onx = session.run(None, {input_name: features})[1]
    logger.info("Predictions completed")
    return pred_onx

@app.route('/')
def home():
    logger.info("Homepage requested")
    return render_template('index.html')

@app.route('/predict')
def predict():
    logger.info("Prediction endpoint called")
    try:
       # Get recent episodes
        episodes_df = get_recent_episodes()
        
        # Retain episode and date for later
        episode_metadata = episodes_df.select(["episode", "date"])
        
        # Engineer features
        features_df = engineer_features(episodes_df)
        
        # Make predictions (exclude non-feature columns like 'episode' and 'date')
        predictions = predict_bangers(features_df.to_numpy())
        
        # Combine results
        results = []
        for i, row in enumerate(features_df.iter_rows(named=True)):
            results.append({
                'episode': episode_metadata[i, "episode"],
                'probability': float(round(predictions[i]["yes"], 2)),
                'date': episode_metadata[i, "date"]
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        logger.info(f"Successfully processed {len(results)} episodes")
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    # Add more explicit startup logging
    print(f"Starting server on port 8080...")
    print(f"Debug mode: {app.debug}")
    print(f"Environment: {os.environ.get('FLASK_ENV')}")
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)