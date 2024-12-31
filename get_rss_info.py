import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional
import csv
import os
from urllib.parse import urlparse

class PodcastRSSParser:
    """Parser for extracting episode information from podcast RSS feeds."""
    
    def __init__(self, rss_url: str, headers: Dict = None):
        """
        Initialize the parser with an RSS feed URL and optional headers.
        
        Args:
            rss_url (str): The URL of the podcast RSS feed
            headers (Dict, optional): Additional HTTP headers for authentication
        """
        self.rss_url = rss_url
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def fetch_feed(self) -> Optional[str]:
        """
        Fetch the RSS feed content with authentication support.
        
        Returns:
            str: The XML content of the feed
            None: If there was an error fetching the feed
        """
        try:
            response = requests.get(self.rss_url, headers=self.headers, allow_redirects=True)
            response.raise_for_status()
            
            # Check if we got XML content
            content_type = response.headers.get('content-type', '').lower()
            if 'xml' not in content_type and 'rss' not in content_type:
                print(f"Warning: Response may not be RSS/XML content (got {content_type})")
            
            return response.text
            
        except requests.RequestException as e:
            parsed_url = urlparse(self.rss_url)
            if 'supportingcast.fm' in parsed_url.netloc:
                print("Error: This appears to be a Supporting Cast private feed URL.")
                print("Make sure:")
                print("1. The URL is complete and hasn't expired")
                print("2. You're using a fresh URL from your Supporting Cast account")
                print(f"Technical details: {str(e)}")
            else:
                print(f"Error fetching RSS feed: {e}")
            return None
            
    def parse_episodes(self) -> List[Dict]:
        """
        Parse the RSS feed and extract episode information.
        
        Returns:
            list: List of dictionaries containing episode information
        """
        feed_content = self.fetch_feed()
        if not feed_content:
            return []
            
        episodes = []
        try:
            # Parse XML content
            root = ET.fromstring(feed_content)
            
            # Find all item elements (episodes)
            channel = root.find('channel')
            if channel is None:
                return []
            
            # Get podcast title
            podcast_title = channel.find('title')
            if podcast_title is not None:
                print(f"Podcast: {podcast_title.text}")
                
            for item in channel.findall('item'):
                episode = {
                    'title': item.find('title').text if item.find('title') is not None else 'Unknown Title',
                    'date': self._parse_date(item.find('pubDate').text) if item.find('pubDate') is not None else None,
                    'description': item.find('description').text if item.find('description') is not None else '',
                    'duration': item.find('.//{http://www.itunes.com/dtds/podcast-1.0.dtd}duration').text 
                        if item.find('.//{http://www.itunes.com/dtds/podcast-1.0.dtd}duration') is not None else '',
                    'url': item.find('enclosure').get('url') if item.find('enclosure') is not None else ''
                }
                episodes.append(episode)
                
        except ET.ParseError as e:
            print(f"Error parsing RSS feed XML: {e}")
            return []
            
        return episodes

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse the publication date string into a datetime object.
        
        Args:
            date_str (str): Date string from RSS feed
            
        Returns:
            datetime: Parsed datetime object
            None: If parsing fails
        """
        try:
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
            except ValueError:
                print(f"Could not parse date: {date_str}")
                return None

    def save_to_csv(self, episodes: List[Dict], output_file: str) -> bool:
        """
        Save episode information to a CSV file.
        
        Args:
            episodes (List[Dict]): List of episode dictionaries
            output_file (str): Path to the output CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if not episodes:
                    print("No episodes to write to CSV.")
                    return False
                    
                # Get fields from the first episode
                fieldnames = episodes[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header and rows
                writer.writeheader()
                for episode in episodes:
                    # Convert datetime to string for CSV
                    if episode['date']:
                        episode['date'] = episode['date'].strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow(episode)
                    
            print(f"Successfully saved {len(episodes)} episodes to {output_file}")
            return True
            
        except IOError as e:
            print(f"Error saving to CSV: {e}")
            return False

def main():
    # Supporting Cast RSS feed URL
    rss_url = "https://duncdon.supportingcast.fm/content/eyJ0IjoicCIsImMiOiI1MzUiLCJ1IjoiMTQyODU0IiwiZCI6IjE1OTcyNjE1ODYiLCJrIjoxNDJ9fGRjODdmZWEzMTA0MjcwYWJhZGQxZjQ0N2FlNzMyOWUyNDJhMTVhODYzNTBjZjY2Mzc4ZDlhZDg3YTBhZTU4NTY.rss"
    output_file = "output/podcast_episodes.csv"
    
    # Initialize parser with custom headers
    parser = PodcastRSSParser(rss_url)
    episodes = parser.parse_episodes()
    
    if episodes:
        print(f"Found {len(episodes)} episodes")
        parser.save_to_csv(episodes, output_file)
    else:
        print("No episodes found or error occurred while parsing the feed.")

if __name__ == "__main__":
    main()