import os
import re
import time
import csv
import json
import logging
from datetime import datetime
from urllib.parse import urlparse
from dateutil import parser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing the required packages
try:
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup, Comment
except ImportError:
    logger.error("Missing required packages. Please install them with:")
    logger.error("pip install selenium webdriver-manager beautifulsoup4 pandas python-dateutil")
    exit(1)

class FootballMatchScraper:
    def __init__(self, headless=True, debug=False):
        """Initialize the football match scraper"""
        self.driver = None
        self.headless = headless
        self.debug = debug
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
    def start_driver(self):
        """Start the Selenium WebDriver"""
        options = Options()
        if self.headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # Add user agent to avoid detection
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)  # 30 second timeout
        
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
    
    def get_match_links(self, team_url, num_matches=7):
        """Get the most recent match links for a team"""
        matches_url = f"{team_url}/matchlogs/all_comps/schedule/"
        logger.info(f"Loading matches page: {matches_url}")
        self.driver.get(matches_url)
        time.sleep(3)  # Wait for page to load
        
        # Find all match links in the table
        match_links = []
        try:
            match_rows = self.driver.find_elements(By.XPATH, "//table[@id='matchlogs_for']/tbody/tr")
            
            for row in match_rows:
                # Skip rows without links or with 'Preview' instead of 'Match Report'
                try:
                    link_elem = row.find_element(By.XPATH, ".//td[@data-stat='match_report']/a")
                    if link_elem.text == "Match Report":
                        match_links.append(link_elem.get_attribute('href'))
                        if len(match_links) >= num_matches:
                            break
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting match links: {e}")
            
        return match_links
    
    def extract_team_name(self, url):
        """Extract team name from URL"""
        match = re.search(r'/([^/]+)-Stats$', url)
        if match:
            return match.group(1).replace('-', ' ')
        return "Unknown Team"
    
    def uncomment_html(self, html_content):
        """Extract contents of HTML comments that contain tables"""
        # Pattern to find commented-out tables or divs
        pattern = re.compile(r'<!--\s*(<(?:table|div)[^>]*>.*?</(?:table|div)>)\s*-->', re.DOTALL)
        
        # Replace comments with their content
        return pattern.sub(r'\1', html_content)
    
    def extract_data_from_comment(self, soup, element_id):
        """Extract data from a commented section with the given ID"""
        element = soup.find(id=element_id)
        if not element:
            return None
            
        # Look for comments within this element
        comments = element.find_all(string=lambda text: isinstance(text, Comment))
        
        for comment in comments:
            # Parse the comment as HTML
            comment_soup = BeautifulSoup(comment, 'html.parser')
            
            # Look for tables or necessary elements
            table = comment_soup.find('table')
            if table:
                return table
                
        return None
    
    def extract_match_stats_table(self, soup):
        """Extract the match stats table using multiple methods"""
        # Method 1: Direct table access
        stats_table = soup.find('table', {'id': lambda x: x and 'stats_' in x and 'all' in x})
        
        # Method 2: Look in comments for tables 
        if not stats_table:
            # Find all divs that might contain commented stats tables
            divs = soup.find_all('div', {'id': lambda x: x and 'all_stats' in x})
            for div in divs:
                comments = div.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    # Parse the comment content as HTML
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    # Look for tables
                    table = comment_soup.find('table')
                    if table:
                        stats_table = table
                        break
                if stats_table:
                    break
                    
        # Method 3: Use pandas to read all tables and find the stats one
        if not stats_table:
            try:
                # Get the page HTML after JavaScript execution
                html_content = self.driver.page_source
                
                # Uncomment any HTML comments that might contain tables
                clean_html = self.uncomment_html(html_content)
                
                # Try to read all tables
                tables = pd.read_html(clean_html)
                
                # Look for the stats table based on column names
                for i, table in enumerate(tables):
                    # Check if this looks like a stats table (has possession, shots, etc.)
                    if any(col.lower() in str(table.columns).lower() for col in ['possession', 'shots', 'passes', 'fouls']):
                        logger.debug(f"Found stats table at index {i}")
                        return table
            except Exception as e:
                logger.debug(f"Error using pandas to find tables: {e}")
        
        return stats_table
    
    def normalize_date(self, date_str):
        """Convert date string to ISO format"""
        if not date_str:
            return None
        
        try:
            # Remove venue time and local time phrases
            date_str = re.sub(r'\(venue time\)|\(local time\)', '', date_str)
            # Parse and format as ISO
            dt_obj = parser.parse(date_str)
            return dt_obj.strftime('%Y-%m-%d')
        except Exception as e:
            logger.debug(f"Error parsing date '{date_str}': {e}")
            return date_str
    
    def get_int_value(self, text):
        """Convert text to integer, handling commas and other non-numeric chars"""
        if text is None:
            return None
            
        # Convert to string if not already
        text = str(text)
        
        # Remove commas and other non-numeric chars except decimal point
        text = re.sub(r'[^\d.-]', '', text)
        
        if text.strip() == '':
            return None
            
        try:
            return int(float(text))
        except:
            return None
    
    def get_float_value(self, text):
        """Convert text to float, handling percentage signs and commas"""
        if text is None:
            return None
            
        # Convert to string if not already
        text = str(text)
        
        # Remove % and other non-numeric chars except decimal point
        text = re.sub(r'[^\d.-]', '', text)
        
        if text.strip() == '':
            return None
            
        try:
            return float(text)
        except:
            return None
    
    def scrape_match_data(self, match_url, team_name):
        """Scrape data for a specific match"""
        logger.info(f"Scraping match: {match_url}")
        self.driver.get(match_url)
        time.sleep(3)  # Wait for page to load
        
        # Extract match ID from URL for a unique identifier
        url_parts = match_url.split('/')
        match_id = url_parts[5] if len(url_parts) >= 6 else "unknown"
        
        logger.debug(f"Match ID: {match_id}")
        
        # Get page source and create soup
        html_content = self.driver.page_source
        
        # Uncomment any HTML comments that might contain tables
        clean_html = self.uncomment_html(html_content)
        
        # Parse the HTML
        soup = BeautifulSoup(clean_html, 'html.parser')
        
        # Initialize match data dictionary
        match_data = {
            'match_id': match_id,
            'team': team_name,
            'venue': None,
            'date': None,
            'competition': None,
            'result': None,
            'home_team': None,
            'away_team': None,
            'home_goals': None,
            'away_goals': None,
            'home_xg': None,
            'away_xg': None,
            'home_possession': None,
            'away_possession': None,
            'home_shots': None,
            'away_shots': None,
            'home_shots_on_target': None,
            'away_shots_on_target': None,
            'home_big_chances': None,
            'away_big_chances': None,
            'home_passes': None,
            'away_passes': None,
            'home_pass_pct': None,
            'away_pass_pct': None,
            'home_corners': None,
            'away_corners': None,
            'home_fouls': None,
            'away_fouls': None,
            'home_yellow_cards': None,
            'away_yellow_cards': None,
            'home_red_cards': None,
            'away_red_cards': None,
            'match_url': match_url
        }
        
        # Extract basic match information
        try:
            # Get date
            try:
                date_div = soup.select_one('.venuetime')
                if date_div:
                    date_str = date_div.text.strip()
                    match_data['date'] = self.normalize_date(date_str)
                    logger.debug(f"Date: {match_data['date']}")
            except Exception as e:
                logger.debug(f"Error extracting date: {e}")
            
            # Get competition
            try:
                comp_div = soup.select_one('.scorebox_meta div')
                if comp_div:
                    match_data['competition'] = comp_div.text.strip()
                    logger.debug(f"Competition: {match_data['competition']}")
            except Exception as e:
                logger.debug(f"Error extracting competition: {e}")
            
            # Get teams and score
            try:
                # Method 1: Using score boxes
                scorebox = soup.select_one('.scorebox')
                if scorebox:
                    teams = scorebox.select('a[itemprop="name"]')
                    scores = scorebox.select('.score')
                    
                    if len(teams) >= 2 and len(scores) >= 2:
                        match_data['home_team'] = teams[0].text.strip()
                        match_data['away_team'] = teams[1].text.strip()
                        match_data['home_goals'] = self.get_int_value(scores[0].text)
                        match_data['away_goals'] = self.get_int_value(scores[1].text)
                        
                        # Determine venue and result
                        if team_name.lower() in match_data['home_team'].lower():
                            match_data['venue'] = 'Home'
                            if match_data['home_goals'] > match_data['away_goals']:
                                match_data['result'] = 'W'
                            elif match_data['home_goals'] < match_data['away_goals']:
                                match_data['result'] = 'L'
                            else:
                                match_data['result'] = 'D'
                        else:
                            match_data['venue'] = 'Away'
                            if match_data['away_goals'] > match_data['home_goals']:
                                match_data['result'] = 'W'
                            elif match_data['away_goals'] < match_data['home_goals']:
                                match_data['result'] = 'L'
                            else:
                                match_data['result'] = 'D'
                        
                        logger.debug(f"Teams: {match_data['home_team']} vs {match_data['away_team']}")
                        logger.debug(f"Score: {match_data['home_goals']} - {match_data['away_goals']}")
                        logger.debug(f"Venue: {match_data['venue']}, Result: {match_data['result']}")
                
                # Method 2: If Method 1 fails, try alternative selectors
                if not match_data['home_team'] or not match_data['away_team']:
                    logger.debug("Using alternative method to get teams and scores")
                    # Alternative approach - check the content to see what's available
                    
                    # Try getting from the title
                    title = soup.find('title')
                    if title:
                        title_text = title.text
                        match = re.search(r'(.+?)\s+vs\.\s+(.+?)\s+-', title_text)
                        if match:
                            match_data['home_team'] = match.group(1).strip()
                            match_data['away_team'] = match.group(2).strip()
            
            except Exception as e:
                logger.debug(f"Error extracting teams and score: {e}")
            
            # Get xG values
            try:
                for div in soup.select('.scorebox_meta div'):
                    if 'xG' in div.text:
                        xg_values = re.findall(r'(\d+\.\d+)', div.text)
                        if len(xg_values) >= 2:
                            match_data['home_xg'] = float(xg_values[0])
                            match_data['away_xg'] = float(xg_values[1])
                            logger.debug(f"xG: {match_data['home_xg']} - {match_data['away_xg']}")
                            break
            except Exception as e:
                logger.debug(f"Error extracting xG: {e}")
            
            # Extract match statistics
            stats_table = self.extract_match_stats_table(soup)
            
            if stats_table is not None:
                logger.debug("Found stats table")
                
                # If it's a pandas DataFrame, process it directly
                if isinstance(stats_table, pd.DataFrame):
                    try:
                        # Extract stats based on the DataFrame structure
                        for index, row in stats_table.iterrows():
                            stat_name = row.iloc[0].lower() if len(row) > 0 else ''
                            home_val = row.iloc[1] if len(row) > 1 else None
                            away_val = row.iloc[2] if len(row) > 2 else None
                            
                            self.process_stat(match_data, stat_name, home_val, away_val)
                    except Exception as e:
                        logger.debug(f"Error processing DataFrame stats: {e}")
                
                # If it's a BeautifulSoup element, process it as HTML
                else:
                    try:
                        # Process HTML stats table
                        rows = stats_table.select('tbody > tr')
                        
                        for row in rows:
                            try:
                                stat_header = row.select_one('th')
                                if not stat_header:
                                    continue
                                
                                stat_name = stat_header.text.strip().lower()
                                values = row.select('td')
                                
                                if len(values) < 2:
                                    continue
                                
                                home_val = values[0].text.strip()
                                away_val = values[1].text.strip()
                                
                                self.process_stat(match_data, stat_name, home_val, away_val)
                            except Exception as e:
                                logger.debug(f"Error processing row in stats table: {e}")
                    except Exception as e:
                        logger.debug(f"Error processing HTML stats table: {e}")
            else:
                logger.warning("No stats table found")
            
            # Try an alternative approach if we're still missing data
            if (match_data['home_possession'] is None or match_data['away_possession'] is None or
                    match_data['home_shots'] is None or match_data['away_shots'] is None):
                logger.debug("Trying alternative approach to extract stats")
                
                # Try to find individual stats sections
                for stat_name, stat_id in [
                    ('possession', 'all_possession'),
                    ('shots', 'all_shots'),
                    ('passing', 'all_passing'),
                    ('misc', 'all_misc')
                ]:
                    try:
                        section = soup.find('div', {'id': stat_id})
                        if section:
                            # Try to extract data from the section
                            comments = section.find_all(string=lambda text: isinstance(text, Comment))
                            for comment in comments:
                                comment_soup = BeautifulSoup(comment, 'html.parser')
                                # Look for tables in the comment
                                tables = comment_soup.find_all('table')
                                for table in tables:
                                    rows = table.select('tbody > tr')
                                    for row in rows:
                                        # Extract and process the stats
                                        try:
                                            stat_header = row.select_one('th')
                                            if not stat_header:
                                                continue
                                            
                                            stat_name = stat_header.text.strip().lower()
                                            values = row.select('td')
                                            
                                            if len(values) < 2:
                                                continue
                                            
                                            home_val = values[0].text.strip()
                                            away_val = values[1].text.strip()
                                            
                                            self.process_stat(match_data, stat_name, home_val, away_val)
                                        except Exception as e:
                                            logger.debug(f"Error processing row in alternative stats extraction: {e}")
                    except Exception as e:
                        logger.debug(f"Error extracting {stat_name} stats: {e}")
        
        except Exception as e:
            logger.error(f"Error scraping match {match_url}: {e}")
        
        return match_data
    
    def process_stat(self, match_data, stat_name, home_val, away_val):
        """Process a statistic and update match_data dictionary"""
        logger.debug(f"Processing stat: {stat_name} - Home: {home_val}, Away: {away_val}")
        
        # Possession
        if any(x in stat_name for x in ['possession', 'poss']):
            match_data['home_possession'] = self.get_int_value(home_val)
            match_data['away_possession'] = self.get_int_value(away_val)
        
        # Shots
        elif any(x == stat_name for x in ['shots', 'total shots']):
            match_data['home_shots'] = self.get_int_value(home_val)
            match_data['away_shots'] = self.get_int_value(away_val)
        
        # Shots on target
        elif any(x in stat_name for x in ['on target', 'shots on target']):
            match_data['home_shots_on_target'] = self.get_int_value(home_val)
            match_data['away_shots_on_target'] = self.get_int_value(away_val)
        
        # Big chances
        elif 'big chance' in stat_name:
            match_data['home_big_chances'] = self.get_int_value(home_val)
            match_data['away_big_chances'] = self.get_int_value(away_val)
        
        # Passes
        elif any(x in stat_name for x in ['passes', 'total pass']) and 'accuracy' not in stat_name and 'success' not in stat_name:
            match_data['home_passes'] = self.get_int_value(home_val)
            match_data['away_passes'] = self.get_int_value(away_val)
        
        # Pass completion
        elif any(x in stat_name for x in ['pass acc', 'pass success', 'pass compl', 'pass%']):
            match_data['home_pass_pct'] = self.get_float_value(home_val)
            match_data['away_pass_pct'] = self.get_float_value(away_val)
        
        # Corners
        elif 'corner' in stat_name:
            match_data['home_corners'] = self.get_int_value(home_val)
            match_data['away_corners'] = self.get_int_value(away_val)
        
        # Fouls
        elif stat_name == 'fouls' or ('foul' in stat_name and 'committed' in stat_name):
            match_data['home_fouls'] = self.get_int_value(home_val)
            match_data['away_fouls'] = self.get_int_value(away_val)
        
        # Yellow cards
        elif any(x in stat_name for x in ['yellow', 'caution', 'yellow card']):
            match_data['home_yellow_cards'] = self.get_int_value(home_val)
            match_data['away_yellow_cards'] = self.get_int_value(away_val)
        
        # Red cards
        elif any(x in stat_name for x in ['red', 'dismissal', 'send off', 'red card']):
            match_data['home_red_cards'] = self.get_int_value(home_val)
            match_data['away_red_cards'] = self.get_int_value(away_val)
    
    def scrape_team_matches(self, team_url, num_matches=7):
        """Scrape recent matches for a specific team"""
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {num_matches} recent matches for {team_name}")
        
        # Get match links
        match_links = self.get_match_links(team_url, num_matches)
        logger.info(f"Found {len(match_links)} match links")
        
        # Scrape data for each match
        match_data_list = []
        for link in match_links:
            try:
                match_data = self.scrape_match_data(link, team_name)
                match_data_list.append(match_data)
                # Be nice to the server
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error scraping match {link}: {e}")
        
        return match_data_list
    
    def save_to_csv(self, match_data_list, output_file):
        """Save match data to CSV file"""
        if not match_data_list:
            logger.warning("No data to save")
            return
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Get all possible field names from all matches
        fieldnames = set()
        for match in match_data_list:
            fieldnames.update(match.keys())
        
        # Order fields in a logical way
        ordered_fields = [
            'match_id', 'team', 'venue', 'date', 'competition', 'result',
            'home_team', 'away_team', 'home_goals', 'away_goals', 'home_xg', 'away_xg',
            'home_possession', 'away_possession', 'home_shots', 'away_shots',
            'home_shots_on_target', 'away_shots_on_target', 'home_big_chances', 'away_big_chances',
            'home_passes', 'away_passes', 'home_pass_pct', 'away_pass_pct',
            'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
            'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards',
            'match_url'
        ]
        
        # Add any missing fields that weren't in our ordered list
        fieldnames = [f for f in ordered_fields if f in fieldnames] + [f for f in fieldnames if f not in ordered_fields]
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(match_data_list)
        
        logger.info(f"Saved {len(match_data_list)} matches to {output_file}")
        
        # Also save as JSON for easier debugging
        json_file = output_file.replace('.csv', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(match_data_list, f, indent=2)
        
        logger.info(f"Saved JSON data to {json_file}")

def main():
    """Main function to run the scraper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape recent football match data from FBref')
    parser.add_argument('--team', type=str, help='FBref team URL (e.g., https://fbref.com/en/equipes/822bd0ba/Liverpool-Stats)')
    parser.add_argument('--matches', type=int, default=7, help='Number of recent matches to scrape (default: 7)')
    parser.add_argument('--output', type=str, default='data', help='Output directory (default: data)')
    parser.add_argument('--no-headless', action='store_true', help='Run Chrome in visible mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if not args.team:
        parser.error("--team is required")
    
    # Initialize scraper
    scraper = FootballMatchScraper(headless=not args.no_headless, debug=args.debug)
    scraper.start_driver()
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Scrape team
        team_name = scraper.extract_team_name(args.team)
        match_data_list = scraper.scrape_team_matches(args.team, args.matches)
        
        # Save to CSV
        output_file = f"{args.output}/{team_name}_matches_{timestamp}.csv"
        scraper.save_to_csv(match_data_list, output_file)
        
    finally:
        # Always close the driver
        scraper.close_driver()

if __name__ == "__main__":
    main()