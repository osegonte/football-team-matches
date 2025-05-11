import os
import re
import time
import json
import logging
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
    
    def get_match_links(self, team_url, num_matches=7):
        """Get the most recent match links for a team"""
        matches_url = f"{team_url}/matchlogs/all_comps/schedule/"
        logger.info(f"Loading matches page: {matches_url}")
        self.driver.get(matches_url)
        time.sleep(2)
        
        match_links = []
        try:
            match_rows = self.driver.find_elements(By.XPATH, "//table[@id='matchlogs_for']/tbody/tr")
            
            for row in match_rows:
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
    
    def extract_comments(self, soup):
        """Extract HTML from comments"""
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if 'table' in comment:
                comment_soup = BeautifulSoup(comment, 'lxml')
                tables = comment_soup.find_all('table')
                if tables:
                    return tables
        return []
    
    def extract_match_info(self, soup):
        """Extract basic match information"""
        match_info = {
            'date': None,
            'competition': None,
            'home_team': None,
            'away_team': None,
            'home_goals': None,
            'away_goals': None,
            'home_xg': None,
            'away_xg': None,
        }
        
        # Get date
        date_div = soup.select_one('.venuetime')
        if date_div:
            match_info['date'] = date_div.text.strip()
        
        # Get competition
        comp_div = soup.select_one('.scorebox_meta div')
        if comp_div:
            match_info['competition'] = comp_div.text.strip()
        
        # Get teams and score
        teams = soup.select('.scorebox > div > .scores > div > a')
        scores = soup.select('.scorebox > div > .scores > div.score')
        
        if len(teams) >= 2:
            match_info['home_team'] = teams[0].text.strip()
            match_info['away_team'] = teams[1].text.strip()
        
        if len(scores) >= 2:
            try:
                match_info['home_goals'] = int(scores[0].text.strip())
                match_info['away_goals'] = int(scores[1].text.strip())
            except:
                pass
        
        # Get xG
        for div in soup.select('.scorebox_meta div'):
            if 'xG' in div.text:
                xg_values = re.findall(r'(\d+\.\d+)', div.text)
                if len(xg_values) >= 2:
                    match_info['home_xg'] = float(xg_values[0])
                    match_info['away_xg'] = float(xg_values[1])
                break
        
        return match_info
    
    def extract_match_stats(self, soup):
        """Extract match statistics from tables"""
        stats = {
            'home_possession': None,
            'away_possession': None,
            'home_shots': None,
            'away_shots': None,
            'home_shots_on_target': None,
            'away_shots_on_target': None,
            'home_corners': None,
            'away_corners': None,
            'home_fouls': None,
            'away_fouls': None,
            'home_yellow_cards': None,
            'away_yellow_cards': None,
            'home_red_cards': None,
            'away_red_cards': None,
            'home_passes': None,
            'away_passes': None,
            'home_pass_pct': None,
            'away_pass_pct': None,
        }
        
        # First try to get stats from regular tables
        tables = soup.find_all('table', {'id': lambda x: x and 'all_stats' in x})
        if not tables:
            # If not found, try to extract from comments
            tables = self.extract_comments(soup)
        
        for table in tables:
            rows = table.select('tbody > tr')
            for row in rows:
                stat_name_elem = row.select_one('th')
                if not stat_name_elem:
                    continue
                
                stat_name = stat_name_elem.text.strip().lower()
                values = row.select('td')
                
                if len(values) < 2:
                    continue
                
                home_val = values[0].text.strip()
                away_val = values[1].text.strip()
                
                # Process common stats
                if 'possession' in stat_name:
                    stats['home_possession'] = self.extract_number(home_val)
                    stats['away_possession'] = self.extract_number(away_val)
                elif stat_name in ['shots', 'total shots']:
                    stats['home_shots'] = self.extract_number(home_val)
                    stats['away_shots'] = self.extract_number(away_val)
                elif 'on target' in stat_name:
                    stats['home_shots_on_target'] = self.extract_number(home_val)
                    stats['away_shots_on_target'] = self.extract_number(away_val)
                elif 'corner' in stat_name:
                    stats['home_corners'] = self.extract_number(home_val)
                    stats['away_corners'] = self.extract_number(away_val)
                elif 'foul' in stat_name:
                    stats['home_fouls'] = self.extract_number(home_val)
                    stats['away_fouls'] = self.extract_number(away_val)
                elif 'yellow' in stat_name:
                    stats['home_yellow_cards'] = self.extract_number(home_val)
                    stats['away_yellow_cards'] = self.extract_number(away_val)
                elif 'red' in stat_name:
                    stats['home_red_cards'] = self.extract_number(home_val)
                    stats['away_red_cards'] = self.extract_number(away_val)
                elif stat_name in ['passes', 'total passes']:
                    stats['home_passes'] = self.extract_number(home_val)
                    stats['away_passes'] = self.extract_number(away_val)
                elif 'pass' in stat_name and ('accuracy' in stat_name or 'completion' in stat_name):
                    stats['home_pass_pct'] = self.extract_number(home_val)
                    stats['away_pass_pct'] = self.extract_number(away_val)
        
        return stats
    
    def extract_number(self, text):
        """Extract number from text, handling percentages and commas"""
        if not text:
            return None
        
        text = text.replace('%', '').replace(',', '')
        try:
            return int(float(text))
        except:
            return None
    
    def scrape_match_data(self, match_url, team_name):
        """Scrape data for a specific match"""
        self.driver.get(match_url)
        time.sleep(2)
        
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract match ID from URL
        match_id = match_url.split('/')[-3]
        
        # Get match info and stats
        match_info = self.extract_match_info(soup)
        match_stats = self.extract_match_stats(soup)
        
        # Determine venue and result based on team name
        venue = None
        result = None
        
        if match_info['home_team'] and match_info['away_team']:
            if team_name.lower() in match_info['home_team'].lower():
                venue = 'Home'
                if match_info['home_goals'] is not None and match_info['away_goals'] is not None:
                    if match_info['home_goals'] > match_info['away_goals']:
                        result = 'W'
                    elif match_info['home_goals'] < match_info['away_goals']:
                        result = 'L'
                    else:
                        result = 'D'
            else:
                venue = 'Away'
                if match_info['home_goals'] is not None and match_info['away_goals'] is not None:
                    if match_info['away_goals'] > match_info['home_goals']:
                        result = 'W'
                    elif match_info['away_goals'] < match_info['home_goals']:
                        result = 'L'
                    else:
                        result = 'D'
        
        # Combine all data
        match_data = {
            'match_id': match_id,
            'match_url': match_url,
            'date': match_info['date'],
            'competition': match_info['competition'],
            'venue': venue,
            'result': result,
            'home_team': match_info['home_team'],
            'away_team': match_info['away_team'],
            'home_goals': match_info['home_goals'],
            'away_goals': match_info['away_goals'],
            'home_xg': match_info['home_xg'],
            'away_xg': match_info['away_xg'],
            'team': team_name,
            **match_stats
        }
        
        return match_data
    
    def scrape_team_matches(self, team_url, num_matches=7):
        """Scrape recent matches for a specific team"""
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {num_matches} recent matches for {team_name}")
        
        match_links = self.get_match_links(team_url, num_matches)
        logger.info(f"Found {len(match_links)} match links")
        
        match_data_list = []
        for i, link in enumerate(match_links):
            try:
                logger.info(f"Scraping match {i+1}/{len(match_links)}: {link}")
                match_data = self.scrape_match_data(link, team_name)
                match_data_list.append(match_data)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping match {link}: {e}")
        
        return match_data_list
    
    def save_data(self, match_data_list, output_dir='data'):
        """Save match data to CSV and JSON files"""
        if not match_data_list:
            logger.warning("No data to save")
            return None, None
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        team_name = match_data_list[0]['team'].replace(' ', '_')
        
        # Save as CSV
        csv_file = f"{output_dir}/{team_name}_matches_{timestamp}.csv"
        df = pd.DataFrame(match_data_list)
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved {len(match_data_list)} matches to {csv_file}")
        
        # Save as JSON
        json_file = f"{output_dir}/{team_name}_matches_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(match_data_list, f, indent=2)
        logger.info(f"Saved JSON data to {json_file}")
        
        return csv_file, json_file

def main():
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
    
    scraper = FootballMatchScraper(headless=not args.no_headless, debug=args.debug)
    scraper.start_driver()
    
    try:
        match_data_list = scraper.scrape_team_matches(args.team, args.matches)
        scraper.save_data(match_data_list, args.output)
    finally:
        scraper.close_driver()

if __name__ == "__main__":
    main()