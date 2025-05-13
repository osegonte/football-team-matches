"""
FBRef Football Match Data Scraper
--------------------------------
A robust, high-performance scraper for extracting match data from FBref.com.
Features:
- Direct table extraction using BeautifulSoup with proper comment handling
- Comprehensive match statistics including xG, possession, shots, etc.
- Support for scraping multiple teams and seasons
- Efficient caching to reduce server load
- Detailed CSV and JSON output

Usage:
  python scraper.py --team https://fbref.com/en/squads/822bd0ba/Liverpool-Stats
"""

import os
import re
import time
import json
import logging
import hashlib
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from urllib.parse import urlparse
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try importing required packages
try:
    import pandas as pd
    import requests
    from tqdm import tqdm
    from dateutil import parser
    from bs4 import BeautifulSoup, Comment
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    logger.error("Please install required packages with:")
    logger.error("pip install pandas requests beautifulsoup4 tqdm python-dateutil lxml")
    exit(1)


class Cache:
    """Simple disk-based cache for web responses"""
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """Initialize the cache
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"
    
    def get(self, url: str) -> Optional[str]:
        """Get cached content for URL if exists and not expired"""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(hours=self.ttl_hours):
            return None
            
        # Read and return cached content
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def set(self, url: str, content: str) -> None:
        """Save content to cache"""
        cache_path = self._get_cache_path(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def clear(self) -> None:
        """Clear all cache files"""
        for cache_file in self.cache_dir.glob("*.html"):
            cache_file.unlink()
        logger.info(f"Cleared cache directory: {self.cache_dir}")


class FootballMatchScraper:
    """A comprehensive football match data scraper for FBRef"""
    
    def __init__(
        self, 
        cache_enabled: bool = True,
        cache_ttl: int = 24,
        user_agent: str = None,
        debug: bool = False,
        force_cache_clear: bool = False
    ):
        """Initialize the football match scraper
        
        Args:
            cache_enabled: Enable response caching
            cache_ttl: Cache time-to-live in hours
            user_agent: Custom user agent string
            debug: Enable debug logging
            force_cache_clear: Force clearing the cache on init
        """
        self.debug = debug
        self.cache_enabled = cache_enabled
        self.cache = Cache(ttl_hours=cache_ttl) if cache_enabled else None
        
        # Clear cache if requested
        if force_cache_clear and self.cache:
            self.cache.clear()
        
        # Default user agent if none provided - use a more browser-like one
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/121.0.0.0 Safari/537.36'
        )
        
        # Session for requests with better headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://fbref.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def fetch_url(self, url: str) -> str:
        """Fetch URL content with caching support
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content
        """
        # Normalize URL to use /en/squads/ path
        url = url.replace("/en/equipes/", "/en/squads/")
        
        # Check cache first
        if self.cache_enabled:
            cached = self.cache.get(url)
            if cached:
                logger.debug(f"Using cached version of {url}")
                return cached
        
        # Fetch using requests
        logger.debug(f"Fetching {url}")
        try:
            # Add a small delay to avoid rate limiting
            time.sleep(2)
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html = response.text
            
            # Save raw HTML for debugging if needed
            if self.debug:
                debug_file = f"debug_{hashlib.md5(url.encode()).hexdigest()[:8]}.html"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.debug(f"Saved raw HTML to {debug_file}")
            
            # Cache the result
            if html and self.cache_enabled:
                self.cache.set(url, html)
            
            return html
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""
    
    def extract_team_name(self, url: str) -> str:
        """Extract team name from URL
        
        Args:
            url: Team URL
            
        Returns:
            Team name
        """
        # Normalize URL to use /en/squads/ path
        url = url.replace("/en/equipes/", "/en/squads/")
        
        # Try to extract from URL path
        match = re.search(r'/([^/]+)-Stats$', url)
        if match:
            return match.group(1).replace('-', ' ')
        
        # Alternative: try to fetch and extract from title
        html = self.fetch_url(url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            if title:
                match = re.search(r'^(.+?) Stats', title.text)
                if match:
                    return match.group(1)
        
        return "Unknown Team"
    
    def uncomment_html(self, html_content: str) -> str:
        """Extract contents of HTML comments that contain tables or divs
        
        Args:
            html_content: HTML to process
            
        Returns:
            Processed HTML with comments uncommented
        """
        # Pattern to find commented-out tables, divs or sections
        pattern = re.compile(r'<!--\s*(<(?:table|div|section)[^>]*>.*?</(?:table|div|section)>)\s*-->', re.DOTALL)
        
        # Replace comments with their content
        return pattern.sub(r'\1', html_content)
    
    def get_match_links(self, team_url: str, num_matches: int = 7) -> List[Dict[str, str]]:
        """Get the most recent match links for a team
        
        Args:
            team_url: Team URL
            num_matches: Number of recent matches to get
            
        Returns:
            List of dictionaries containing match details
        """
        # Normalize URL to use /en/squads/ path
        team_url = team_url.replace("/en/equipes/", "/en/squads/")
        matches_url = f"{team_url}/matchlogs/all_comps/schedule/"
        logger.info(f"Loading matches page: {matches_url}")
        
        # Fetch HTML
        html = self.fetch_url(matches_url)
        if not html:
            logger.error("Failed to fetch matches page")
            return []
            
        # Process HTML to extract comments BEFORE parsing
        html = self.uncomment_html(html)
        
        # Create soup from processed HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try BeautifulSoup approach first (more reliable)
        match_links = []
        
        # Look for match report links in the schedule table
        for a in soup.select('td[data-stat="match_report"] a'):
            if a.text.strip() == "Match Report":
                href = a.get('href', '')
                match_id = href.split('/')[2] if len(href.split('/')) > 2 else None
                
                if match_id:
                    # Get parent row for additional details
                    row = a.find_parent('tr')
                    if not row:
                        continue
                        
                    # Get date and competition info
                    date_cell = row.select_one('td[data-stat="date"]')
                    comp_cell = row.select_one('td[data-stat="comp"]')
                    home_cell = row.select_one('td[data-stat="home_team"]')
                    away_cell = row.select_one('td[data-stat="away_team"]')
                    
                    match_details = {
                        'match_id': match_id,
                        'date': date_cell.text.strip() if date_cell else None,
                        'competition': comp_cell.text.strip() if comp_cell else None,
                        'home_team': home_cell.text.strip() if home_cell else None,
                        'away_team': away_cell.text.strip() if away_cell else None,
                        'match_url': "https://fbref.com" + href if href.startswith('/') else href
                    }
                    
                    match_links.append(match_details)
                    
                    if len(match_links) >= num_matches:
                        break
        
        # If BeautifulSoup approach failed, try pandas as fallback
        if not match_links:
            logger.debug("BeautifulSoup approach found no links, trying pandas fallback")
            try:
                # Use pandas to parse HTML tables
                tables = pd.read_html(io.StringIO(html))
                
                # Find the Scores & Fixtures table
                matches_df = None
                for table in tables:
                    if len(table.columns) >= 7 and 'Date' in table.columns and 'Match Report' in table.columns:
                        matches_df = table
                        break
                
                if matches_df is not None:
                    # Clean up: drop rows with no date (separators)
                    matches_df = matches_df.dropna(subset=["Date"])
                    
                    # Get match report links (URLs aren't in the DataFrame)
                    report_links = {}
                    for a in soup.select('td[data-stat="match_report"] a'):
                        if a.text.strip() == "Match Report":
                            match_id = a.get('href', '').split('/')[2] if len(a.get('href', '').split('/')) > 2 else None
                            if match_id:
                                report_links[match_id] = "https://fbref.com" + a.get('href') if a.get('href').startswith('/') else a.get('href')
                    
                    # Process matches
                    for _, row in matches_df.iterrows():
                        for match_id, url in report_links.items():
                            match_details = {
                                'match_id': match_id,
                                'date': str(row['Date']) if 'Date' in row else None,
                                'competition': str(row['Comp']) if 'Comp' in row else None,
                                'home_team': str(row['Home']) if 'Home' in row else None,
                                'away_team': str(row['Away']) if 'Away' in row else None,
                                'match_url': url
                            }
                            match_links.append(match_details)
                            
                            if len(match_links) >= num_matches:
                                break
                        
                        if len(match_links) >= num_matches:
                            break
            except Exception as e:
                logger.error(f"Error using pandas fallback: {e}")
        
        logger.info(f"Found {len(match_links)} match links")
        return match_links
    
    def scrape_match_data(self, match_details: Dict[str, str], team_name: str) -> Dict[str, Any]:
        """Scrape data for a specific match
        
        Args:
            match_details: Dictionary containing match details
            team_name: Team name
            
        Returns:
            Dictionary of match data
        """
        match_url = match_details['match_url']
        match_id = match_details['match_id']
        
        logger.info(f"Scraping match: {match_url}")
        
        # Initialize match data with details we already have
        match_data = {
            'match_id': match_id,
            'team': team_name,
            'date': match_details.get('date'),
            'competition': match_details.get('competition'),
            'home_team': match_details.get('home_team'),
            'away_team': match_details.get('away_team'),
            'match_url': match_url,
            # Initialize all stats fields to None
            'venue': None,
            'result': None,
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
        }
        
        # Fetch the match page
        html = self.fetch_url(match_url)
        if not html:
            logger.error(f"Failed to fetch match page: {match_url}")
            return match_data
        
        # Uncomment HTML to expose tables in comments
        html = self.uncomment_html(html)
        
        # Create BeautifulSoup object for initial parsing
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract basic match info
        self.extract_basic_match_info(soup, match_data)
        
        # Determine venue and result
        self.determine_venue_result(team_name, match_data)
        
        # Extract match statistics using BeautifulSoup
        self.extract_match_stats(soup, match_data)
        
        # Fallback to pandas approach if stats are still missing
        if all(match_data[key] is None for key in ['home_possession', 'home_shots']):
            self.extract_match_stats_with_pandas(html, match_data)
        
        return match_data
    
    def extract_basic_match_info(self, soup: BeautifulSoup, match_data: Dict[str, Any]) -> None:
        """Extract basic match information from the soup
        
        Args:
            soup: BeautifulSoup object
            match_data: Match data dictionary to update
        """
        # Extract the scorebox
        scorebox = soup.select_one('.scorebox')
        if scorebox:
            # Get team names
            teams = scorebox.select('.teams .team a[itemprop="name"]')
            if len(teams) >= 2:
                match_data['home_team'] = teams[0].text.strip()
                match_data['away_team'] = teams[1].text.strip()
            
            # Get scores
            scores = scorebox.select('.score')
            if len(scores) >= 2:
                try:
                    match_data['home_goals'] = int(scores[0].text.strip())
                    match_data['away_goals'] = int(scores[1].text.strip())
                except (ValueError, TypeError):
                    pass
            
            # Get match date
            date_div = soup.select_one('.scorebox_meta .venuetime')
            if date_div:
                match_data['date'] = date_div.text.strip()
            
            # Get competition
            comp_div = soup.select_one('.scorebox_meta div:first-child')
            if comp_div:
                match_data['competition'] = comp_div.text.strip()
            
            # Get xG if available
            for div in soup.select('.scorebox_meta div'):
                if 'xG' in div.text:
                    xg_values = re.findall(r'(\d+\.\d+)', div.text)
                    if len(xg_values) >= 2:
                        match_data['home_xg'] = float(xg_values[0])
                        match_data['away_xg'] = float(xg_values[1])
                    break
    
    def determine_venue_result(self, team_name: str, match_data: Dict[str, Any]) -> None:
        """Determine venue and result for the specified team
        
        Args:
            team_name: Team name
            match_data: Match data dictionary to update
        """
        if not match_data['home_team'] or not match_data['away_team']:
            return
        
        # Determine if the team is home or away
        home_team_lower = match_data['home_team'].lower()
        away_team_lower = match_data['away_team'].lower()
        team_name_lower = team_name.lower()
        
        # Check for team name in home or away team
        if team_name_lower in home_team_lower:
            match_data['venue'] = 'Home'
            
            # Determine result if scores are available
            if match_data['home_goals'] is not None and match_data['away_goals'] is not None:
                if match_data['home_goals'] > match_data['away_goals']:
                    match_data['result'] = 'W'
                elif match_data['home_goals'] < match_data['away_goals']:
                    match_data['result'] = 'L'
                else:
                    match_data['result'] = 'D'
                    
        elif team_name_lower in away_team_lower:
            match_data['venue'] = 'Away'
            
            # Determine result if scores are available
            if match_data['home_goals'] is not None and match_data['away_goals'] is not None:
                if match_data['away_goals'] > match_data['home_goals']:
                    match_data['result'] = 'W'
                elif match_data['away_goals'] < match_data['home_goals']:
                    match_data['result'] = 'L'
                else:
                    match_data['result'] = 'D'
    
    def extract_match_stats(self, soup: BeautifulSoup, match_data: Dict[str, Any]) -> None:
        """Extract match statistics using BeautifulSoup
        
        Args:
            soup: BeautifulSoup object
            match_data: Match data dictionary to update
        """
        # Look for the stats table - multiple possible locations
        stats_table = None
        
        # Try different selectors for stats table
        for selector in [
            '#team_stats',
            '#all_shots_tbl',
            'table[id*="stats"]',
            'table[class*="stats"]',
            'div[id*="stats"] table',
            'div[class*="stat"] table'
        ]:
            tables = soup.select(selector)
            if tables:
                stats_table = tables[0]
                break
        
        if not stats_table:
            logger.debug("Could not find stats table with BeautifulSoup")
            return
        
        # Process each row in the stats table
        for row in stats_table.select('tr'):
            # Skip rows without TDs
            if not row.select('td'):
                continue
            
            # Get all cells - usually stat name, home value, away value
            cells = row.select('td, th')
            if len(cells) < 3:
                continue
            
            # Get the stat name, home value, and away value
            stat_name = cells[0].text.strip().lower()
            home_val = cells[1].text.strip() if len(cells) > 1 else ""
            away_val = cells[2].text.strip() if len(cells) > 2 else ""
            
            # Process different stats
            if any(term in stat_name for term in ['possession', 'poss']):
                match_data['home_possession'] = self.extract_percentage(home_val)
                match_data['away_possession'] = self.extract_percentage(away_val)
                
            elif any(term in stat_name for term in ['shots', 'total shots']):
                match_data['home_shots'] = self.extract_integer(home_val)
                match_data['away_shots'] = self.extract_integer(away_val)
                
            elif any(term in stat_name for term in ['on target', 'on-target', 'shots on target']):
                match_data['home_shots_on_target'] = self.extract_integer(home_val)
                match_data['away_shots_on_target'] = self.extract_integer(away_val)
                
            elif 'big chance' in stat_name:
                match_data['home_big_chances'] = self.extract_integer(home_val)
                match_data['away_big_chances'] = self.extract_integer(away_val)
                
            elif any(term == stat_name for term in ['passes', 'total pass']) and 'accuracy' not in stat_name:
                match_data['home_passes'] = self.extract_integer(home_val)
                match_data['away_passes'] = self.extract_integer(away_val)
                
            elif any(term in stat_name for term in ['pass acc', 'pass%', 'pass completion']):
                match_data['home_pass_pct'] = self.extract_percentage(home_val)
                match_data['away_pass_pct'] = self.extract_percentage(away_val)
                
            elif 'corner' in stat_name:
                match_data['home_corners'] = self.extract_integer(home_val)
                match_data['away_corners'] = self.extract_integer(away_val)
                
            elif 'foul' in stat_name:
                match_data['home_fouls'] = self.extract_integer(home_val)
                match_data['away_fouls'] = self.extract_integer(away_val)
                
            elif any(term in stat_name for term in ['yellow', 'caution', 'yellow card']):
                match_data['home_yellow_cards'] = self.extract_integer(home_val)
                match_data['away_yellow_cards'] = self.extract_integer(away_val)
                
            elif any(term in stat_name for term in ['red', 'dismissal', 'send off', 'red card']):
                match_data['home_red_cards'] = self.extract_integer(home_val)
                match_data['away_red_cards'] = self.extract_integer(away_val)
    
    def extract_match_stats_with_pandas(self, html: str, match_data: Dict[str, Any]) -> None:
        """Extract match statistics using pandas.read_html as fallback
        
        Args:
            html: HTML content
            match_data: Match data dictionary to update
        """
        try:
            # Try to find stats tables using pandas
            tables = pd.read_html(io.StringIO(html))
            
            for table in tables:
                # Skip tables that are too small
                if len(table.columns) < 3 or len(table) < 2:
                    continue
                
                # Convert first column to string to avoid errors with numeric values
                first_col = table.iloc[:, 0].astype(str).str.lower()
                
                # Check if common stat terms exist in the first column
                stats_terms = ['possession', 'shots', 'on target', 'pass', 'corner', 'foul', 'yellow', 'red']
                
                if any(term in ' '.join(first_col) for term in stats_terms):
                    # Process each stat row
                    for _, row in table.iterrows():
                        stat_name = str(row.iloc[0]).lower()
                        
                        # Get home and away values
                        home_val = row.iloc[1] if len(row) > 1 else None
                        away_val = row.iloc[2] if len(row) > 2 else None
                        
                        # Process stats
                        if 'possession' in stat_name or 'poss' in stat_name:
                            match_data['home_possession'] = self.extract_percentage(home_val)
                            match_data['away_possession'] = self.extract_percentage(away_val)
                            
                        elif stat_name in ['shots', 'total shots']:
                            match_data['home_shots'] = self.extract_integer(home_val)
                            match_data['away_shots'] = self.extract_integer(away_val)
                            
                        elif 'on target' in stat_name or 'shots on target' in stat_name:
                            match_data['home_shots_on_target'] = self.extract_integer(home_val)
                            match_data['away_shots_on_target'] = self.extract_integer(away_val)
                            
                        elif 'big chance' in stat_name:
                            match_data['home_big_chances'] = self.extract_integer(home_val)
                            match_data['away_big_chances'] = self.extract_integer(away_val)
                            
                        elif stat_name in ['passes', 'total pass'] and 'accuracy' not in stat_name:
                            match_data['home_passes'] = self.extract_integer(home_val)
                            match_data['away_passes'] = self.extract_integer(away_val)
                            
                        elif any(x in stat_name for x in ['pass acc', 'pass%', 'pass completion']):
                            match_data['home_pass_pct'] = self.extract_percentage(home_val)
                            match_data['away_pass_pct'] = self.extract_percentage(away_val)
                            
                        elif 'corner' in stat_name:
                            match_data['home_corners'] = self.extract_integer(home_val)
                            match_data['away_corners'] = self.extract_integer(away_val)
                            
                        elif 'foul' in stat_name:
                            match_data['home_fouls'] = self.extract_integer(home_val)
                            match_data['away_fouls'] = self.extract_integer(away_val)
                            
                        elif any(x in stat_name for x in ['yellow', 'caution', 'yellow card']):
                            match_data['home_yellow_cards'] = self.extract_integer(home_val)
                            match_data['away_yellow_cards'] = self.extract_integer(away_val)
                            
                        elif any(x in stat_name for x in ['red', 'dismissal', 'send off', 'red card']):
                            match_data['home_red_cards'] = self.extract_integer(home_val)
                            match_data['away_red_cards'] = self.extract_integer(away_val)
        
        except Exception as e:
            logger.error(f"Error extracting stats with pandas: {e}")
    
    def extract_integer(self, value: Any) -> Optional[int]:
        """Extract integer from value
        
        Args:
            value: Value to extract integer from
            
        Returns:
            Integer or None
        """
        if value is None:
            return None
            
        try:
            # Convert to string and remove non-digit characters except minus sign
            value_str = str(value).strip()
            # Remove commas used as thousands separators
            value_str = value_str.replace(',', '')
            # Extract digits and minus sign
            digits = re.search(r'-?\d+', value_str)
            if digits:
                return int(digits.group())
            return None
        except (ValueError, TypeError):
            return None
    
    def extract_percentage(self, value: Any) -> Optional[float]:
        """Extract percentage from value
        
        Args:
            value: Value to extract percentage from
            
        Returns:
            Percentage value (0-100) or None
        """
        if value is None:
            return None
            
        try:
            # Convert to string and handle percentage sign
            value_str = str(value).strip()
            # Remove percentage sign if present
            value_str = value_str.replace('%', '')
            # Extract decimal value
            return float(value_str)
        except (ValueError, TypeError):
            return None
    
    def scrape_team_matches(self, team_url: str, num_matches: int = 7) -> List[Dict[str, Any]]:
        """Scrape recent matches for a specific team
        
        Args:
            team_url: Team URL
            num_matches: Number of recent matches to scrape
            
        Returns:
            List of match data dictionaries
        """
        # Normalize URL to use /en/squads/ path
        team_url = team_url.replace("/en/equipes/", "/en/squads/")
        
        # Extract team name
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {num_matches} recent matches for {team_name}")
        
        # Get match links
        match_links = self.get_match_links(team_url, num_matches)
        
        if not match_links:
            logger.error("No match links found")
            return []
        
        # Create progress bar
        pbar = tqdm(total=len(match_links), desc="Scraping matches", unit="match")
        
        # Scrape data for each match
        match_data_list = []
        for match_details in match_links:
            try:
                match_data = self.scrape_match_data(match_details, team_name)
                if match_data:
                    match_data_list.append(match_data)
                # Update progress bar
                pbar.update(1)
                # Be nice to the server
                time.sleep(3)  # Increased delay to avoid rate limiting
            except Exception as e:
                logger.error(f"Error scraping match {match_details['match_url']}: {e}")
                pbar.update(1)
        
        pbar.close()
        return match_data_list
    
    def save_data(self, match_data_list: List[Dict[str, Any]], output_dir: str = 'data', 
                 file_format: str = 'both') -> Tuple[Optional[str], Optional[str]]:
        """Save match data to CSV and/or JSON files
        
        Args:
            match_data_list: List of match data dictionaries
            output_dir: Output directory
            file_format: 'csv', 'json', or 'both'
            
        Returns:
            Tuple of (csv_path, json_path) or None for formats not saved
        """
        if not match_data_list:
            logger.warning("No data to save")
            return None, None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp and filename base
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        team_name = match_data_list[0]['team'].replace(' ', '_')
        base_filename = f"{team_name}_matches_{timestamp}"
        
        csv_path = None
        json_path = None
        
        # Save as CSV
        if file_format in ['csv', 'both']:
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            
            # Create DataFrame
            df = pd.DataFrame(match_data_list)
            
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
            
            # Reorder columns if they exist
            cols = [col for col in ordered_fields if col in df.columns]
            df = df[cols]
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(match_data_list)} matches to {csv_path}")
        
        # Save as JSON
        if file_format in ['json', 'both']:
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(match_data_list, f, indent=2, default=str)
            
            logger.info(f"Saved JSON data to {json_path}")
        
        return csv_path, json_path
def main():
    """Main entry point for the scraper"""
    parser = argparse.ArgumentParser(
        description='Enhanced Football Match Data Scraper for FBRef',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--team', 
        type=str, 
        help='FBref team URL (e.g., https://fbref.com/en/squads/822bd0ba/Liverpool-Stats)'
    )
    
    parser.add_argument(
        '--matches', 
        type=int, 
        default=7, 
        help='Number of recent matches to scrape'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='data', 
        help='Output directory'
    )
    
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['csv', 'json', 'both'], 
        default='both',
        help='Output file format(s)'
    )
    
    parser.add_argument(
        '--no-cache', 
        action='store_true', 
        help='Disable caching'
    )
    
    parser.add_argument(
        '--clear-cache', 
        action='store_true', 
        help='Clear cache before running'
    )
    
    parser.add_argument(
        '--cache-ttl', 
        type=int, 
        default=24, 
        help='Cache time-to-live in hours'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--user-agent',
        type=str,
        help='Custom user agent string'
    )
    
    parser.add_argument(
        '--teams-file',
        type=str,
        help='Path to a text file with team URLs (one per line)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.team and not args.teams_file:
        parser.error("Either --team or --teams-file is required")
    
    # Initialize scraper
    scraper = FootballMatchScraper(
        cache_enabled=not args.no_cache,
        cache_ttl=args.cache_ttl,
        user_agent=args.user_agent,
        debug=args.debug,
        force_cache_clear=args.clear_cache
    )
    
    try:
        # Get team URLs
        team_urls = []
        if args.teams_file:
            with open(args.teams_file, 'r') as f:
                team_urls = [line.strip() for line in f if line.strip()]
        
        if args.team:
            team_urls.append(args.team)
        
        # Process each team
        for team_url in team_urls:
            # Normalize URL to use /en/squads/ path
            team_url = team_url.replace("/en/equipes/", "/en/squads/")
            
            team_name = scraper.extract_team_name(team_url)
            logger.info(f"Processing team: {team_name}")
            
            # Scrape team matches
            match_data_list = scraper.scrape_team_matches(team_url, args.matches)
            
            # Save data
            if match_data_list:
                scraper.save_data(match_data_list, args.output, args.format)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()