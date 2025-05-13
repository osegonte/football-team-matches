#!/usr/bin/env python3
"""
FBRef Football Match Data Scraper
--------------------------------
A robust, high-performance scraper for extracting match data from FBRef.com.
Features:
- Asynchronous operation for improved performance
- Intelligent caching to reduce server load
- Comprehensive error handling with multiple fallback mechanisms
- Rich progress visualization
- Advanced data extraction and validation

Usage:
  python improved_fbref_scraper.py --team https://fbref.com/en/equipes/822bd0ba/Liverpool-Stats
"""

import os
import re
import time
import json
import logging
import hashlib
import asyncio
import argparse
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

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
    import aiohttp
    import aiofiles
    from tqdm import tqdm
    from dateutil import parser
    from bs4 import BeautifulSoup, Comment
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    logger.error("Please install required packages with:")
    logger.error("pip install selenium webdriver-manager beautifulsoup4 pandas python-dateutil tqdm aiohttp aiofiles")
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
    
    async def get(self, url: str) -> Optional[str]:
        """Get cached content for URL if exists and not expired"""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(hours=self.ttl_hours):
            return None
            
        # Read and return cached content
        async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
            return await f.read()
    
    async def set(self, url: str, content: str) -> None:
        """Save content to cache"""
        cache_path = self._get_cache_path(url)
        async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
            await f.write(content)


class FootballMatchScraper:
    """A comprehensive football match data scraper for FBRef"""
    
    def __init__(
        self, 
        headless: bool = True, 
        debug: bool = False, 
        cache_enabled: bool = True,
        cache_ttl: int = 24,
        user_agent: str = None
    ):
        """Initialize the football match scraper
        
        Args:
            headless: Run browser in headless mode
            debug: Enable debug logging
            cache_enabled: Enable response caching
            cache_ttl: Cache time-to-live in hours
            user_agent: Custom user agent string
        """
        self.driver = None
        self.headless = headless
        self.debug = debug
        self.cache_enabled = cache_enabled
        self.cache = Cache(ttl_hours=cache_ttl) if cache_enabled else None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.session = None  # Will be initialized in start_session
        
        # Default user agent if none provided
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/121.0.0.0 Safari/537.36'
        )
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    async def start_session(self):
        """Start aiohttp session for async requests"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={'User-Agent': self.user_agent}
            )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def start_driver(self):
        """Start the Selenium WebDriver with optimized settings"""
        options = Options()
        
        if self.headless:
            options.add_argument('--headless=new')
        
        # Performance and stability options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-infobars')
        options.add_argument('--disable-notifications')
        options.add_argument(f'user-agent={self.user_agent}')
        
        # Extra performance tweaks
        options.add_argument('--blink-settings=imagesEnabled=false')  # Disable images
        
        # Handle platform-specific issues
        if platform.system() == 'Linux':
            options.add_argument('--disable-setuid-sandbox')
            options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)  # 30 second timeout
        
        # Set window size to a reasonable desktop size
        self.driver.set_window_size(1366, 768)
        
        logger.debug("WebDriver started successfully")
    
    def close_driver(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.debug("WebDriver closed")
    
    async def fetch_url(self, url: str, with_selenium: bool = False) -> str:
        """Fetch URL content with caching support
        
        Args:
            url: URL to fetch
            with_selenium: Use Selenium instead of aiohttp
            
        Returns:
            HTML content
        """
        # Check cache first
        if self.cache_enabled:
            cached = await self.cache.get(url)
            if cached:
                logger.debug(f"Using cached version of {url}")
                return cached
        
        # Fetch using appropriate method
        if with_selenium:
            html = await self._fetch_with_selenium(url)
        else:
            html = await self._fetch_with_aiohttp(url)
        
        # Cache the result
        if html and self.cache_enabled:
            await self.cache.set(url, html)
        
        return html
    
    async def _fetch_with_aiohttp(self, url: str) -> str:
        """Fetch URL using aiohttp
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content
        """
        await self.start_session()
        
        try:
            async with self.session.get(url, timeout=30) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            logger.debug(f"Error fetching {url} with aiohttp: {e}")
            # Fallback to selenium if aiohttp fails
            return await self._fetch_with_selenium(url)
    
    async def _fetch_with_selenium(self, url: str) -> str:
        """Fetch URL using Selenium
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content
        """
        if not self.driver:
            self.start_driver()
        
        def _fetch():
            self.driver.get(url)
            time.sleep(3)  # Wait for JavaScript to execute
            return self.driver.page_source
        
        # Run in thread pool to not block the event loop
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _fetch
        )
    
    def extract_team_name(self, url: str) -> str:
        """Extract team name from URL
        
        Args:
            url: Team URL
            
        Returns:
            Team name
        """
        # Try to extract from URL path
        match = re.search(r'/([^/]+)-Stats$', url)
        if match:
            return match.group(1).replace('-', ' ')
        
        # Alternative method: get from page title
        if self.driver:
            try:
                self.driver.get(url)
                time.sleep(2)
                title = self.driver.title
                match = re.search(r'^(.+?) Stats', title)
                if match:
                    return match.group(1)
            except Exception as e:
                logger.debug(f"Error extracting team name from page title: {e}")
        
        return "Unknown Team"
    
    async def get_match_links(self, team_url: str, num_matches: int = 7) -> List[str]:
        """Get the most recent match links for a team
        
        Args:
            team_url: Team URL
            num_matches: Number of recent matches to get
            
        Returns:
            List of match URLs
        """
        matches_url = f"{team_url}/matchlogs/all_comps/schedule/"
        logger.info(f"Loading matches page: {matches_url}")
        
        # This page needs JavaScript, so we use Selenium
        html = await self.fetch_url(matches_url, with_selenium=True)
        soup = BeautifulSoup(html, 'html.parser')
        
        match_links = []
        try:
            # Find matches table
            match_table = soup.find('table', {'id': 'matchlogs_for'})
            if not match_table:
                logger.error("Could not find matches table")
                return []
            
            # Extract match links
            rows = match_table.select('tbody > tr')
            for row in rows:
                try:
                    link_cell = row.find('td', {'data-stat': 'match_report'})
                    if not link_cell:
                        continue
                        
                    link = link_cell.find('a')
                    if link and link.text == "Match Report":
                        match_links.append(link['href'])
                        if len(match_links) >= num_matches:
                            break
                except Exception as e:
                    logger.debug(f"Error processing match row: {e}")
                    continue
            
            # Make links absolute if they're relative
            match_links = [
                f"https://fbref.com{link}" if link.startswith('/') else link 
                for link in match_links
            ]
            
        except Exception as e:
            logger.error(f"Error getting match links: {e}")
        
        return match_links
    
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
    
    def extract_match_id(self, match_url: str) -> str:
        """Extract match ID from URL
        
        Args:
            match_url: Match URL
            
        Returns:
            Match ID
        """
        # Try to extract from URL path
        url_parts = match_url.split('/')
        if len(url_parts) >= 6:
            return url_parts[5]
        
        # If that fails, use a hash of the URL
        return hashlib.md5(match_url.encode()).hexdigest()[:8]
    
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Convert date string to ISO format
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            ISO format date (YYYY-MM-DD) or None
        """
        if not date_str:
            return None
        
        try:
            # Remove venue time and local time phrases
            clean_str = re.sub(r'\(venue time\)|\(local time\)', '', date_str).strip()
            
            # Parse and format as ISO
            dt_obj = parser.parse(clean_str)
            return dt_obj.strftime('%Y-%m-%d')
        except Exception as e:
            logger.debug(f"Error parsing date '{date_str}': {e}")
            return None
    
    def extract_number(self, text: Union[str, int, float, None], to_int: bool = True) -> Optional[Union[int, float]]:
        """Extract numeric value from text
        
        Args:
            text: Text to extract number from
            to_int: Convert to integer if True, float if False
            
        Returns:
            Extracted number or None
        """
        if text is None:
            return None
        
        # Convert to string if not already
        text = str(text)
        
        # Remove non-numeric chars except decimal point and minus sign
        text = re.sub(r'[^\d.-]', '', text)
        
        if text.strip() == '':
            return None
        
        try:
            value = float(text)
            return int(value) if to_int else value
        except:
            return None
    
    def extract_stats_table(self, soup: BeautifulSoup) -> Optional[Union[BeautifulSoup, pd.DataFrame]]:
        """Extract the match stats table using multiple methods
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Stats table as BeautifulSoup or DataFrame, or None if not found
        """
        # Method 1: Direct table access
        stats_table = soup.find('table', {'id': lambda x: x and 'stats_' in x and 'all' in x})
        if stats_table:
            logger.debug("Found stats table with direct access")
            return stats_table
        
        # Method 2: Look in comments for tables 
        divs = soup.find_all('div', {'id': lambda x: x and 'all_stats' in x})
        for div in divs:
            comments = div.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                # Parse the comment content as HTML
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')
                if table:
                    logger.debug("Found stats table in comments")
                    return table
        
        # Method 3: Try to find by structure
        for div in soup.find_all('div', class_='table_container'):
            table = div.find('table')
            if table:
                # Check if this looks like a stats table
                th_texts = [th.text.strip().lower() for th in table.find_all('th')]
                if any(x in ' '.join(th_texts) for x in ['possession', 'shots', 'passes', 'fouls']):
                    logger.debug("Found stats table by structure")
                    return table
        
        # Method 4: Use pandas to read all tables
        try:
            html_content = str(soup)
            tables = pd.read_html(html_content)
            
            # Look for the stats table based on column names
            for table in tables:
                column_text = ' '.join(str(col).lower() for col in table.columns)
                row_text = ' '.join(str(x).lower() for x in table.iloc[:, 0] if isinstance(x, str))
                
                if any(term in column_text or term in row_text for term in 
                       ['possession', 'shot', 'pass', 'foul', 'corner']):
                    logger.debug("Found stats table using pandas")
                    return table
        except Exception as e:
            logger.debug(f"Error using pandas to find tables: {e}")
        
        logger.warning("No stats table found by any method")
        return None
    
    def extract_match_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract basic match information
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary of match information
        """
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
            date_str = date_div.text.strip()
            match_info['date'] = self.normalize_date(date_str)
            logger.debug(f"Date: {match_info['date']}")
        
        # Get competition
        comp_div = soup.select_one('.scorebox_meta div')
        if comp_div:
            match_info['competition'] = comp_div.text.strip()
            logger.debug(f"Competition: {match_info['competition']}")
        
        # Get teams and score
        try:
            # Method 1: Using score boxes
            scorebox = soup.select_one('.scorebox')
            if scorebox:
                teams = scorebox.select('a[itemprop="name"]') or scorebox.select('.teamname a')
                scores = scorebox.select('.score')
                
                if len(teams) >= 2 and len(scores) >= 2:
                    match_info['home_team'] = teams[0].text.strip()
                    match_info['away_team'] = teams[1].text.strip()
                    match_info['home_goals'] = self.extract_number(scores[0].text)
                    match_info['away_goals'] = self.extract_number(scores[1].text)
                    
                    logger.debug(f"Teams: {match_info['home_team']} vs {match_info['away_team']}")
                    logger.debug(f"Score: {match_info['home_goals']} - {match_info['away_goals']}")
                
            # Method 2: If Method 1 fails, try alternative selectors
            if not match_info['home_team'] or not match_info['away_team']:
                logger.debug("Using alternative method to get teams and scores")
                
                # Try getting from the title
                title = soup.find('title')
                if title:
                    title_text = title.text
                    match = re.search(r'(.+?)\s+vs\.\s+(.+?)\s+-', title_text)
                    if match:
                        match_info['home_team'] = match.group(1).strip()
                        match_info['away_team'] = match.group(2).strip()
                        
                        # Try to extract score from title
                        score_match = re.search(r'(\d+)-(\d+)', title_text)
                        if score_match:
                            match_info['home_goals'] = int(score_match.group(1))
                            match_info['away_goals'] = int(score_match.group(2))
                
        except Exception as e:
            logger.debug(f"Error extracting teams and score: {e}")
        
        # Get xG values
        try:
            for div in soup.select('.scorebox_meta div'):
                if 'xG' in div.text:
                    xg_values = re.findall(r'(\d+\.\d+)', div.text)
                    if len(xg_values) >= 2:
                        match_info['home_xg'] = float(xg_values[0])
                        match_info['away_xg'] = float(xg_values[1])
                        logger.debug(f"xG: {match_info['home_xg']} - {match_info['away_xg']}")
                        break
        except Exception as e:
            logger.debug(f"Error extracting xG: {e}")
        
        return match_info
    
    def determine_result(self, team_name: str, home_team: str, away_team: str, 
                        home_goals: int, away_goals: int) -> Tuple[str, str]:
        """Determine match venue and result for the specified team
        
        Args:
            team_name: Team name to determine result for
            home_team: Home team name
            away_team: Away team name
            home_goals: Home team goals
            away_goals: Away team goals
            
        Returns:
            Tuple of (venue, result)
        """
        venue = None
        result = None
        
        if not home_team or not away_team or home_goals is None or away_goals is None:
            return venue, result
        
        team_name_lower = team_name.lower()
        home_team_lower = home_team.lower()
        away_team_lower = away_team.lower()
        
        # Check if team name appears in either home or away team
        home_match = team_name_lower in home_team_lower or any(word in home_team_lower 
                                                             for word in team_name_lower.split() 
                                                             if len(word) > 3)
        away_match = team_name_lower in away_team_lower or any(word in away_team_lower 
                                                             for word in team_name_lower.split() 
                                                             if len(word) > 3)
        
        if home_match:
            venue = 'Home'
            if home_goals > away_goals:
                result = 'W'
            elif home_goals < away_goals:
                result = 'L'
            else:
                result = 'D'
        elif away_match:
            venue = 'Away'
            if away_goals > home_goals:
                result = 'W'
            elif away_goals < home_goals:
                result = 'L'
            else:
                result = 'D'
        
        return venue, result
    
    def process_stat(self, match_data: Dict[str, Any], stat_name: str, 
                     home_val: Union[str, int, float], away_val: Union[str, int, float]) -> None:
        """Process a statistic and update match_data dictionary
        
        Args:
            match_data: Match data dictionary to update
            stat_name: Name of the statistic
            home_val: Home team value
            away_val: Away team value
        """
        stat_name = str(stat_name).lower()
        logger.debug(f"Processing stat: {stat_name} - Home: {home_val}, Away: {away_val}")
        
        # Dictionary of stat patterns and their corresponding field names
        stat_patterns = {
            'possession': ('home_possession', 'away_possession', True),
            'poss': ('home_possession', 'away_possession', True),
            'shots': ('home_shots', 'away_shots', True),
            'total shots': ('home_shots', 'away_shots', True),
            'on target': ('home_shots_on_target', 'away_shots_on_target', True),
            'shots on target': ('home_shots_on_target', 'away_shots_on_target', True),
            'big chance': ('home_big_chances', 'away_big_chances', True),
            'passes': ('home_passes', 'away_passes', True),
            'total pass': ('home_passes', 'away_passes', True),
            'pass acc': ('home_pass_pct', 'away_pass_pct', False),
            'pass success': ('home_pass_pct', 'away_pass_pct', False),
            'pass compl': ('home_pass_pct', 'away_pass_pct', False),
            'pass%': ('home_pass_pct', 'away_pass_pct', False),
            'corner': ('home_corners', 'away_corners', True),
            'fouls': ('home_fouls', 'away_fouls', True),
            'foul committed': ('home_fouls', 'away_fouls', True),
            'yellow': ('home_yellow_cards', 'away_yellow_cards', True),
            'caution': ('home_yellow_cards', 'away_yellow_cards', True),
            'yellow card': ('home_yellow_cards', 'away_yellow_cards', True),
            'red': ('home_red_cards', 'away_red_cards', True),
            'dismissal': ('home_red_cards', 'away_red_cards', True),
            'send off': ('home_red_cards', 'away_red_cards', True),
            'red card': ('home_red_cards', 'away_red_cards', True),
        }
        
        # Check each pattern
        for pattern, (home_field, away_field, to_int) in stat_patterns.items():
            if pattern in stat_name:
                # Don't override if we already have a value and not dealing with pass accuracy
                if pattern not in ['pass acc', 'pass success', 'pass compl', 'pass%']:
                    if match_data.get(home_field) is not None and match_data.get(away_field) is not None:
                        continue
                        
                match_data[home_field] = self.extract_number(home_val, to_int)
                match_data[away_field] = self.extract_number(away_val, to_int)
                break
    
    def extract_match_stats(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract match statistics from HTML
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary of match statistics
        """
        stats = {
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
        
        # Extract stats from table
        stats_table = self.extract_stats_table(soup)
        
        if stats_table is not None:
            # If it's a pandas DataFrame, process it directly
            if isinstance(stats_table, pd.DataFrame):
                try:
                    # Extract stats based on the DataFrame structure
                    for index, row in stats_table.iterrows():
                        stat_name = row.iloc[0].lower() if len(row) > 0 else ''
                        home_val = row.iloc[1] if len(row) > 1 else None
                        away_val = row.iloc[2] if len(row) > 2 else None
                        
                        self.process_stat(stats, stat_name, home_val, away_val)
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
                            
                            self.process_stat(stats, stat_name, home_val, away_val)
                        except Exception as e:
                            logger.debug(f"Error processing row in stats table: {e}")
                except Exception as e:
                    logger.debug(f"Error processing HTML stats table: {e}")
        
        # Try alternative sections if we're missing key stats
        if (stats['home_possession'] is None or stats['away_possession'] is None or
                stats['home_shots'] is None or stats['away_shots'] is None):
            
            logger.debug("Trying alternative sections for stats")
            
            for section_id in ['all_shots', 'all_possession', 'all_passes', 'all_defense']:
                section = soup.find('div', {'id': section_id})
                if not section:
                    continue
                
                # Search for tables in this section
                tables = section.find_all('table')
                
                # Also look in comments
                comments = section.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    tables.extend(comment_soup.find_all('table'))
                
                # Process each table
                for table in tables:
                    rows = table.select('tbody > tr')
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
                            
                            self.process_stat(stats, stat_name, home_val, away_val)
                        except Exception as e:
                            continue
        
        return stats
    
    async def scrape_match_data(self, match_url: str, team_name: str) -> Dict[str, Any]:
        """Scrape data for a specific match
        
        Args:
            match_url: Match URL
            team_name: Team name
            
        Returns:
            Dictionary of match data
        """
        logger.info(f"Scraping match: {match_url}")
        
        # Extract match ID from URL
        match_id = self.extract_match_id(match_url)
        
        # Fetch HTML and clean it
        html = await self.fetch_url(match_url, with_selenium=True)
        html = self.uncomment_html(html)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Create match data dictionary with empty values
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
        
        try:
            # Extract basic match information
            match_info = self.extract_match_info(soup)
            match_data.update(match_info)
            
            # Determine venue and result
            venue, result = self.determine_result(
                team_name, 
                match_data['home_team'], 
                match_data['away_team'],
                match_data['home_goals'],
                match_data['away_goals']
            )
            match_data['venue'] = venue
            match_data['result'] = result
            
            # Extract match statistics
            match_stats = self.extract_match_stats(soup)
            match_data.update(match_stats)
            
        except Exception as e:
            logger.error(f"Error scraping match {match_url}: {e}")
        
        return match_data
    
    async def scrape_team_matches(self, team_url: str, num_matches: int = 7) -> List[Dict[str, Any]]:
        """Scrape recent matches for a specific team
        
        Args:
            team_url: Team URL
            num_matches: Number of recent matches to scrape
            
        Returns:
            List of match data dictionaries
        """
        # Start browser if not already started
        if not self.driver:
            self.start_driver()
        
        # Extract team name
        team_name = self.extract_team_name(team_url)
        logger.info(f"Scraping {num_matches} recent matches for {team_name}")
        
        # Get match links
        match_links = await self.get_match_links(team_url, num_matches)
        logger.info(f"Found {len(match_links)} match links")
        
        if not match_links:
            logger.error("No match links found")
            return []
        
        # Create progress bar
        pbar = tqdm(total=len(match_links), desc="Scraping matches", unit="match")
        
        # Scrape data for each match
        match_data_list = []
        for link in match_links:
            try:
                match_data = await self.scrape_match_data(link, team_name)
                if match_data:
                    match_data_list.append(match_data)
                # Update progress bar
                pbar.update(1)
                # Be nice to the server
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error scraping match {link}: {e}")
                pbar.update(1)
        
        pbar.close()
        return match_data_list
    
    async def save_data(self, match_data_list: List[Dict[str, Any]], output_dir: str = 'data', 
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
            fieldnames = [f for f in ordered_fields if f in fieldnames] + \
                         [f for f in fieldnames if f not in ordered_fields]
            
            # Convert to DataFrame and save
            df = pd.DataFrame(match_data_list)
            # Reorder columns based on our list
            df = df[[col for col in fieldnames if col in df.columns]]
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved {len(match_data_list)} matches to {csv_path}")
        
        # Save as JSON
        if file_format in ['json', 'both']:
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(match_data_list, f, indent=2, default=str)
            
            logger.info(f"Saved JSON data to {json_path}")
        
        return csv_path, json_path


async def main():
    """Main entry point for the scraper"""
    parser = argparse.ArgumentParser(
        description='Enhanced Football Match Data Scraper for FBRef',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--team', 
        type=str, 
        help='FBref team URL (e.g., https://fbref.com/en/equipes/822bd0ba/Liverpool-Stats)'
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
        '--no-headless', 
        action='store_true', 
        help='Run Chrome in visible mode'
    )
    
    parser.add_argument(
        '--no-cache', 
        action='store_true', 
        help='Disable caching'
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
        headless=not args.no_headless,
        debug=args.debug,
        cache_enabled=not args.no_cache,
        cache_ttl=args.cache_ttl,
        user_agent=args.user_agent
    )
    
    try:
        # Start necessary services
        scraper.start_driver()
        await scraper.start_session()
        
        # Get team URLs
        team_urls = []
        if args.teams_file:
            with open(args.teams_file, 'r') as f:
                team_urls = [line.strip() for line in f if line.strip()]
        
        if args.team:
            team_urls.append(args.team)
        
        # Process each team
        for team_url in team_urls:
            team_name = scraper.extract_team_name(team_url)
            logger.info(f"Processing team: {team_name}")
            
            # Scrape team matches
            match_data_list = await scraper.scrape_team_matches(team_url, args.matches)
            
            # Save data
            if match_data_list:
                await scraper.save_data(match_data_list, args.output, args.format)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        await scraper.close_session()
        scraper.close_driver()


if __name__ == "__main__":
    # Create and run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()