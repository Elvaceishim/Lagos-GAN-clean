#!/usr/bin/env python3
"""
Lagos-Specific Architecture Scraper
Focused scraper for authentic Lagos, Nigeria building styles and architecture.

This scraper targets:
1. Traditional Lagos/Yoruba architecture 
2. Lagos Island colonial buildings
3. Modern Lagos residential developments
4. Nigerian architectural styles
5. West African building patterns

Usage:
    python lagos_specific_scraper.py --dataset lagos --limit 500
    python lagos_specific_scraper.py --dataset duplex --limit 500
"""

import os
import sys
import argparse
from data_scraper import DataScraper
import logging

logger = logging.getLogger(__name__)

class LagosSpecificScraper:
    """Scraper specifically designed for Lagos/Nigerian architecture."""
    
    def __init__(self):
        self.base_scraper = DataScraper()
        
        # Lagos-specific search terms
        self.lagos_traditional_queries = [
            "lagos traditional house",
            "yoruba traditional architecture",
            "lagos island colonial building",
            "nigerian traditional house",
            "west african traditional architecture",
            "lagos heritage building",
            "yoruba compound house",
            "nigerian vernacular architecture",
            "lagos old building",
            "nigerian colonial architecture",
            "west african mud house",
            "yoruba traditional compound",
            "lagos indigenous architecture",
            "nigerian traditional dwelling",
            "west african courtyard house"
        ]
        
        self.lagos_modern_queries = [
            "lagos modern house",
            "nigerian modern architecture",
            "lagos residential building",
            "nigerian duplex house",
            "lagos contemporary house",
            "nigerian modern home",
            "west african modern architecture",
            "lagos apartment building",
            "nigerian residential design",
            "lagos housing development",
            "nigerian architectural design",
            "west african contemporary building",
            "lagos urban housing",
            "nigerian modern villa",
            "lagos real estate architecture"
        ]
        
        self.duplex_specific_queries = [
            "nigerian duplex house",
            "lagos duplex building",
            "west african duplex",
            "nigerian two-story house",
            "lagos modern duplex",
            "nigerian residential duplex",
            "west african duplex design",
            "lagos duplex architecture",
            "nigerian duplex home",
            "african duplex house",
            "lagos contemporary duplex",
            "nigerian duplex villa",
            "west african duplex structure",
            "lagos duplex development",
            "nigerian modern duplex design"
        ]
    
    def scrape_lagos_traditional(self, limit_per_query: int = 50, total_limit: int = 500):
        """Scrape traditional Lagos/Yoruba architecture."""
        logger.info(f"Starting Lagos traditional architecture scraping (total limit: {total_limit})")
        
        total_collected = 0
        for i, query in enumerate(self.lagos_traditional_queries):
            if total_collected >= total_limit:
                break
                
            remaining = min(limit_per_query, total_limit - total_collected)
            logger.info(f"Query {i+1}/{len(self.lagos_traditional_queries)}: '{query}' (limit: {remaining})")
            
            try:
                # Try Wikimedia first (most reliable)
                self.base_scraper.scrape_dataset('wikimedia', query, 'lagos', remaining)
                total_collected += remaining
                
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        logger.info(f"Completed Lagos traditional scraping. Target: {total_limit}")
        return total_collected
    
    def scrape_lagos_modern_duplex(self, limit_per_query: int = 50, total_limit: int = 500):
        """Scrape modern Lagos duplex houses."""
        logger.info(f"Starting Lagos modern duplex scraping (total limit: {total_limit})")
        
        total_collected = 0
        for i, query in enumerate(self.duplex_specific_queries):
            if total_collected >= total_limit:
                break
                
            remaining = min(limit_per_query, total_limit - total_collected)
            logger.info(f"Query {i+1}/{len(self.duplex_specific_queries)}: '{query}' (limit: {remaining})")
            
            try:
                # Try Wikimedia first
                self.base_scraper.scrape_dataset('wikimedia', query, 'duplex', remaining)
                total_collected += remaining
                
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        logger.info(f"Completed Lagos duplex scraping. Target: {total_limit}")
        return total_collected
    
    def scrape_with_api_sources(self, dataset: str, limit: int = 500):
        """Use API sources with Lagos-specific terms."""
        if dataset == 'lagos':
            queries = self.lagos_traditional_queries + self.lagos_modern_queries
        elif dataset == 'duplex':
            queries = self.duplex_specific_queries
        else:
            logger.error(f"Unknown dataset: {dataset}")
            return 0
        
        logger.info(f"Starting API-based Lagos scraping for {dataset} (limit: {limit})")
        
        # Load API keys
        unsplash_key = os.getenv('UNSPLASH_API_KEY')
        pixabay_key = os.getenv('PIXABAY_API_KEY')
        
        sources = []
        if unsplash_key and unsplash_key != "your_unsplash_api_key_here":
            sources.append('unsplash')
        if pixabay_key and pixabay_key != "your_pixabay_api_key_here":
            sources.append('pixabay')
        sources.append('wikimedia')  # Always available
        
        limit_per_source = limit // len(sources)
        limit_per_query = max(10, limit_per_source // len(queries))
        
        total_collected = 0
        for source in sources:
            for query in queries:
                if total_collected >= limit:
                    break
                    
                remaining = min(limit_per_query, limit - total_collected)
                logger.info(f"Source: {source}, Query: '{query}' (limit: {remaining})")
                
                try:
                    self.base_scraper.scrape_dataset(source, query, dataset, remaining)
                    total_collected += remaining
                except Exception as e:
                    logger.error(f"Error with {source} + '{query}': {e}")
                    continue
        
        return total_collected

def main():
    parser = argparse.ArgumentParser(description="Lagos-Specific Architecture Scraper")
    parser.add_argument("--dataset", choices=['lagos', 'duplex'], 
                        required=True, help="Target dataset")
    parser.add_argument("--limit", type=int, default=500, 
                        help="Total images to collect")
    parser.add_argument("--method", choices=['wikimedia', 'api', 'both'], 
                        default='both', help="Scraping method")
    parser.add_argument("--per-query", type=int, default=25,
                        help="Images per search query")
    
    args = parser.parse_args()
    
    scraper = LagosSpecificScraper()
    
    if args.method == 'wikimedia' or args.method == 'both':
        if args.dataset == 'lagos':
            scraper.scrape_lagos_traditional(args.per_query, args.limit)
        elif args.dataset == 'duplex':
            scraper.scrape_lagos_modern_duplex(args.per_query, args.limit)
    
    if args.method == 'api' or args.method == 'both':
        scraper.scrape_with_api_sources(args.dataset, args.limit // 2 if args.method == 'both' else args.limit)

if __name__ == "__main__":
    main()
