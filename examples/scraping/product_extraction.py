"""
Example: Product Data Extraction

Demonstrates structured extraction using JSON schemas for e-commerce data.
Shows both single product and product listing extraction patterns.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import json
import os
from flybrowser import FlyBrowser

# Schema for single product extraction
PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "string"},
        "rating": {"type": "number"},
        "reviews_count": {"type": "integer"},
        "in_stock": {"type": "boolean"},
        "description": {"type": "string"}
    },
    "required": ["name", "price"]
}

# Schema for product listing extraction
PRODUCT_LIST_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {"type": "string"},
            "rating": {"type": "number"},
            "url": {"type": "string"}
        }
    }
}


async def extract_single_product(url: str):
    """
    Extract detailed information from a single product page.
    
    Args:
        url: Product page URL
        
    Returns:
        Product dictionary
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Extract with schema
        result = await browser.extract(
            "Extract the product information from this page",
            schema=PRODUCT_SCHEMA
        )
        
        if result.success:
            print("Product Information:")
            print(json.dumps(result.data, indent=2))
            return result.data
        else:
            print(f"Extraction failed: {result.error}")
            return None


async def extract_product_listing(url: str):
    """
    Extract products from a category or search results page.
    
    Args:
        url: Category or search results URL
        
    Returns:
        List of product dictionaries
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # Extract all products on page
        result = await browser.extract(
            "Extract all product names, prices, ratings, and URLs from this page",
            schema=PRODUCT_LIST_SCHEMA
        )
        
        if result.success:
            products = result.data
            print(f"Found {len(products)} products:")
            
            # Display formatted results
            print(f"\n{'Name':<40} {'Price':<12} {'Rating':<8}")
            print("-" * 60)
            for product in products[:10]:  # Show first 10
                name = product.get('name', 'Unknown')[:38]
                price = product.get('price', 'N/A')[:10]
                rating = product.get('rating', 'N/A')
                print(f"{name:<40} {price:<12} {rating}")
            
            return products
        else:
            print(f"Extraction failed: {result.error}")
            return []


async def extract_table_data(url: str, table_description: str):
    """
    Extract tabular data from a page.
    
    Args:
        url: Page URL containing table
        table_description: Description of the table to extract
        
    Returns:
        List of row dictionaries
    """
    # Schema for stock market table (example)
    TABLE_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "symbol": {"type": "string"},
                "price": {"type": "string"},
                "change": {"type": "string"},
                "volume": {"type": "string"}
            }
        }
    }
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        result = await browser.extract(
            f"Extract the {table_description} table with all columns",
            schema=TABLE_SCHEMA
        )
        
        if result.success:
            return result.data
        else:
            print(f"Extraction failed: {result.error}")
            return []


async def conditional_extraction(url: str):
    """
    Adapt extraction strategy based on page type.
    
    Args:
        url: Any URL to analyze and extract
        
    Returns:
        Extracted data based on page type
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(url)
        
        # First, identify the page type
        page_info = await browser.extract(
            "What type of page is this? (product, article, listing, profile, other)"
        )
        
        page_type = page_info.data.lower() if page_info.success else "unknown"
        print(f"Detected page type: {page_type}")
        
        # Extract based on page type
        if "product" in page_type:
            result = await browser.extract(
                "Extract product name, price, description, and specifications",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "string"},
                        "description": {"type": "string"},
                        "specs": {"type": "object"}
                    }
                }
            )
        elif "article" in page_type:
            result = await browser.extract(
                "Extract article title, author, publish date, and main content",
                schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "date": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }
            )
        elif "listing" in page_type:
            result = await browser.extract(
                "Extract all items with their names and prices",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"}
                        }
                    }
                }
            )
        else:
            result = await browser.extract(
                "Extract the main content and key information from this page"
            )
        
        return result.data if result.success else None


async def main():
    """Main entry point demonstrating various extraction patterns."""
    print("=" * 60)
    print("Product Data Extraction Examples")
    print("=" * 60)
    
    # Example 1: Extract from a product listing page
    print("\n--- Example 1: Product Listing Extraction ---")
    # Using example.com as placeholder - replace with actual URL
    await extract_product_listing("https://news.ycombinator.com")
    
    print("\n--- Example 2: Conditional Extraction ---")
    await conditional_extraction("https://news.ycombinator.com")


if __name__ == "__main__":
    asyncio.run(main())
