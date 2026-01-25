"""
Example: Price Monitor

Monitors product prices and detects changes.
Demonstrates stateful scraping with history tracking.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from flybrowser import FlyBrowser

PRICE_HISTORY_FILE = "price_history.json"


def load_price_history() -> dict:
    """Load existing price history."""
    if Path(PRICE_HISTORY_FILE).exists():
        with open(PRICE_HISTORY_FILE) as f:
            return json.load(f)
    return {}


def save_price_history(history: dict):
    """Save price history."""
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


async def monitor_prices(products: list[dict]):
    """
    Monitor prices for a list of products.
    
    Args:
        products: List of {"name": str, "url": str}
        
    Returns:
        List of price change alerts
    """
    history = load_price_history()
    alerts = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        log_verbosity="minimal",
    ) as browser:
        for product in products:
            name = product["name"]
            url = product["url"]
            
            print(f"Checking: {name}")
            
            try:
                await browser.goto(url)
                
                # Extract current price
                result = await browser.extract(
                    "What is the current price of this product?",
                    schema={"type": "object", "properties": {"price": {"type": "string"}}}
                )
                
                if not result.success:
                    print(f"  Failed to get price: {result.error}")
                    continue
                
                current_price = result.data.get("price", "Unknown")
                timestamp = datetime.now().isoformat()
                
                # Initialize history for this product
                if name not in history:
                    history[name] = {
                        "url": url,
                        "prices": []
                    }
                
                # Get previous price
                prices = history[name]["prices"]
                previous_price = prices[-1]["price"] if prices else None
                
                # Record new price
                prices.append({
                    "price": current_price,
                    "timestamp": timestamp
                })
                
                # Keep only last 100 price points
                if len(prices) > 100:
                    history[name]["prices"] = prices[-100:]
                
                # Check for price change
                if previous_price and previous_price != current_price:
                    alert = f"{name}: {previous_price} -> {current_price}"
                    alerts.append({
                        "product": name,
                        "url": url,
                        "old_price": previous_price,
                        "new_price": current_price,
                        "timestamp": timestamp
                    })
                    print(f"  PRICE CHANGE: {alert}")
                else:
                    print(f"  Current price: {current_price}")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    # Save updated history
    save_price_history(history)
    
    # Report alerts
    if alerts:
        print("\n" + "=" * 50)
        print("PRICE ALERTS")
        print("=" * 50)
        for alert in alerts:
            print(f"  {alert['product']}: {alert['old_price']} -> {alert['new_price']}")
    else:
        print("\nNo price changes detected.")
    
    return alerts


async def compare_prices_across_sites(product_name: str, sites: list[dict]):
    """
    Compare prices for a product across multiple sites.
    
    Args:
        product_name: Name of product to search for
        sites: List of {"name": str, "url": str}
        
    Returns:
        Sorted list of prices by site
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
        log_verbosity="minimal",
    ) as browser:
        for site in sites:
            print(f"\nSearching {site['name']}...")
            
            try:
                await browser.goto(site["url"])
                
                # Search for product
                await browser.act(f"Search for '{product_name}'")
                await asyncio.sleep(2)
                
                # Get best match
                product = await browser.extract(
                    f"Find the product most similar to '{product_name}' "
                    f"and get its exact name, price, availability, and rating",
                    schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"},
                            "available": {"type": "boolean"},
                            "rating": {"type": "number"},
                            "shipping": {"type": "string"}
                        }
                    }
                )
                
                if product.success and product.data:
                    results.append({
                        "store": site["name"],
                        "url": site["url"],
                        **product.data
                    })
                    print(f"  Found: {product.data.get('name')} at {product.data.get('price')}")
                else:
                    print(f"  Product not found")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    # Sort by price (extracting numeric value)
    def extract_price(item):
        price_str = item.get("price", "$99999")
        try:
            return float(''.join(c for c in price_str if c.isdigit() or c == '.'))
        except:
            return 99999
    
    results.sort(key=extract_price)
    
    # Display comparison
    print("\n" + "=" * 60)
    print("PRICE COMPARISON RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        availability = "In Stock" if result.get("available") else "Out of Stock"
        rating = result.get("rating", "N/A")
        print(f"\n{i}. {result['store']}")
        print(f"   Product: {result.get('name')}")
        print(f"   Price: {result.get('price')}")
        print(f"   Rating: {rating}")
        print(f"   Status: {availability}")
        print(f"   Shipping: {result.get('shipping', 'Unknown')}")
    
    if results:
        best = results[0]
        print(f"\n*** BEST DEAL: {best['store']} at {best.get('price')} ***")
    
    return results


async def get_price_history_summary(product_name: str):
    """
    Get price history summary for a product.
    
    Args:
        product_name: Name of product
        
    Returns:
        Price history summary
    """
    history = load_price_history()
    
    if product_name not in history:
        print(f"No history found for: {product_name}")
        return None
    
    product_history = history[product_name]
    prices = product_history["prices"]
    
    if not prices:
        print(f"No price data for: {product_name}")
        return None
    
    # Extract numeric prices
    numeric_prices = []
    for entry in prices:
        try:
            price_str = entry["price"]
            price_num = float(''.join(c for c in price_str if c.isdigit() or c == '.'))
            numeric_prices.append(price_num)
        except:
            pass
    
    if not numeric_prices:
        print("Could not parse prices")
        return None
    
    summary = {
        "product": product_name,
        "url": product_history["url"],
        "data_points": len(prices),
        "current_price": prices[-1]["price"],
        "lowest_price": min(numeric_prices),
        "highest_price": max(numeric_prices),
        "average_price": sum(numeric_prices) / len(numeric_prices),
        "first_recorded": prices[0]["timestamp"],
        "last_recorded": prices[-1]["timestamp"],
    }
    
    print(f"\nPrice History Summary: {product_name}")
    print("-" * 40)
    print(f"  Current: {summary['current_price']}")
    print(f"  Lowest:  ${summary['lowest_price']:.2f}")
    print(f"  Highest: ${summary['highest_price']:.2f}")
    print(f"  Average: ${summary['average_price']:.2f}")
    print(f"  Data Points: {summary['data_points']}")
    
    return summary


async def main():
    """Main entry point demonstrating price monitoring."""
    print("=" * 60)
    print("Price Monitor Examples")
    print("=" * 60)
    
    # Example products to monitor (using placeholder URLs)
    products = [
        {"name": "Example Product 1", "url": "https://news.ycombinator.com"},
        {"name": "Example Product 2", "url": "https://example.com"},
    ]
    
    print("\n--- Monitoring Prices ---")
    alerts = await monitor_prices(products)
    
    print("\n--- Price History Summary ---")
    for product in products:
        await get_price_history_summary(product["name"])


if __name__ == "__main__":
    asyncio.run(main())
