"""
Example: Restaurant Booking Automation

Demonstrates autonomous booking workflow for restaurant reservations.
This matches the "Multi-Step Booking" example from the README Autonomous Mode section.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def book_restaurant_table(
    booking_site_url: str,
    booking_details: dict,
    max_time_seconds: int = 600
):
    """
    Book a table for 4 people at an Italian restaurant near downtown.
    
    Args:
        booking_site_url: URL of booking site (e.g., OpenTable)
        booking_details: Dictionary with booking information
        max_time_seconds: Maximum time allowed (10 minutes default)
        
    Returns:
        Booking confirmation result
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto(booking_site_url)
        
        # Use agent for complex multi-step booking
        result = await browser.agent(
            task="Book a table for 4 people at an Italian restaurant near downtown",
            context={
                "location": booking_details.get("location"),
                "date": booking_details.get("date"),
                "party_size": booking_details.get("party_size"),
                "cuisine": booking_details.get("cuisine"),
                "name": booking_details.get("name"),
                "phone": booking_details.get("phone"),
                "email": booking_details.get("email"),
                "special_requests": booking_details.get("special_requests", "")
            },
            max_time_seconds=max_time_seconds
        )
        
        if result.success:
            print(f"Reservation: {result.data}")
            print(f"Duration: {result.execution.duration_seconds:.1f}s")
            print(f"LLM Cost: ${result.llm_usage.cost_usd:.4f}")
        else:
            print(f"Booking failed: {result.error}")
        
        return result


async def search_and_book():
    """Search for restaurants and book the best match."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        log_verbosity="verbose",
    ) as browser:
        await browser.goto("https://opentable.com")
        
        # Multi-step autonomous workflow
        result = await browser.agent(
            task="""
            1. Search for Italian restaurants in San Francisco near downtown
            2. Filter by available tables for 4 people on Saturday at 7pm
            3. Find a highly-rated restaurant (4+ stars)
            4. Book a table
            5. Enter guest information
            6. Confirm the reservation
            """,
            context={
                "location": "San Francisco, CA",
                "cuisine": "Italian",
                "area": "downtown",
                "date": "Saturday",
                "time": "7pm",
                "party_size": 4,
                "min_rating": 4.0,
                "guest_name": "John Doe",
                "guest_phone": "555-987-6543",
                "guest_email": "john@example.com",
                "special_requests": "Window table if available"
            },
            max_iterations=50,
            max_time_seconds=600
        )
        
        if result.success:
            print("\n=== Booking Confirmation ===")
            print(f"Result: {result.data}")
            print(f"\nExecution Details:")
            print(f"  Steps completed: {result.execution.iterations}")
            print(f"  Total time: {result.execution.duration_seconds:.1f}s")
            print(f"  LLM tokens used: {result.llm_usage.total_tokens:,}")
            print(f"  Cost: ${result.llm_usage.cost_usd:.4f}")
        
        return result


async def book_with_requirements():
    """Book with specific requirements and preferences."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://resy.com")
        
        result = await browser.agent(
            task="Find and book a restaurant that meets all requirements",
            context={
                "location": "New York, NY",
                "neighborhood": "West Village",
                "cuisine": "French",
                "date": "Friday, January 31st",
                "time": "8:00 PM",
                "party_size": 2,
                "requirements": {
                    "outdoor_seating": True,
                    "accepts_reservations": True,
                    "price_range": "$$$",
                    "min_rating": 4.5
                },
                "guest_info": {
                    "name": "Alice Johnson",
                    "phone": "212-555-1234",
                    "email": "alice@example.com"
                },
                "special_occasion": "Anniversary",
                "dietary_restrictions": "Vegetarian options needed"
            },
            max_iterations=40,
            max_time_seconds=480
        )
        
        return result


async def modify_existing_reservation():
    """Modify an existing restaurant reservation."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://opentable.com")
        
        # Log in first
        await browser.act("Click on Sign In")
        await asyncio.sleep(1)
        
        # Store credentials securely
        email_id = browser.store_credential("user_email", "user@example.com", "email")
        pwd_id = browser.store_credential("user_pwd", "password123", "password")
        
        await browser.secure_fill("#email", email_id)
        await browser.secure_fill("#password", pwd_id)
        await browser.act("Click Sign In button")
        await asyncio.sleep(2)
        
        # Modify reservation
        result = await browser.agent(
            task="Find my reservation for Bella Italia on Saturday and change it to Sunday at 6pm instead of 7pm",
            context={
                "restaurant_name": "Bella Italia",
                "current_date": "Saturday",
                "current_time": "7pm",
                "new_date": "Sunday",
                "new_time": "6pm",
                "party_size": 4  # Keep same
            },
            max_iterations=30
        )
        
        if result.success:
            print(f"Reservation modified: {result.data}")
        
        return result


async def cancel_reservation():
    """Cancel a restaurant reservation."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://opentable.com/my-account/reservations")
        
        result = await browser.agent(
            task="Find and cancel the reservation for Trattoria Roma on Saturday at 8pm",
            context={
                "restaurant_name": "Trattoria Roma",
                "date": "Saturday",
                "time": "8pm",
                "confirmation_number": "ABC123456"
            },
            max_iterations=20
        )
        
        return result


async def main():
    """Main entry point for booking examples."""
    print("=" * 60)
    print("Restaurant Booking Automation Examples")
    print("=" * 60)
    
    # Example 1: Book a table (matching README example)
    print("\n--- Example 1: Book Italian Restaurant ---")
    result = await book_restaurant_table(
        booking_site_url="https://opentable.com",
        booking_details={
            "location": "San Francisco, CA",
            "date": "Saturday at 7pm",
            "party_size": 4,
            "cuisine": "Italian",
            "name": "John Doe",
            "phone": "555-987-6543",
            "email": "john@example.com"
        }
    )
    
    # Example 2: Search and book
    print("\n--- Example 2: Search and Book ---")
    # await search_and_book()
    
    # Print summary
    if result and result.success:
        print("\n=== Booking Summary ===")
        print(f"Success: {result.success}")
        print(f"Confirmation: {result.data}")
        print(f"Time: {result.execution.duration_seconds:.1f}s")
        print(f"Cost: ${result.llm_usage.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
