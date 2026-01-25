"""
Example: E2E Checkout Testing

End-to-end test for e-commerce checkout flow.
Demonstrates complete user journey testing from product selection to order placement.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from dataclasses import dataclass
from flybrowser import FlyBrowser


@dataclass
class TestResult:
    """Result of a test step."""
    step: str
    passed: bool
    message: str
    screenshot_id: str | None = None


class CheckoutTest:
    """End-to-end checkout flow test."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results: list[TestResult] = []
        self.browser: FlyBrowser | None = None
    
    async def setup(self):
        """Initialize browser for testing."""
        self.browser = FlyBrowser(
            llm_provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            headless=True,
            log_verbosity="minimal",
        )
        await self.browser.__aenter__()
        print(f"Browser initialized for {self.base_url}")
    
    async def teardown(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.__aexit__(None, None, None)
            print("Browser closed")
    
    async def record_step(self, step: str, passed: bool, message: str):
        """Record a test step result."""
        screenshot = await self.browser.screenshot() if self.browser else None
        self.results.append(TestResult(
            step=step,
            passed=passed,
            message=message,
            screenshot_id=screenshot.get("screenshot_id") if screenshot else None
        ))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {step}: {message}")
    
    async def test_navigate_to_product(self, product_name: str) -> bool:
        """Test navigating to a specific product."""
        print("\n--- Step 1: Navigate to Product ---")
        
        await self.browser.goto(self.base_url)
        
        # Search for product
        result = await self.browser.act(f"Search for '{product_name}'")
        if not result.success:
            await self.record_step("Product Search", False, "Failed to search for product")
            return False
        
        # Click on product
        result = await self.browser.act(f"Click on the first product result matching '{product_name}'")
        if not result.success:
            await self.record_step("Product Selection", False, "Failed to select product")
            return False
        
        # Verify product page
        product_info = await self.browser.extract(
            "What is the product name and price shown on this page?"
        )
        
        if product_info.success and product_info.data:
            await self.record_step(
                "Product Navigation",
                True,
                f"Found product: {product_info.data}"
            )
            return True
        else:
            await self.record_step("Product Navigation", False, "Could not verify product page")
            return False
    
    async def test_add_to_cart(self) -> bool:
        """Test adding product to cart."""
        print("\n--- Step 2: Add to Cart ---")
        
        # Click add to cart
        result = await self.browser.act("Click the 'Add to Cart' or 'Add to Bag' button")
        
        if not result.success:
            await self.record_step("Add to Cart", False, "Failed to click add to cart button")
            return False
        
        await asyncio.sleep(2)  # Wait for cart update
        
        # Verify item was added
        cart_check = await self.browser.extract(
            "Is there a cart indicator showing items have been added? "
            "What does the cart count show?"
        )
        
        if cart_check.success:
            await self.record_step("Add to Cart", True, f"Cart updated: {cart_check.data}")
            return True
        else:
            await self.record_step("Add to Cart", False, "Could not verify cart update")
            return False
    
    async def test_view_cart(self) -> bool:
        """Test viewing the shopping cart."""
        print("\n--- Step 3: View Cart ---")
        
        # Navigate to cart
        result = await self.browser.act("Click on the shopping cart icon or 'View Cart' link")
        
        if not result.success:
            await self.record_step("View Cart", False, "Failed to navigate to cart")
            return False
        
        await asyncio.sleep(1)
        
        # Verify cart contents
        cart_contents = await self.browser.extract(
            "What items are in the shopping cart? "
            "Include item names, quantities, and prices."
        )
        
        if cart_contents.success and cart_contents.data:
            await self.record_step("View Cart", True, f"Cart contents: {cart_contents.data}")
            return True
        else:
            await self.record_step("View Cart", False, "Could not verify cart contents")
            return False
    
    async def test_proceed_to_checkout(self) -> bool:
        """Test proceeding to checkout."""
        print("\n--- Step 4: Proceed to Checkout ---")
        
        # Click checkout button
        result = await self.browser.act(
            "Click the 'Checkout', 'Proceed to Checkout', or 'Continue to Checkout' button"
        )
        
        if not result.success:
            await self.record_step("Proceed to Checkout", False, "Failed to click checkout")
            return False
        
        await asyncio.sleep(2)
        
        # Verify checkout page
        page_check = await self.browser.extract(
            "What checkout form fields are visible? "
            "Is this the shipping, billing, or payment page?"
        )
        
        if page_check.success:
            await self.record_step(
                "Proceed to Checkout",
                True,
                f"Checkout page loaded: {page_check.data}"
            )
            return True
        else:
            await self.record_step("Proceed to Checkout", False, "Could not verify checkout page")
            return False
    
    async def test_fill_shipping_info(self, shipping_info: dict) -> bool:
        """Test filling shipping information."""
        print("\n--- Step 5: Fill Shipping Info ---")
        
        # Fill shipping form using agent for complex form handling
        result = await self.browser.agent(
            task="Fill out the shipping address form with the provided information",
            context=f"""
            Use these shipping details:
            - First Name: {shipping_info.get('first_name', 'Test')}
            - Last Name: {shipping_info.get('last_name', 'User')}
            - Address: {shipping_info.get('address', '123 Test Street')}
            - City: {shipping_info.get('city', 'Test City')}
            - State/Province: {shipping_info.get('state', 'CA')}
            - ZIP/Postal Code: {shipping_info.get('zip', '12345')}
            - Country: {shipping_info.get('country', 'United States')}
            - Phone: {shipping_info.get('phone', '555-123-4567')}
            - Email: {shipping_info.get('email', 'test@example.com')}
            
            Fill in all available fields. Some fields may not be present on all sites.
            """
        )
        
        if result.success:
            await self.record_step("Fill Shipping Info", True, "Shipping info filled")
            return True
        else:
            await self.record_step("Fill Shipping Info", False, f"Failed: {result.message}")
            return False
    
    async def test_continue_to_payment(self) -> bool:
        """Test continuing to payment section."""
        print("\n--- Step 6: Continue to Payment ---")
        
        # Click continue button
        result = await self.browser.act(
            "Click 'Continue', 'Continue to Payment', or 'Next' button to proceed"
        )
        
        if not result.success:
            await self.record_step("Continue to Payment", False, "Failed to continue")
            return False
        
        await asyncio.sleep(2)
        
        # Check for payment form
        payment_check = await self.browser.extract(
            "Is there a payment form visible? What payment methods are available?"
        )
        
        if payment_check.success:
            await self.record_step(
                "Continue to Payment",
                True,
                f"Payment section: {payment_check.data}"
            )
            return True
        else:
            await self.record_step("Continue to Payment", False, "Could not verify payment page")
            return False
    
    async def test_order_summary(self) -> bool:
        """Test order summary verification."""
        print("\n--- Step 7: Verify Order Summary ---")
        
        # Extract order summary
        summary = await self.browser.extract(
            "What is the order summary? Include subtotal, shipping cost, "
            "taxes, and total amount."
        )
        
        if summary.success and summary.data:
            # Verify totals make sense
            await self.record_step(
                "Order Summary",
                True,
                f"Order summary: {summary.data}"
            )
            return True
        else:
            await self.record_step("Order Summary", False, "Could not extract order summary")
            return False
    
    def print_results(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("E2E CHECKOUT TEST RESULTS")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.step}")
            print(f"       {result.message}")
        
        print("-" * 60)
        print(f"Total: {passed}/{total} steps passed")
        
        if passed == total:
            print("STATUS: ALL TESTS PASSED")
        else:
            print(f"STATUS: {total - passed} TESTS FAILED")
        
        print("=" * 60)


async def run_checkout_test(
    store_url: str,
    product_name: str,
    shipping_info: dict | None = None
):
    """
    Run complete checkout flow test.
    
    Args:
        store_url: Base URL of the e-commerce store
        product_name: Product to search for and add to cart
        shipping_info: Shipping details dictionary
    """
    # Default shipping info for testing
    if shipping_info is None:
        shipping_info = {
            "first_name": "Test",
            "last_name": "Automation",
            "address": "123 Test Street",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94102",
            "country": "United States",
            "phone": "555-123-4567",
            "email": "test@example.com"
        }
    
    test = CheckoutTest(store_url)
    
    try:
        await test.setup()
        
        # Run test steps in sequence
        if await test.test_navigate_to_product(product_name):
            if await test.test_add_to_cart():
                if await test.test_view_cart():
                    if await test.test_proceed_to_checkout():
                        if await test.test_fill_shipping_info(shipping_info):
                            await test.test_continue_to_payment()
                            await test.test_order_summary()
        
    finally:
        await test.teardown()
    
    test.print_results()
    return test.results


async def test_guest_checkout(store_url: str, product_name: str):
    """
    Test guest checkout flow without account creation.
    
    Args:
        store_url: Base URL of the e-commerce store
        product_name: Product to add to cart
    """
    print("=" * 60)
    print("Testing Guest Checkout Flow")
    print("=" * 60)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(store_url)
        
        # Add product to cart
        await browser.act(f"Search for {product_name} and add the first result to cart")
        await browser.act("Go to the shopping cart")
        await browser.act("Click checkout or proceed to checkout")
        
        # Look for guest checkout option
        guest_option = await browser.observe(
            "Find any 'Guest Checkout', 'Continue as Guest', "
            "or 'Checkout without account' option"
        )
        
        if guest_option.success and guest_option.elements:
            print(f"Found guest checkout option: {guest_option.elements[0]}")
            await browser.act("Click on the guest checkout option")
            print("PASS: Guest checkout flow available")
        else:
            print("INFO: No explicit guest checkout option found (may proceed directly)")


async def test_cart_persistence(store_url: str, product_name: str):
    """
    Test that cart persists after page refresh.
    
    Args:
        store_url: Base URL of the e-commerce store
        product_name: Product to add to cart
    """
    print("=" * 60)
    print("Testing Cart Persistence")
    print("=" * 60)
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(store_url)
        
        # Add item to cart
        await browser.act(f"Search for {product_name}")
        await browser.act("Add the first product to cart")
        
        # Get cart count before refresh
        before = await browser.extract("What is the cart item count?")
        print(f"Cart count before refresh: {before.data}")
        
        # Refresh page
        await browser.goto(store_url)
        await asyncio.sleep(2)
        
        # Get cart count after refresh
        after = await browser.extract("What is the cart item count?")
        print(f"Cart count after refresh: {after.data}")
        
        if before.data == after.data:
            print("PASS: Cart persisted after refresh")
        else:
            print("FAIL: Cart did not persist after refresh")


async def main():
    """Main entry point for checkout tests."""
    print("E2E Checkout Testing Examples")
    print("Note: These tests use demo/sandbox e-commerce sites")
    print()
    
    # Example with a demo store (using a common demo site)
    # In real usage, replace with your actual test environment
    demo_store = "https://www.saucedemo.com"
    
    # Run basic checkout test
    await run_checkout_test(
        store_url=demo_store,
        product_name="Sauce Labs Backpack",
        shipping_info={
            "first_name": "Test",
            "last_name": "User",
            "zip": "94102"
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
