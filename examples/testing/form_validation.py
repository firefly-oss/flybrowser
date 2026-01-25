"""
Example: Form Validation Testing

Tests input validation on web forms.
Demonstrates testing validation errors, field requirements, and form submission.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def test_form_validation(form_url: str):
    """
    Test that form validation works correctly.
    
    Args:
        form_url: URL of the form to test
        
    Returns:
        List of test results
    """
    results = []
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(form_url)
        
        # Test 1: Empty form submission
        print("Test 1: Empty form submission")
        await browser.act("Click the Submit button without filling any fields")
        
        # Check for validation errors
        errors = await browser.extract("What error messages are shown on the form?")
        
        if errors.success and errors.data:
            print(f"  PASS: Validation errors shown: {errors.data}")
            results.append(("Empty form validation", True, errors.data))
        else:
            print("  FAIL: No validation errors shown")
            results.append(("Empty form validation", False, None))
        
        # Test 2: Invalid email format
        print("\nTest 2: Invalid email format")
        await browser.goto(form_url)  # Reset form
        await browser.act("Type 'invalid-email' in the email field")
        await browser.act("Click outside the email field to trigger validation")
        
        email_error = await browser.extract(
            "Is there an error message about the email format?"
        )
        
        has_email_error = email_error.success and "invalid" in str(email_error.data).lower()
        if has_email_error:
            print("  PASS: Email format error shown")
            results.append(("Invalid email validation", True, email_error.data))
        else:
            print("  FAIL: No email format error")
            results.append(("Invalid email validation", False, email_error.data))
        
        # Test 3: Password requirements
        print("\nTest 3: Password requirements")
        await browser.goto(form_url)  # Reset form
        await browser.act("Type 'test@example.com' in the email field")
        await browser.act("Type '123' in the password field")
        await browser.act("Click outside the password field")
        
        pwd_error = await browser.extract(
            "What does the password requirement error say?"
        )
        
        if pwd_error.success and pwd_error.data:
            print(f"  PASS: Password error shown: {pwd_error.data}")
            results.append(("Password requirements", True, pwd_error.data))
        else:
            print("  FAIL: No password requirement error")
            results.append(("Password requirements", False, None))
        
        # Test 4: Valid form data
        print("\nTest 4: Valid form submission")
        await browser.goto(form_url)  # Reset form
        await browser.act("Type 'test@example.com' in the email field")
        await browser.act("Type 'SecurePassword123!' in the password field")
        await browser.act("Type 'SecurePassword123!' in the confirm password field")
        await browser.act("Check any required checkboxes like terms agreement")
        
        # Verify no errors are shown
        final_errors = await browser.extract(
            "Are there any validation errors currently visible on the form?"
        )
        
        no_errors = not final_errors.data or "no" in str(final_errors.data).lower()
        if no_errors:
            print("  PASS: Form validates successfully")
            results.append(("Valid form passes", True, None))
        else:
            print(f"  FAIL: Unexpected errors: {final_errors.data}")
            results.append(("Valid form passes", False, final_errors.data))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    passed = sum(1 for _, p, _ in results if p)
    print(f"Passed: {passed}/{len(results)}")
    for name, passed, details in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    
    return results


async def test_form_submission(form_url: str, form_data: dict):
    """
    Test complete form submission.
    
    Args:
        form_url: URL of the form
        form_data: Dictionary of field names to values
        
    Returns:
        Submission result
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        await browser.goto(form_url)
        
        # Fill out the form
        print("Filling form fields...")
        for field_name, value in form_data.items():
            await browser.act(f"Type '{value}' in the {field_name} field")
        
        # Take screenshot before submission
        before_screenshot = await browser.screenshot()
        print("  Screenshot captured (before)")
        
        # Submit the form
        print("Submitting form...")
        await browser.act("Click the Submit button")
        
        # Wait for response
        await asyncio.sleep(2)
        
        # Verify submission
        result = await browser.extract(
            "Is there a success message? What does it say?"
        )
        
        # Take screenshot after submission
        after_screenshot = await browser.screenshot()
        print("  Screenshot captured (after)")
        
        success = result.success and any(
            word in str(result.data).lower() 
            for word in ["success", "thank", "submitted", "received"]
        )
        
        print(f"\nSubmission result: {'SUCCESS' if success else 'FAILED'}")
        print(f"  Message: {result.data}")
        
        return {
            "success": success,
            "message": result.data,
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot
        }


async def test_required_fields(form_url: str, field_names: list[str]):
    """
    Test that required fields show validation errors.
    
    Args:
        form_url: URL of the form
        field_names: List of field names expected to be required
        
    Returns:
        Dictionary of field validation results
    """
    results = {}
    
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        headless=True,
    ) as browser:
        for field in field_names:
            await browser.goto(form_url)
            
            print(f"Testing required field: {field}")
            
            # Fill all fields EXCEPT the one we're testing
            for other_field in field_names:
                if other_field != field:
                    await browser.act(f"Type 'test value' in the {other_field} field")
            
            # Try to submit
            await browser.act("Click the Submit button")
            await asyncio.sleep(1)
            
            # Check for error on the empty field
            error = await browser.extract(
                f"Is there an error message for the {field} field?"
            )
            
            has_error = error.success and error.data and \
                       ("required" in str(error.data).lower() or 
                        "error" in str(error.data).lower())
            
            results[field] = {
                "required": has_error,
                "error_message": error.data
            }
            
            status = "REQUIRED" if has_error else "OPTIONAL"
            print(f"  [{status}] {field}: {error.data}")
    
    return results


async def main():
    """Main entry point for form validation tests."""
    print("=" * 60)
    print("Form Validation Testing Examples")
    print("=" * 60)
    
    # Example: Test a contact form (using placeholder URL)
    print("\n--- Testing Form Validation ---")
    await test_form_validation("https://example.com/contact")
    
    # Example: Test form submission
    print("\n--- Testing Form Submission ---")
    await test_form_submission(
        "https://example.com/contact",
        {
            "name": "John Doe",
            "email": "john@example.com",
            "message": "This is a test message."
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
