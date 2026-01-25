"""
Example: Job Application Form Automation

Demonstrates autonomous form filling and submission for job applications.
This matches the "Form Automation" example from the README Autonomous Mode section.

Prerequisites:
- pip install flybrowser
- export OPENAI_API_KEY="sk-..."
"""

import asyncio
import os
from flybrowser import FlyBrowser


async def fill_job_application(
    application_url: str,
    applicant_data: dict,
    max_iterations: int = 30,
    max_time_seconds: int = 300
):
    """
    Fill out and submit a job application form autonomously.
    
    Args:
        application_url: URL of the job application form
        applicant_data: Dictionary with applicant information
        max_iterations: Maximum agent iterations
        max_time_seconds: Maximum time allowed
        
    Returns:
        Result of the application submission
    """
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto(application_url)
        
        # Use agent for complex multi-step form filling
        result = await browser.agent(
            task="Fill out and submit the job application",
            context={
                "name": applicant_data.get("name"),
                "email": applicant_data.get("email"),
                "phone": applicant_data.get("phone"),
                "position": applicant_data.get("position"),
                "experience_years": applicant_data.get("experience_years"),
                "cover_letter": applicant_data.get("cover_letter"),
                "resume_uploaded": applicant_data.get("resume_path") is not None
            },
            max_iterations=max_iterations,
            max_time_seconds=max_time_seconds
        )
        
        if result.success:
            print(f"Application submitted! Confirmation: {result.data}")
            result.pprint()  # Pretty-print execution summary and LLM usage
            return result
        else:
            print(f"Failed: {result.error}")
            return result


async def simple_contact_form():
    """Fill a simple contact form (simpler example)."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        # Using example.com as placeholder
        await browser.goto("https://example.com/contact")
        
        # Autonomous form filling
        result = await browser.agent(
            task="Fill out and submit the contact form",
            context={
                "name": "John Smith",
                "email": "john@example.com",
                "subject": "Product Inquiry",
                "message": "I am interested in learning more about your enterprise solutions."
            },
            max_iterations=20
        )
        
        if result.success:
            print("Form submitted successfully!")
            print(f"Confirmation: {result.data}")
            print(f"Duration: {result.execution.duration_seconds:.1f}s")
            print(f"LLM Cost: ${result.llm_usage.cost_usd:.4f}")
        
        return result


async def multi_page_application():
    """Handle multi-page application form."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        log_verbosity="verbose",
    ) as browser:
        await browser.goto("https://jobs.example.com/apply/senior-engineer")
        
        # Agent will navigate through multiple pages automatically
        result = await browser.agent(
            task="""
            Complete the multi-page job application:
            1. Fill personal information
            2. Add work experience
            3. Upload resume
            4. Answer screening questions
            5. Submit application
            """,
            context={
                "name": "Jane Smith",
                "email": "jane@example.com",
                "phone": "555-123-4567",
                "position": "Senior Engineer",
                "experience_years": 5,
                "current_company": "Tech Corp",
                "linkedin": "linkedin.com/in/janesmith",
                "github": "github.com/janesmith",
                "cover_letter": "I am excited to apply for this position...",
                "willing_to_relocate": True,
                "expected_salary": "$150,000"
            },
            max_iterations=50,
            max_time_seconds=600
        )
        
        if result.success:
            print("\n=== Application Complete ===")
            print(f"Submission confirmed: {result.data}")
            print(f"\nExecution Summary:")
            print(f"  Total iterations: {result.execution.iterations}")
            print(f"  Duration: {result.execution.duration_seconds:.1f}s")
            print(f"  Actions taken: {len(result.execution.actions_taken)}")
            print(f"\nLLM Usage:")
            print(f"  Total tokens: {result.llm_usage.total_tokens:,}")
            print(f"  Cost: ${result.llm_usage.cost_usd:.4f}")
        
        return result


async def application_with_file_upload():
    """Handle application with resume upload."""
    async with FlyBrowser(
        llm_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
    ) as browser:
        await browser.goto("https://careers.example.com/apply")
        
        # Note: For file uploads, pre-upload the file to a location the browser can access
        resume_path = "/path/to/resume.pdf"
        
        # Manual file upload first (if needed)
        if os.path.exists(resume_path):
            await browser.act(f"Upload the file at {resume_path} to the resume field")
        
        # Then let agent handle the rest
        result = await browser.agent(
            task="Complete the job application form and submit",
            context={
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "555-987-6543",
                "position": "Software Engineer",
                "resume_uploaded": True,  # Signal that resume is already uploaded
                "years_experience": 3,
                "education": "BS Computer Science",
                "cover_letter": "I am passionate about building great software..."
            },
            max_iterations=30
        )
        
        return result


async def main():
    """Main entry point for job application examples."""
    print("=" * 60)
    print("Job Application Form Automation Examples")
    print("=" * 60)
    
    # Example 1: Simple contact form
    print("\n--- Example 1: Simple Contact Form ---")
    await simple_contact_form()
    
    # Example 2: Full job application (using example URLs)
    print("\n--- Example 2: Job Application Form ---")
    result = await fill_job_application(
        application_url="https://jobs.example.com/apply",
        applicant_data={
            "name": "Jane Smith",
            "email": "jane@example.com",
            "phone": "555-123-4567",
            "position": "Senior Engineer",
            "experience_years": 5,
            "cover_letter": "I am excited to apply for this position because..."
        }
    )
    
    # Print summary
    if result and result.success:
        print("\n=== Application Summary ===")
        print(f"Success: {result.success}")
        print(f"Data: {result.data}")
        print(f"Iterations: {result.execution.iterations}")
        print(f"Duration: {result.execution.duration_seconds:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
