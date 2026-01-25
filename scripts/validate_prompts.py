#!/usr/bin/env python3
"""
Prompt Template Validation Script

Validates all prompt templates in the system:
- Templates load correctly
- Required fields are present
- Templates render without errors
- No duplicate template names
- Metadata is complete

Usage:
    python scripts/validate_prompts.py
    python scripts/validate_prompts.py --verbose
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flybrowser.prompts import PromptManager
from flybrowser.prompts.registry import PromptRegistry
import argparse


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}[ok]{Colors.END} {msg}")


def print_error(msg):
    print(f"{Colors.RED}[fail]{Colors.END} {msg}")


def print_warning(msg):
    print(f"{Colors.YELLOW}[warning]{Colors.END} {msg}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.END}")
    print("=" * len(msg))


def validate_template_loading(verbose=False):
    """Test that all templates load correctly"""
    print_header("1. Template Loading")
    
    try:
        pm = PromptManager()
        template_count = len(pm.registry.templates)
        print_success(f"Loaded {template_count} templates")
        
        if verbose:
            for key in sorted(pm.registry.templates.keys()):
                print(f"   - {key}")
        
        return True, template_count
    except Exception as e:
        print_error(f"Failed to load templates: {e}")
        return False, 0


def validate_template_structure(pm, verbose=False):
    """Validate template structure and required fields"""
    print_header("2. Template Structure")
    
    required_fields = ['name', 'version', 'system_template', 'user_template']
    errors = []
    warnings = []
    
    for key, template in pm.registry.templates.items():
        # Check required fields
        for field in required_fields:
            if not hasattr(template, field) or not getattr(template, field):
                errors.append(f"{key}: Missing required field '{field}'")
        
        # Check metadata
        if not hasattr(template, 'metadata') or not template.metadata:
            warnings.append(f"{key}: Missing metadata")
        
        # Check version format
        version = getattr(template, 'version', '')
        if not version or not any(c.isdigit() for c in version):
            warnings.append(f"{key}: Invalid version format '{version}'")
    
    if errors:
        for error in errors:
            print_error(error)
        return False
    
    if warnings:
        for warning in warnings:
            print_warning(warning)
    
    if not errors:
        print_success(f"All templates have required fields")
    
    return True


def validate_template_rendering(pm, verbose=False):
    """Test that templates can render with sample data"""
    print_header("3. Template Rendering")
    
    # Sample data for different template types
    test_cases = {
        'obstacle_detection': {
            'url': 'https://example.com',
            'title': 'Test Page',
            'has_modals': True,
            'modal_count': 1,
            'overlay_count': 2,
            'overlays': '[]',
            'focused_element': 'null',
        },
        'element_detection': {
            'description': 'login button',
            'url': 'https://example.com',
            'title': 'Login Page',
            'html_snippet': '<button id="login">Login</button>',
            'screenshot_available': False,
        },
        'action_planning': {
            'instruction': 'Click the submit button',
            'url': 'https://example.com',
            'title': 'Form Page',
            'visible_elements': 'button#submit\ninput[name="email"]',
        },
        'data_extraction': {
            'query': 'Extract product name',
            'url': 'https://example.com',
            'title': 'Product Page',
            'rendered_content': '<h1>Product Name</h1>',
        },
    }
    
    errors = []
    tested = 0
    
    for template_name, variables in test_cases.items():
        try:
            prompts = pm.get_prompt(template_name, **variables)
            
            # Verify output structure
            if 'system' not in prompts or 'user' not in prompts:
                errors.append(f"{template_name}: Missing 'system' or 'user' in output")
            elif not prompts['system'] or not prompts['user']:
                errors.append(f"{template_name}: Empty 'system' or 'user' prompt")
            else:
                tested += 1
                if verbose:
                    print_success(f"{template_name} renders correctly")
        
        except Exception as e:
            errors.append(f"{template_name}: Rendering failed - {str(e)[:60]}")
    
    if errors:
        for error in errors:
            print_error(error)
        return False
    
    print_success(f"Tested {tested} templates, all render correctly")
    return True


def validate_no_duplicates(pm, verbose=False):
    """Check for duplicate template names (different versions OK)"""
    print_header("4. Duplicate Detection")
    
    names = {}
    for key in pm.registry.templates.keys():
        name = key.split(':')[0]
        names[name] = names.get(name, 0) + 1
    
    duplicates = {name: count for name, count in names.items() if count > 1}
    
    if duplicates:
        print_info(f"Found {len(duplicates)} templates with multiple versions:")
        if verbose:
            for name, count in duplicates.items():
                print(f"   - {name}: {count} versions")
    else:
        print_info("No duplicate template names found")
    
    return True


def validate_directory_organization(verbose=False):
    """Check that templates are organized in subdirectories"""
    print_header("5. Directory Organization")
    
    import os
    base_path = Path(__file__).parent.parent / "flybrowser" / "prompts" / "templates"
    
    expected_dirs = ['agents', 'tools', 'search', 'orchestration', 'autonomous']
    missing = []
    
    for dirname in expected_dirs:
        dir_path = base_path / dirname
        if not dir_path.exists():
            missing.append(dirname)
        else:
            template_count = len([f for f in os.listdir(dir_path) if f.endswith('.yaml')])
            if verbose:
                print_success(f"{dirname}/: {template_count} templates")
    
    if missing:
        print_warning(f"Missing expected directories: {', '.join(missing)}")
        return False
    
    if not verbose:
        print_success(f"All expected directories present")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate FlyBrowser prompt templates')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}FlyBrowser Prompt Template Validation{Colors.END}")
    print("=" * 60)
    
    # Run validations
    success = True
    
    loading_ok, template_count = validate_template_loading(args.verbose)
    success = success and loading_ok
    
    if loading_ok:
        pm = PromptManager()
        success = success and validate_template_structure(pm, args.verbose)
        success = success and validate_template_rendering(pm, args.verbose)
        success = success and validate_no_duplicates(pm, args.verbose)
    
    success = success and validate_directory_organization(args.verbose)
    
    # Summary
    print_header("Summary")
    if success:
        print_success(f"All validations passed! ({template_count} templates)")
        return 0
    else:
        print_error("Some validations failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
