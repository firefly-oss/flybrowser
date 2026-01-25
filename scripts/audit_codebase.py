#!/usr/bin/env python3
# Copyright 2026 Firefly Software Solutions Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive codebase audit script.

This script:
1. Adds Apache 2.0 copyright headers to all Python files
2. Checks for module docstrings
3. Identifies files needing better documentation
4. Generates a detailed audit report

Usage:
    python scripts/audit_codebase.py [--fix] [--report-only]
    
Options:
    --fix: Automatically add copyright headers
    --report-only: Only generate report, don't modify files
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

COPYRIGHT_HEADER = '''# Copyright 2026 Firefly Software Solutions Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''

def has_copyright(content: str) -> bool:
    """Check if file already has copyright header."""
    return "Copyright 2026 Firefly Software Solutions Inc" in content[:500]

def has_license_header(content: str) -> bool:
    """Check if file has Apache 2.0 license header."""
    return "Apache License" in content[:500]

def has_module_docstring(content: str) -> bool:
    """Check if file has a module-level docstring."""
    lines = content.strip().split('\n')
    
    # Skip shebang and copyright
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        start_idx = i
        break
    
    # Check for docstring after imports or at start
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        line = lines[i].strip()
        if line.startswith('"""') or line.startswith("'''"):
            return True
        if line and not line.startswith('from ') and not line.startswith('import '):
            # Hit actual code without docstring
            return False
    
    return False

def add_copyright_header(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_copyright(content):
            return False  # Already has copyright
        
        # Handle shebang if present
        if content.startswith('#!'):
            lines = content.split('\n', 1)
            new_content = lines[0] + '\n' + COPYRIGHT_HEADER + lines[1]
        else:
            new_content = COPYRIGHT_HEADER + content
        
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def count_functions_and_classes(content: str) -> Tuple[int, int]:
    """Count functions and classes in a file."""
    functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
    classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
    return functions, classes

def analyze_file(file_path: Path) -> dict:
    """Analyze a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.count('\n') + 1
        funcs, classes = count_functions_and_classes(content)
        
        return {
            'path': str(file_path),
            'lines': lines,
            'has_copyright': has_copyright(content),
            'has_license': has_license_header(content),
            'has_docstring': has_module_docstring(content),
            'functions': funcs,
            'classes': classes,
            'needs_attention': not has_copyright(content) or not has_module_docstring(content),
        }
    except Exception as e:
        return {
            'path': str(file_path),
            'error': str(e),
            'needs_attention': True,
        }

def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', 'build', 'dist', '.egg-info'}
    
    for path in root_dir.rglob('*.py'):
        # Skip excluded directories
        if any(ex in path.parts for ex in exclude_dirs):
            continue
        python_files.append(path)
    
    return sorted(python_files)

def generate_report(analyses: List[dict]) -> str:
    """Generate audit report."""
    total = len(analyses)
    with_copyright = sum(1 for a in analyses if a.get('has_copyright', False))
    with_docstring = sum(1 for a in analyses if a.get('has_docstring', False))
    needs_attention = [a for a in analyses if a.get('needs_attention', False)]
    
    total_lines = sum(a.get('lines', 0) for a in analyses)
    total_functions = sum(a.get('functions', 0) for a in analyses)
    total_classes = sum(a.get('classes', 0) for a in analyses)
    
    report = f"""
{'='*80}
FLYBROWSER CODEBASE AUDIT REPORT
{'='*80}

SUMMARY:
--------
Total Python files: {total}
Total lines of code: {total_lines:,}
Total functions: {total_functions}
Total classes: {total_classes}

COPYRIGHT HEADERS:
------------------
Files with copyright: {with_copyright}/{total} ({with_copyright/total*100:.1f}%)
Files without: {total - with_copyright}

MODULE DOCSTRINGS:
------------------
Files with docstrings: {with_docstring}/{total} ({with_docstring/total*100:.1f}%)
Files without: {total - with_docstring}

FILES NEEDING ATTENTION: {len(needs_attention)}
{'='*80}
"""
    
    if needs_attention:
        report += "\nFILES REQUIRING UPDATES:\n" + "-"*80 + "\n"
        for analysis in needs_attention:
            path = analysis['path']
            issues = []
            if not analysis.get('has_copyright'):
                issues.append("Missing copyright")
            if not analysis.get('has_docstring'):
                issues.append("Missing docstring")
            if 'error' in analysis:
                issues.append(f"Error: {analysis['error']}")
            
            report += f"\n{path}\n"
            report += f"  Issues: {', '.join(issues)}\n"
            if 'lines' in analysis:
                report += f"  Size: {analysis['lines']} lines, {analysis['functions']} functions, {analysis['classes']} classes\n"
    
    report += "\n" + "="*80 + "\n"
    return report

def main():
    parser = argparse.ArgumentParser(description='Audit FlyBrowser codebase')
    parser.add_argument('--fix', action='store_true', help='Add copyright headers to files')
    parser.add_argument('--report-only', action='store_true', help='Only generate report')
    parser.add_argument('--output', default='audit_report.txt', help='Report output file')
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    flybrowser_dir = project_root / 'flybrowser'
    
    if not flybrowser_dir.exists():
        print(f"Error: Cannot find flybrowser directory at {flybrowser_dir}")
        sys.exit(1)
    
    print(f"Scanning {flybrowser_dir}...")
    python_files = find_python_files(flybrowser_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Analyze all files
    print("\nAnalyzing files...")
    analyses = []
    for file_path in python_files:
        analysis = analyze_file(file_path)
        analyses.append(analysis)
    
    # Add copyright headers if requested
    if args.fix and not args.report_only:
        print("\nAdding copyright headers...")
        added = 0
        for file_path in python_files:
            if add_copyright_header(file_path):
                added += 1
                print(f"  Added to: {file_path.relative_to(project_root)}")
        print(f"\nAdded copyright headers to {added} files")
    
    # Generate report
    report = generate_report(analyses)
    
    # Save report
    report_path = project_root / args.output
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + report)
    
    # Return exit code based on issues found
    needs_attention = sum(1 for a in analyses if a.get('needs_attention', False))
    if needs_attention > 0 and not args.fix:
        print(f"\n  {needs_attention} files need attention. Run with --fix to add copyright headers.")
        return 1
    
    print("\n Audit complete!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
