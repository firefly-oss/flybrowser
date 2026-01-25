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
Deep codebase analysis script.

Identifies:
1. Dead/unused functions and classes
2. Unused imports
3. TODO/FIXME comments
4. Deprecated patterns
5. Potential bugs (unreachable code, always-true conditions, etc.)
6. Code complexity metrics

Usage:
    python scripts/deep_analysis.py [--output report.json]
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code for various issues."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.imports: Set[str] = set()
        self.used_names: Set[str] = set()
        self.defined_functions: Dict[str, int] = {}
        self.defined_classes: Dict[str, int] = {}
        self.todos: List[Tuple[int, str]] = []
        self.issues: List[Dict[str, Any]] = []
        self.complexity: int = 1
        
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split('.')[0])
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined_functions[node.name] = node.lineno
        # Check for empty function bodies (stub functions)
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.issues.append({
                'type': 'empty_function',
                'line': node.lineno,
                'name': node.name,
                'severity': 'info'
            })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined_classes[node.name] = node.lineno
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        # Check for always-true conditions
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.issues.append({
                'type': 'always_true_condition',
                'line': node.lineno,
                'severity': 'warning'
            })
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try) -> None:
        self.complexity += 1
        # Check for bare except
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append({
                    'type': 'bare_except',
                    'line': handler.lineno,
                    'severity': 'warning'
                })
        self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content, filename=str(filepath))
        analyzer = CodeAnalyzer(str(filepath))
        analyzer.visit(tree)
        
        # Find TODO/FIXME comments
        todos = []
        for i, line in enumerate(content.split('\n'), 1):
            if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                todos.append({'line': i, 'text': line.strip()})
        
        # Check for unused imports (simple heuristic)
        potentially_unused_imports = []
        for imp in analyzer.imports:
            if imp not in analyzer.used_names and imp not in ['__future__', 'typing']:
                potentially_unused_imports.append(imp)
        
        # Check for unused functions (not called in same file)
        potentially_unused_functions = []
        for func_name, line in analyzer.defined_functions.items():
            if func_name not in analyzer.used_names and not func_name.startswith('_'):
                potentially_unused_functions.append({'name': func_name, 'line': line})
        
        return {
            'file': str(filepath),
            'complexity': analyzer.complexity,
            'functions': len(analyzer.defined_functions),
            'classes': len(analyzer.defined_classes),
            'imports': len(analyzer.imports),
            'todos': todos,
            'potentially_unused_imports': potentially_unused_imports,
            'potentially_unused_functions': potentially_unused_functions,
            'issues': analyzer.issues,
        }
    except SyntaxError as e:
        return {
            'file': str(filepath),
            'error': f'Syntax error: {e}',
        }
    except Exception as e:
        return {
            'file': str(filepath),
            'error': f'Analysis error: {e}',
        }


def find_deprecated_patterns(filepath: Path) -> List[Dict[str, Any]]:
    """Find deprecated patterns in code."""
    deprecated = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for deprecated patterns
        patterns = {
            r'print\s*\(': {'msg': 'Use logging instead of print', 'severity': 'info'},
            r'time\.sleep': {'msg': 'Consider using asyncio.sleep', 'severity': 'info'},
            r'import\s+urllib\b': {'msg': 'Consider using httpx or requests', 'severity': 'info'},
        }
        
        for pattern, info in patterns.items():
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                deprecated.append({
                    'line': line_num,
                    'pattern': pattern,
                    'message': info['msg'],
                    'severity': info['severity']
                })
    except Exception:
        pass
    
    return deprecated


def generate_report(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive analysis report."""
    total_files = len(analyses)
    total_complexity = sum(a.get('complexity', 0) for a in analyses)
    total_todos = sum(len(a.get('todos', [])) for a in analyses)
    
    files_with_issues = [a for a in analyses if a.get('issues') or a.get('todos') or a.get('potentially_unused_imports')]
    
    high_complexity_files = [
        {'file': a['file'], 'complexity': a.get('complexity', 0)}
        for a in analyses
        if a.get('complexity', 0) > 50
    ]
    
    return {
        'summary': {
            'total_files': total_files,
            'total_complexity': total_complexity,
            'avg_complexity': total_complexity / max(total_files, 1),
            'total_todos': total_todos,
            'files_with_issues': len(files_with_issues),
            'high_complexity_files': len(high_complexity_files),
        },
        'high_complexity_files': sorted(high_complexity_files, key=lambda x: x['complexity'], reverse=True),
        'files_with_issues': files_with_issues[:20],  # Top 20
        'all_analyses': analyses,
    }


def main():
    parser = argparse.ArgumentParser(description='Deep codebase analysis')
    parser.add_argument('--output', default='analysis_report.json', help='Output JSON file')
    parser.add_argument('--text-report', default='analysis_report.txt', help='Text report file')
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    flybrowser_dir = project_root / 'flybrowser'
    
    if not flybrowser_dir.exists():
        print(f"Error: Cannot find flybrowser directory at {flybrowser_dir}")
        sys.exit(1)
    
    print(f"Analyzing {flybrowser_dir}...")
    
    # Find all Python files
    python_files = []
    for path in flybrowser_dir.rglob('*.py'):
        if '__pycache__' not in str(path):
            python_files.append(path)
    
    print(f"Found {len(python_files)} Python files")
    
    # Analyze all files
    print("\nAnalyzing files...")
    analyses = []
    for filepath in python_files:
        analysis = analyze_file(filepath)
        
        # Add deprecated patterns
        deprecated = find_deprecated_patterns(filepath)
        if deprecated:
            analysis['deprecated_patterns'] = deprecated
        
        analyses.append(analysis)
        
        # Print progress
        if len(analyses) % 10 == 0:
            print(f"  Analyzed {len(analyses)}/{len(python_files)} files...")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(analyses)
    
    # Save JSON report
    json_path = project_root / args.output
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved to: {json_path}")
    
    # Generate text report
    text_report = generate_text_report(report)
    text_path = project_root / args.text_report
    with open(text_path, 'w') as f:
        f.write(text_report)
    print(f"Text report saved to: {text_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total files analyzed: {report['summary']['total_files']}")
    print(f"Average complexity: {report['summary']['avg_complexity']:.1f}")
    print(f"Total TODOs/FIXMEs: {report['summary']['total_todos']}")
    print(f"Files with issues: {report['summary']['files_with_issues']}")
    print(f"High complexity files (>50): {report['summary']['high_complexity_files']}")
    
    if report['high_complexity_files']:
        print("\nMost complex files:")
        for f in report['high_complexity_files'][:5]:
            print(f"  {Path(f['file']).name}: {f['complexity']}")
    
    print("\n Analysis complete!")
    return 0


def generate_text_report(report: Dict[str, Any]) -> str:
    """Generate human-readable text report."""
    lines = [
        "="*80,
        "FLYBROWSER DEEP CODE ANALYSIS REPORT",
        "="*80,
        "",
        "SUMMARY:",
        "-"*80,
        f"Total files analyzed: {report['summary']['total_files']}",
        f"Average complexity: {report['summary']['avg_complexity']:.1f}",
        f"Total TODOs/FIXMEs: {report['summary']['total_todos']}",
        f"Files with potential issues: {report['summary']['files_with_issues']}",
        f"High complexity files (>50): {report['summary']['high_complexity_files']}",
        "",
    ]
    
    if report['high_complexity_files']:
        lines.extend([
            "HIGH COMPLEXITY FILES:",
            "-"*80,
        ])
        for f in report['high_complexity_files'][:10]:
            lines.append(f"  {Path(f['file']).name}: complexity={f['complexity']}")
        lines.append("")
    
    # Files with issues
    if report['files_with_issues']:
        lines.extend([
            "FILES WITH POTENTIAL ISSUES:",
            "-"*80,
        ])
        for analysis in report['files_with_issues'][:20]:
            filepath = Path(analysis['file']).name
            issues = []
            if analysis.get('todos'):
                issues.append(f"{len(analysis['todos'])} TODOs")
            if analysis.get('potentially_unused_imports'):
                issues.append(f"{len(analysis['potentially_unused_imports'])} unused imports")
            if analysis.get('issues'):
                issues.append(f"{len(analysis['issues'])} code issues")
            
            if issues:
                lines.append(f"\n{filepath}:")
                lines.append(f"  Issues: {', '.join(issues)}")
    
    lines.append("\n" + "="*80)
    return "\n".join(lines)


if __name__ == '__main__':
    sys.exit(main())
