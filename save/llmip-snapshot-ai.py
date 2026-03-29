#!/usr/bin/env python3
"""
LLMIP Snapshot AI Commit Message Generator
Analyzes git changes and generates descriptive commit messages.
"""

import subprocess
import sys
import os
from datetime import datetime

# ERCOT-specific file categories
CATEGORIES = {
    'models': ['.pt', '.pth', '.pkl', '.h5', '.joblib', '.onnx', 'models/', 'rf_spread_model', 'rf_spread_model.pkl'],
    'notebooks': ['.ipynb', 'notebooks/'],
    'data': ['.csv', '.parquet', '.feather', '.geojson', '.graphml', '.kml',
              'data/raw/', 'data/processed/', 'data/final/', 'data/external/',
              'DAM_prices_DATA.csv', 'DAM_constraint_DATA.csv', 'DAM_loadFRCST_DATA.csv',
              'DAM_wind_DATA.csv', 'Outrage_DATA.csv'],
    'reports': ['.html', '.pdf', '.png', '.jpg', '.svg',
              'reports/', 'reports/figures/', 'reports/google_earth/', 'reports/qgis/'],
    'scripts': ['.py', '.sh', 'scripts/', 'Makefile', 'pyproject.toml'],
    'docs': ['.md', '.txt', 'docs/', 'mkdocs.yml'],
    'references': ['references/', 'STAR_PTP', 'ERCOT DAM LMP_P2P', '.pdf'],
    'config': ['.yaml', '.json', '.toml', '.env', '.yml']
}

# Action verbs based on file category
ACTION_VERBS = {
    'models': ['Added', 'Updated', 'Refactored', 'Optimized', 'Fixed', 'Trained'],
    'notebooks': ['Updated', 'Completed', 'Fixed', 'Added analysis to'],
    'data': ['Processed', 'Added', 'Updated', 'Cleaned', 'Merged'],
    'reports': ['Generated', 'Updated', 'Added'],
    'scripts': ['Added', 'Refactored', 'Fixed', 'Updated', 'Created'],
    'docs': ['Updated', 'Added', 'Fixed', 'Restructured', 'Clarified'],
    'references': ['Added', 'Updated'],
    'config': ['Updated', 'Added', 'Fixed', 'Configured']
}

def get_changed_files():
    """Get list of changed files in git staging area"""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except subprocess.CalledProcessError as e:
        print(f"Error getting git changes: {e}", file=sys.stderr)
        return []

def categorize_file(filepath):
    """Categorize a file based on its path and extension"""
    filepath = filepath.lower()

    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            if pattern.endswith('/'):
                if filepath.startswith(pattern.rstrip('/')):
                    return category
            elif pattern in filepath:
                return category
    return 'other'

def count_files_by_category(files):
    """Count files by category"""
    counts = {cat: 0 for cat in CATEGORIES.keys()}
    for f in files:
        cat = categorize_file(f)
        counts[cat] = counts.get(cat, 0) + 1
    return counts

def select_action_verb(category, count):
    """Select appropriate action verb based on category"""
    if count > 1:
        return 'Multiple updates to'
    verbs = ACTION_VERBS.get(category, ['Updated'])
    return verbs[0]

def generate_commit_message():
    """Generate a descriptive commit message based on changes"""
    files = get_changed_files()

    if not files:
        return "Empty commit (no changes)"

    # Count files by category
    counts = count_files_by_category(files)

    # Filter out empty categories
    counts = {k: v for k, v in counts.items() if v > 0}

    if not counts:
        return "Empty commit"

    # Generate message parts
    parts = []
    total_files = sum(counts.values())

    for category, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        verb = select_action_verb(category, count)
        parts.append(f"{verb} {category}: {count} update(s)")

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Combine parts
    if len(parts) == 1:
        msg = parts[0]
    elif len(parts) == 2:
        msg = f"{parts[0]} and {parts[1]}"
    else:
        msg = f"{', '.join(parts[:-1])}, and {parts[-1]}"

    # Add total count summary
    msg += f" ({total_files} file(s))"

    return f"{timestamp} | {msg}"

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'auto':
        # Auto-generate commit message
        msg = generate_commit_message()
        print(msg)
    else:
        print("Usage: ercot-snapshot-ai.py auto")
        print("This script is called by ercot-snapshot.sh")
        sys.exit(1)

if __name__ == '__main__':
    main()
