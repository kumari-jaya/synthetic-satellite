#!/usr/bin/env python3
"""
Migration script to help users update their code from TileFormer to Vortx.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPLACEMENTS = [
    # Import statements
    (r'from tileformer\.', 'from vortx.'),
    (r'import tileformer', 'import vortx'),
    # CLI commands
    (r'tileformer\s+([a-z]+)', r'vo \1'),
    # Environment variables
    (r'TILEFORMER_', 'VORTX_'),
    # Configuration
    (r'name:\s*tileformer', 'name: vortx'),
    (r'package:\s*tileformer', 'package: vortx'),
    # Docker
    (r'vortx/tileformer:', 'vortx/vortx:'),
    # URLs
    (r'tileformer\.vortx\.ai', 'vortx.ai'),
    (r'github\.com/vortx-ai/tileformer', 'github.com/vortx-ai/vortx'),
]

def should_process_file(file_path: Path) -> bool:
    """Check if the file should be processed."""
    EXCLUDE_DIRS = {'.git', '.github', 'venv', 'env', '__pycache__', 'node_modules'}
    INCLUDE_EXTENSIONS = {'.py', '.yml', '.yaml', '.md', '.rst', '.txt', '.sh', '.ipynb'}
    
    # Check if file is in excluded directory
    if any(part in EXCLUDE_DIRS for part in file_path.parts):
        return False
    
    # Check file extension
    return file_path.suffix.lower() in INCLUDE_EXTENSIONS

def process_file(file_path: Path) -> Tuple[int, List[str]]:
    """Process a single file and return number of changes and modified lines."""
    changes = 0
    modified_lines = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = content
        for pattern, replacement in REPLACEMENTS:
            new_content, count = re.subn(pattern, replacement, new_content)
            changes += count
            if count > 0:
                modified_lines.append(f"- Replaced '{pattern}' with '{replacement}' ({count} times)")
        
        if changes > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0, []
    
    return changes, modified_lines

def migrate_project(project_path: str, dry_run: bool = False) -> None:
    """Migrate a project from TileFormer to Vortx."""
    project_path = Path(project_path).resolve()
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    
    total_changes = 0
    modified_files = []
    
    logger.info(f"{'Analyzing' if dry_run else 'Migrating'} project: {project_path}")
    
    for root, _, files in os.walk(project_path):
        for file in files:
            file_path = Path(root) / file
            if should_process_file(file_path):
                changes, modified_lines = process_file(file_path)
                if changes > 0:
                    rel_path = file_path.relative_to(project_path)
                    modified_files.append((rel_path, changes, modified_lines))
                    total_changes += changes
    
    # Print summary
    logger.info("\nMigration Summary:")
    logger.info(f"Total files modified: {len(modified_files)}")
    logger.info(f"Total changes made: {total_changes}")
    
    if modified_files:
        logger.info("\nModified files:")
        for file_path, changes, lines in modified_files:
            logger.info(f"\n{file_path} ({changes} changes):")
            for line in lines:
                logger.info(f"  {line}")
    
    if dry_run:
        logger.info("\nThis was a dry run. No files were modified.")
    else:
        logger.info("\nMigration completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Migrate project from TileFormer to Vortx")
    parser.add_argument("project_path", help="Path to the project to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    args = parser.parse_args()
    
    try:
        migrate_project(args.project_path, args.dry_run)
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 