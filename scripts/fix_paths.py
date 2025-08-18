#!/usr/bin/env python3
"""
Path Fix Utility for Fake News Detection System
==============================================

This script fixes common path and import issues across the project.
"""

import os
import sys
import shutil
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__)
    return current_file.parent.parent

def fix_sys_path_imports():
    """Fix sys.path.append statements in Python files"""
    print("🔧 Fixing sys.path imports...")
    
    project_root = get_project_root()
    src_path = project_root / 'src'
    
    # Files that need path fixes
    files_to_fix = [
        'app/unified_app.py',
        'app/backend/api_server.py',
        'scripts/train_baseline_models.py',
        'scripts/test_baseline_models.py',
        'scripts/create_processed_dataset.py',
        'launcher.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if path needs fixing
                if 'sys.path.append' in content:
                    # Replace relative paths with absolute paths
                    old_patterns = [
                        "sys.path.append('../../src')",
                        "sys.path.append('../../src')",
                        "sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))",
                        "sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))",
                        "sys.path.append('src')"
                    ]
                    
                    new_pattern = f"sys.path.append(str(Path(__file__).parent.parent / 'src'))"
                    
                    for old_pattern in old_patterns:
                        if old_pattern in content:
                            content = content.replace(old_pattern, new_pattern)
                            # Add Path import if not present
                            if "from pathlib import Path" not in content:
                                content = "from pathlib import Path\n" + content
                    
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Fixed: {file_path}")
                    fixed_count += 1
                else:
                    print(f"⏭️  No path issues: {file_path}")
                    
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"🔧 Fixed {fixed_count} files")

def create_path_helper():
    """Create a path helper module"""
    print("📁 Creating path helper module...")
    
    project_root = get_project_root()
    helper_content = '''"""
Path Helper for Fake News Detection System
=========================================

This module provides consistent path resolution across the project.
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__)
    return current_file.parent.parent

def get_src_path():
    """Get the src directory path"""
    return get_project_root() / 'src'

def setup_paths():
    """Setup Python paths for the project"""
    src_path = get_src_path()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path

def get_data_path():
    """Get the data directory path"""
    return get_project_root() / 'data'

def get_models_path():
    """Get the models directory path"""
    return get_project_root() / 'models'

def get_processed_data_path():
    """Get the processed data directory path"""
    return get_data_path() / 'processed'

def get_raw_data_path():
    """Get the raw data directory path"""
    return get_data_path() / 'raw'

# Auto-setup paths when imported
setup_paths()
'''
    
    helper_path = project_root / 'src' / 'utils' / 'path_helper.py'
    helper_path.parent.mkdir(exist_ok=True)
    
    with open(helper_path, 'w', encoding='utf-8') as f:
        f.write(helper_content)
    
    print(f"✅ Created: {helper_path}")

def check_data_files():
    """Check and report on data file availability"""
    print("📊 Checking data files...")
    
    project_root = get_project_root()
    
    # Check processed data
    processed_dir = project_root / 'data' / 'processed'
    if processed_dir.exists():
        processed_files = list(processed_dir.glob('*.csv'))
        print(f"✅ Processed data: {len(processed_files)} files")
        for file in processed_files:
            print(f"  - {file.name}")
    else:
        print("❌ Processed data directory not found")
    
    # Check raw data
    raw_dir = project_root / 'data' / 'raw'
    if raw_dir.exists():
        raw_files = list(raw_dir.glob('*.tsv'))
        print(f"✅ Raw data: {len(raw_files)} files")
        for file in raw_files:
            print(f"  - {file.name}")
    else:
        print("❌ Raw data directory not found")
    
    # Check models
    models_dir = project_root / 'models'
    if models_dir.exists():
        model_files = list(models_dir.glob('**/*.pkl'))
        print(f"✅ Models: {len(model_files)} files")
        for file in model_files:
            print(f"  - {file.relative_to(models_dir)}")
    else:
        print("❌ Models directory not found")

def create_processed_data_if_missing():
    """Create processed data if missing"""
    print("🔄 Creating processed data if missing...")
    
    project_root = get_project_root()
    processed_dir = project_root / 'data' / 'processed'
    raw_dir = project_root / 'data' / 'raw'
    
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {processed_dir}")
    
    # Check if processed data exists
    processed_files = list(processed_dir.glob('*.csv'))
    if not processed_files and raw_dir.exists():
        print("⚠️  No processed data found, but raw data exists")
        print("Run: python scripts/create_processed_dataset.py")
    elif not processed_files and not raw_dir.exists():
        print("❌ No data files found")
        print("Please add data files to data/raw/ directory")

def main():
    """Main function"""
    print("🔧 Fake News Detection - Path Fix Utility")
    print("=" * 50)
    
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Fix sys.path imports
    fix_sys_path_imports()
    
    # Create path helper
    create_path_helper()
    
    # Check data files
    check_data_files()
    
    # Create processed data if missing
    create_processed_data_if_missing()
    
    print("\n✅ Path fixes completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create processed data: python scripts/create_processed_dataset.py")
    print("3. Train models: python scripts/train_baseline_models.py")
    print("4. Run app: python launcher.py")

if __name__ == "__main__":
    main()
