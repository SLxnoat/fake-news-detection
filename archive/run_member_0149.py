#!/usr/bin/env python3
"""
Member 0149 - Complete Workflow Runner
ITBIN-2211-0149 - Baseline Models Implementation
"""

import os
import sys
import subprocess

def run_jupyter_notebook(notebook_path):
    """Run a Jupyter notebook"""
    cmd = f'jupyter nbconvert --to notebook --execute --inplace "{notebook_path}"'
    subprocess.run(cmd, shell=True, check=True)

def run_streamlit_app():
    """Run the Streamlit application"""
    cmd = "streamlit run app/streamlit_demo_0149.py"
    subprocess.run(cmd, shell=True)

def main():
    print(" MEMBER 0149 - COMPLETE WORKFLOW")
    print("=" * 50)
    print("ITBIN-2211-0149 - Baseline Models & Performance Evaluation")
    print()
    
    workflow_steps = [
        ("1️⃣ Data Loading & Initial Setup", "notebooks/01_data_loading_0149.ipynb"),
        ("2️⃣ Baseline Training & Evaluation", "notebooks/02_baseline_training_evaluation_0149.ipynb"),
        ("3️⃣ Quick Testing", "notebooks/03_quick_test_0149.ipynb"),
        ("4️⃣ Streamlit Demo", "app/streamlit_demo_0149.py")
    ]
    
    print("Available workflow steps:")
    for i, (desc, file) in enumerate(workflow_steps, 1):
        print(f"{desc} - {file}")
    
    print("\\nSelect option:")
    print("1. Run Data Loading")
    print("2. Run Training & Evaluation") 
    print("3. Run Quick Test")
    print("4. Launch Streamlit Demo")
    print("5. Run Complete Pipeline (1→2→3)")
    print("6. Exit")
    
    choice = input("\\nEnter your choice (1-6): ").strip()
    
    try:
        if choice == "1":
            print("\\n Running Data Loading...")
            run_jupyter_notebook("notebooks/01_data_loading_0149.ipynb")
        elif choice == "2":
            print("\\n Running Training & Evaluation...")
            run_jupyter_notebook("notebooks/02_baseline_training_evaluation_0149.ipynb")
        elif choice == "3":
            print("\\n Running Quick Test...")
            run_jupyter_notebook("notebooks/03_quick_test_0149.ipynb")
        elif choice == "4":
            print("\\n Launching Streamlit Demo...")
            run_streamlit_app()
        elif choice == "5":
            print("\\n Running Complete Pipeline...")
            run_jupyter_notebook("notebooks/01_data_loading_0149.ipynb")
            run_jupyter_notebook("notebooks/02_baseline_training_evaluation_0149.ipynb") 
            run_jupyter_notebook("notebooks/03_quick_test_0149.ipynb")
            print(" Complete pipeline finished!")
        elif choice == "6":
            print(" Goodbye!")
            return
        else:
            print(" Invalid choice!")
            
    except Exception as e:
        print(f" Error: {e}")
        print("Make sure Jupyter and Streamlit are installed and all files exist.")

if __name__ == "__main__":
    main()