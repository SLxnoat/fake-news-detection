#!/usr/bin/env python3
"""
Fake News Detection System Launcher
==================================

This script allows users to launch different components of the system:
1. Unified Application (recommended)
2. Individual Applications
3. API Server
4. Jupyter Notebooks
5. Quick Tests

Author: Team ITBIN-2211
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print the system banner"""
    print("=" * 70)
    print("🔍 FAKE NEWS DETECTION SYSTEM - UNIFIED LAUNCHER")
    print("=" * 70)
    print("Team: ITBIN-2211 (0149, 0169, 0173, 0184)")
    print("=" * 70)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def launch_unified_app():
    """Launch the unified application"""
    print("\n🚀 Launching Unified Fake News Detection Application...")
    
    app_path = "app/unified_app.py"
    if os.path.exists(app_path):
        try:
            # Start Streamlit app
            cmd = [sys.executable, "-m", "streamlit", "run", app_path]
            print(f"Running: {' '.join(cmd)}")
            
            # Open browser after a short delay
            def open_browser():
                time.sleep(3)
                webbrowser.open("http://localhost:8501")
            
            import threading
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Run the app
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print("\n🛑 Application stopped by user")
        except Exception as e:
            print(f"❌ Error launching unified app: {e}")
    else:
        print(f"❌ Unified app not found at: {app_path}")

def launch_individual_apps():
    """Launch individual applications"""
    print("\n🔧 Individual Applications:")
    
    apps = [
        ("Unified App", "app/unified_app.py"),
        ("Main App", "app/backend/app.py"),
        ("Advanced App", "app/backend/advanced_app.py"),
        ("Streamlit Demo", "app/streamlit_demo_0149.py"),
        ("Member 0169 App", "app/0169_streamlit_app.py")
    ]
    
    for i, (name, path) in enumerate(apps, 1):
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{i}. {status} {name}: {path}")
    
    choice = input("\nSelect app to launch (1-5) or press Enter to skip: ").strip()
    
    if choice == "1":
        launch_unified_app()
    elif choice == "2":
        launch_app("app/backend/app.py")
    elif choice == "3":
        launch_app("app/backend/advanced_app.py")
    elif choice == "4":
        launch_app("app/streamlit_demo_0149.py")
    elif choice == "5":
        launch_app("app/0169_streamlit_app.py")
    else:
        print("Skipping individual app launch")

def launch_app(app_path):
    """Launch a specific Streamlit app"""
    if os.path.exists(app_path):
        print(f"\n🚀 Launching {app_path}...")
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", app_path]
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n🛑 App stopped by user")
        except Exception as e:
            print(f"❌ Error launching app: {e}")
    else:
        print(f"❌ App not found: {app_path}")

def launch_api_server():
    """Launch the Flask API server"""
    print("\n🔌 Launching API Server...")
    
    api_path = "app/backend/api_server.py"
    if os.path.exists(api_path):
        try:
            print("Starting Flask API server on http://localhost:5000")
            print("Press Ctrl+C to stop the server")
            
            cmd = [sys.executable, api_path]
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print("\n🛑 API server stopped by user")
        except Exception as e:
            print(f"❌ Error launching API server: {e}")
    else:
        print(f"❌ API server not found at: {api_path}")

def launch_jupyter():
    """Launch Jupyter Notebook"""
    print("\n📓 Launching Jupyter Notebook...")
    
    try:
        # Check if Jupyter is available
        subprocess.run([sys.executable, "-c", "import jupyter"], check=True)
        
        print("Starting Jupyter Notebook...")
        print("Browser will open automatically to http://localhost:8888")
        
        # Open browser after delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8888")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch Jupyter
        subprocess.run([sys.executable, "-m", "jupyter", "notebook"])
        
    except subprocess.CalledProcessError:
        print("❌ Jupyter not installed. Install with: pip install jupyter")
    except KeyboardInterrupt:
        print("\n🛑 Jupyter stopped by user")
    except Exception as e:
        print(f"❌ Error launching Jupyter: {e}")

def run_quick_tests():
    """Run quick tests and checks"""
    print("\n🧪 Running Quick Tests...")
    
    tests = [
        ("Dependency Check", "python scripts/check_dependencies.py"),
        ("Quick Start", "python 0184_quick_start.py"),
        ("Install Dependencies", "python scripts/install_dependencies.py"),
        ("Train Baseline Models", "python scripts/train_baseline_models.py")
    ]
    
    for name, cmd in tests:
        print(f"\n🔍 Running: {name}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name} completed successfully")
                if result.stdout:
                    print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            else:
                print(f"❌ {name} failed")
                if result.stderr:
                    print("Error:", result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
        except Exception as e:
            print(f"❌ Error running {name}: {e}")

def train_baseline_models():
    """Train baseline models for the system"""
    print("\n🤖 Training Baseline Models...")
    
    script_path = "scripts/train_baseline_models.py"
    if os.path.exists(script_path):
        try:
            print("This will train baseline models for fake news detection.")
            print("This may take several minutes depending on your data size.")
            
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                print("🚀 Starting model training...")
                cmd = [sys.executable, script_path]
                subprocess.run(cmd)
            else:
                print("Training cancelled.")
                
        except KeyboardInterrupt:
            print("\n🛑 Training stopped by user")
        except Exception as e:
            print(f"❌ Error during training: {e}")
    else:
        print(f"❌ Training script not found at: {script_path}")

def show_system_info():
    """Show system information"""
    print("\n💻 System Information:")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Working directory
    print(f"Working Directory: {os.getcwd()}")
    
    # Project structure
    print("\n📁 Project Structure:")
    important_files = [
        "app/unified_app.py",
        "app/backend/app.py",
        "app/backend/api_server.py",
        "src/preprocessing/",
        "src/models/",
        "requirements.txt",
        "environment.yml"
    ]
    
    for file_path in important_files:
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"  {status} {file_path}")

def main_menu():
    """Main menu interface"""
    while True:
        print("\n" + "=" * 50)
        print("🎯 MAIN MENU")
        print("=" * 50)
        print("1. 🚀 Launch Unified Application (Recommended)")
        print("2. 🔧 Launch Individual Applications")
        print("3. 🔌 Launch API Server")
        print("4. 📓 Launch Jupyter Notebook")
        print("5. 🧪 Run Quick Tests")
        print("6. 🤖 Train Baseline Models")
        print("7. 💻 Show System Information")
        print("8. ❌ Exit")
        print("=" * 50)
        
        choice = input("\nSelect an option (1-8): ").strip()
        
        if choice == "1":
            launch_unified_app()
        elif choice == "2":
            launch_individual_apps()
        elif choice == "3":
            launch_api_server()
        elif choice == "4":
            launch_jupyter()
        elif choice == "5":
            run_quick_tests()
        elif choice == "6":
            train_baseline_models()
        elif choice == "7":
            show_system_info()
        elif choice == "8":
            print("\n👋 Goodbye! Thank you for using Fake News Detection System!")
            break
        else:
            print("❌ Invalid choice. Please select 1-8.")
        
        if choice in ["1", "2", "3", "4"]:
            input("\nPress Enter to return to main menu...")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️ Some dependencies are missing. Please install them first.")
        print("You can still try to launch applications, but they may not work properly.")
    
    # Show system info
    show_system_info()
    
    # Main menu
    main_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again.")
