"""
Complete Project Runner
Executes the entire adaptive anomaly detection system in order:
1. Train the adaptive model
2. Start the API server
3. Start the dashboard

Usage: python run_all.py
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_training():
    """Step 1: Train the model"""
    print("🚀 Step 1: Training the adaptive model...")
    try:
        from core.training_pipeline import AdaptiveTrainingPipeline

        # Check if data file exists
        data_path = "data/train_001_final.xlsx"
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure the training data is in the data/ directory")
            return False

        pipeline = AdaptiveTrainingPipeline(data_path)
        pipeline.run(steps=60)  # Run simulation for 60 steps
        print("✅ Model training and simulation completed")
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def start_api_server():
    """Step 2: Start the API server"""
    print("🌐 Step 2: Starting API server...")
    try:
        # Start API server in background
        api_process = subprocess.Popen(
            [sys.executable, "api/api_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )

        # Wait a bit for server to start
        time.sleep(3)

        # Check if server is running
        if api_process.poll() is None:
            print("✅ API server started successfully on http://127.0.0.1:8000")
            return api_process
        else:
            stdout, stderr = api_process.communicate()
            print(f"❌ API server failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Failed to start API server: {str(e)}")
        return None

def start_dashboard():
    """Step 3: Start the dashboard"""
    print("📊 Step 3: Starting dashboard...")
    try:
        # Start Streamlit dashboard
        dashboard_process = subprocess.Popen(
            ["streamlit", "run", "dashboard/dashboard_legacy.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )

        # Wait a bit for dashboard to start
        time.sleep(5)

        # Check if dashboard is running
        if dashboard_process.poll() is None:
            print("✅ Dashboard started successfully")
            print("📱 Open your browser to view the dashboard")
            return dashboard_process
        else:
            stdout, stderr = dashboard_process.communicate()
            print(f"❌ Dashboard failed to start: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Failed to start dashboard: {str(e)}")
        return None

def main():
    """Main execution function"""
    print("="*60)
    print("🛩️  ADAPTIVE AVIATION ENGINE MONITORING SYSTEM")
    print("="*60)
    print()

    # Step 1: Training
    if not run_training():
        print("❌ System startup aborted due to training failure")
        return

    print()

    # Step 2: API Server
    api_process = start_api_server()
    if not api_process:
        print("❌ System startup aborted due to API server failure")
        return

    print()

    # Step 3: Dashboard
    dashboard_process = start_dashboard()
    if not dashboard_process:
        print("❌ Dashboard failed to start, but API server is running")
        print("You can manually start the dashboard with: streamlit run dashboard/dashboard_legacy.py")
        api_process.wait()  # Keep API running
        return

    print()
    print("="*60)
    print("🎉 SYSTEM STARTUP COMPLETE!")
    print("="*60)
    print("✅ Model trained and ready")
    print("✅ API server running on http://127.0.0.1:8000")
    print("✅ Dashboard running (check your browser)")
    print()
    print("Press Ctrl+C to stop all services")
    print("="*60)

    try:
        # Keep both processes running
        while True:
            time.sleep(1)
            # Check if either process has died
            if api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
            if dashboard_process.poll() is not None:
                print("❌ Dashboard stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        dashboard_process.wait()
        print("✅ All services stopped")

if __name__ == "__main__":
    main()