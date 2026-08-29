"""
SIH26077 — End-to-End Pipeline Runner
========================================
Orchestrates the complete pipeline:
  1. Verify data loader & spatial alignment
  2. Verify model architecture
  3. Train the model (saves best checkpoint)
  4. Run inference & generate risk maps
  5. Launch interactive web dashboard
"""

import subprocess
import sys
import os


def run_script(command, description):
    """Run a shell command, streaming output in real-time."""
    print(f"\n{'═' * 60}")
    print(f"  🚀 STEP: {description}")
    print(f"  🛠️ COMMAND: {command}")
    print(f"{'═' * 60}\n")
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n  ❌ FAILED: '{description}' exited with code {process.returncode}.")
            sys.exit(process.returncode)
        else:
            print(f"\n  ✅ SUCCESS: '{description}' completed.\n")
            
    except Exception as e:
        print(f"\n  ❌ EXCEPTION: {e}")
        sys.exit(1)


def main():
    print("🌟 SIH26077 — END-TO-END PIPELINE 🌟")
    print("AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting\n")
    
    # Ensure we're in the project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Verify data loading & spatial alignment
    run_script("python data_loader.py", "Checking Data Loader & Spatial Alignment")
    
    # 2. Verify model architecture
    run_script("python model.py", "Testing Spatiotemporal U-Net Architecture")
    
    # 3. Train the model
    if os.path.exists("checkpoints/best_model.pth"):
        print(f"\n{'═' * 60}")
        print("  🚀 STEP: Training Multi-Task Model")
        print("  ✅ SUCCESS: Found existing trained model in checkpoints/best_model.pth. Skipping training to preserve it.")
        print(f"{'═' * 60}\n")
    else:
        run_script("python train.py", "Training Multi-Task Model (GPU Accelerated)")
    
    # 4. Generate predictions
    run_script("python predict.py", "Running Inference & Generating Risk Maps")
    
    # 5. Test XAI module
    run_script("python xai.py", "Testing Explainable AI (GradCAM) Module")
    
    # 6. Test alerting
    run_script("python alerts.py", "Testing Automated Alert Engine")
    
    # 7. Launch dashboard
    print(f"\n{'═' * 60}")
    print("  🌐 STEP: Launching Interactive Web Dashboard")
    print("  🛠️ COMMAND: streamlit run app.py")
    print(f"{'═' * 60}\n")
    print("  The Streamlit dashboard will start on http://localhost:8501")
    print("  Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n  Pipeline stopped by user.")


if __name__ == "__main__":
    main()
