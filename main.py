import subprocess
import sys

if __name__ == "__main__":
    scripts = ["preprocess.py", "model.py", "visualize.py"]
    
    for script in scripts:
        print(f"\nRunning {script}...")
        result = subprocess.run([sys.executable, script], check=True)
    
    print("\nDone! Check data/ and plots/ for outputs.")