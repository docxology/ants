import concurrent.futures
import subprocess
import numpy as np

def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

if __name__ == "__main__":
    scripts = ["figure_1.py", "figure_2.py", "figure_3.py"]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_script, scripts)
