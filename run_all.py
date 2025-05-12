import subprocess
import sys
import os
from datetime import datetime
from glob import glob
import argparse

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False
    class Dummy:
        RESET = RED = GREEN = YELLOW = ''
    Fore = Style = Dummy()

NOTEBOOKS = [
    ("archives/01_data_cleaning.py", ["data/employee_data_cleaned.csv"]),
    ("archives/03_feature_engineering.py", ["data/employee_data_features.csv"]),
    ("archives/02_eda.py", ["results/numeric_summary.csv", "results/categorical_summary.csv"]),
    ("archives/04_modeling.py", ["models/final_lda_model", "results/confusion_matrix.md"]),
    ("archives/05_inference.py", ["results/predictions.csv"]), # adjust as needed
]

LOG_DIR = "logs"
MAX_LOGS_PER_SCRIPT = 5
os.makedirs(LOG_DIR, exist_ok=True)

REQUIRED_PACKAGES = ["pandas", "numpy", "altair", "scikit-learn", "pycaret", "IPython"]

STOP_ON_ERROR = True  # Set to False to continue on error

def check_python_and_packages():
    print(f"Python version: {sys.version}")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"{Fore.YELLOW if COLOR_ENABLED else ''}WARNING: Missing packages: {', '.join(missing)}. Some scripts may fail!{Style.RESET_ALL if COLOR_ENABLED else ''}")
    else:
        print(f"{Fore.GREEN if COLOR_ENABLED else ''}All required packages are installed.{Style.RESET_ALL if COLOR_ENABLED else ''}")

def clean_old_logs(script_name):
    logs = sorted(glob(os.path.join(LOG_DIR, f"{script_name}_*.log")), reverse=True)
    for old_log in logs[MAX_LOGS_PER_SCRIPT:]:
        os.remove(old_log)

def outputs_up_to_date(expected_outputs, script_path):
    if not expected_outputs:
        return False
    if not all(os.path.exists(f) for f in expected_outputs):
        return False
    script_mtime = os.path.getmtime(script_path)
    for out in expected_outputs:
        if os.path.getmtime(out) < script_mtime:
            return False
    return True

def run_script(script_path, expected_outputs):
    script_name = os.path.basename(script_path).replace('.py', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f"{script_name}_{timestamp}.log")
    print(f"\n=== Running: {script_path} ===")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, check=True
        )
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
            f.write('\n--- STDERR ---\n')
            f.write(result.stderr)
        print(f"{Fore.GREEN if COLOR_ENABLED else ''}--- {script_path} completed successfully. Log: {log_file} ---{Style.RESET_ALL if COLOR_ENABLED else ''}")
        # Check for expected outputs
        missing_outputs = [f for f in expected_outputs if not os.path.exists(f)]
        if missing_outputs:
            print(f"{Fore.YELLOW if COLOR_ENABLED else ''}WARNING: Missing expected outputs: {', '.join(missing_outputs)}{Style.RESET_ALL if COLOR_ENABLED else ''}")
        clean_old_logs(script_name)
        return True
    except subprocess.CalledProcessError as e:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(e.stdout or '')
            f.write('\n--- STDERR ---\n')
            f.write(e.stderr or '')
        print(f"{Fore.RED if COLOR_ENABLED else ''}!!! Error in {script_path} !!! See log: {log_file}{Style.RESET_ALL if COLOR_ENABLED else ''}")
        clean_old_logs(script_name)
        return False
    except Exception as ex:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n--- UNEXPECTED EXCEPTION ---\n{str(ex)}\n")
        print(f"{Fore.RED if COLOR_ENABLED else ''}!!! Unexpected error in {script_path}: {ex} !!! See log: {log_file}{Style.RESET_ALL if COLOR_ENABLED else ''}")
        clean_old_logs(script_name)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run all notebook scripts with checkpoint/resume support.")
    parser.add_argument('--force', action='store_true', help='Force re-run all scripts, even if outputs are up-to-date.')
    parser.add_argument('--from-step', type=int, default=1, help='Start from this step (1-based index).')
    args = parser.parse_args()

    check_python_and_packages()
    all_passed = True
    failed_scripts = []
    for idx, (script, outputs) in enumerate(NOTEBOOKS, start=1):
        if idx < args.from_step:
            continue
        if not os.path.exists(script):
            print(f"{Fore.RED if COLOR_ENABLED else ''}File not found: {script}{Style.RESET_ALL if COLOR_ENABLED else ''}")
            all_passed = False
            failed_scripts.append(script)
            if STOP_ON_ERROR:
                break
            continue
        if not args.force and outputs_up_to_date(outputs, script):
            print(f"{Fore.GREEN if COLOR_ENABLED else ''}Skipping {script} (outputs up-to-date).{Style.RESET_ALL if COLOR_ENABLED else ''}")
            continue
        success = run_script(script, outputs)
        if not success:
            all_passed = False
            failed_scripts.append(script)
            if STOP_ON_ERROR:
                break
    if all_passed:
        print(f"\n{Fore.GREEN if COLOR_ENABLED else ''}All scripts ran successfully!{Style.RESET_ALL if COLOR_ENABLED else ''}")
    else:
        print(f"\n{Fore.RED if COLOR_ENABLED else ''}Some scripts failed:{Style.RESET_ALL if COLOR_ENABLED else ''}")
        for script in failed_scripts:
            print(f"{Fore.RED if COLOR_ENABLED else ''} - {script}{Style.RESET_ALL if COLOR_ENABLED else ''}")
        print(f"{Fore.YELLOW if COLOR_ENABLED else ''}Please check the logs/ directory for details.{Style.RESET_ALL if COLOR_ENABLED else ''}")

if __name__ == '__main__':
    main() 