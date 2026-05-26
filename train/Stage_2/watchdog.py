import subprocess
import time

# Absolute path to your shell script
# NOTE: Update this to match your main training script (e.g., finetune_grpo.sh)
sh_script_path = "./finetune_grpo.sh"

def is_process_running(script_path):
    # Use pgrep or ps to find processes matching the script name
    try:
        # pgrep -f matches the full command line
        result = subprocess.run(["pgrep", "-f", script_path], stdout=subprocess.PIPE)
        if result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking process: {e}")
        return False

def start_process(script_path):
    try:
        # Start script as a subprocess without blocking the current program
        subprocess.Popen(["bash", script_path])
        print(f"Started script: {script_path}")
    except Exception as e:
        print(f"Error starting script: {e}")

if __name__ == "__main__":
    while True:
        if is_process_running(sh_script_path):
            print("Process is running, skipping")
        else:
            print("Process not running, restarting")
            start_process(sh_script_path)
        # Sleep for 5 minutes
        time.sleep(300)
