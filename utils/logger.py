import datetime

LOG_FILE = "event_log.txt"

def log_event(event_type, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {event_type}: {message}\n")

def view_log_file():
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()
            print(content if content else "[INFO] Log file is empty.")
    except FileNotFoundError:
        print("[ERROR] Log file not found.")
