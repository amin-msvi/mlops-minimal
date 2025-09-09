import json
import os
from datetime import datetime
from pathlib import Path
import uuid

class SimpleMonitor:
    def __init__(self, log_dir="monitoring_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate files for different types of logs
        self.requests_file = self.log_dir / "requests.jsonl"
        self.failures_file = self.log_dir / "failures.jsonl"
        self.stats_file = self.log_dir / "daily_stats.json"

    def log_request(self, request_data, response_data=None, error=None):
        """Log a request with response or error"""
        timestamp = datetime.now().isoformat()
        request_id = str(uuid.uuid4())[:8]
        
        log_entry = {
            "request_id": request_id,
            "timestamp": timestamp,
            "request": request_data,
        }
        
        if response_data:
            # Successful request
            log_entry["response"] = response_data
            log_entry["status"] = "success"
            self._append_to_file(self.requests_file, log_entry)
        
        if error:
            # Failed request
            log_entry["error"] = str(error)
            log_entry["status"] = "failed"
            self._append_to_file(self.failures_file, log_entry)
        
        # Update daily stats
        self._update_stats(success=response_data is not None)
        
        return request_id

    def _append_to_file(self, file_path, data):
        """Append data to JSONL file"""
        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _update_stats(self, success=True):
        """Update daily statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Load existing stats
        if self.stats_file.exists():
            with open(self.stats_file, "r") as f:
                stats = json.load(f)
        else:
            stats = {}
        
        # Initialize today's stats if not exists
        if today not in stats:
            stats[today] = {"total_requests": 0, "successful": 0, "failed": 0}
        
        # Update counters
        stats[today]["total_requests"] += 1
        if success:
            stats[today]["successful"] += 1
        else:
            stats[today]["failed"] += 1
        
        # Save updated stats
        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def get_today_stats(self):
        """Get today's statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        if not self.stats_file.exists():
            return {"total_requests": 0, "successful": 0, "failed": 0}
        
        with open(self.stats_file, "r") as f:
            stats = json.load(f)
        
        return stats.get(today, {"total_requests": 0, "successful": 0, "failed": 0})
