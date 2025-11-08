"""
Script de monitoring robuste pour suivre les jobs RunPod en temps réel
Usage:
    python monitor_jobs.py <job_id>
    python monitor_jobs.py <job_id> --timeout 3600
"""
import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("❌ ERREUR: Variables d'environnement manquantes")
    print("   Définissez RUNPOD_ENDPOINT_ID et RUNPOD_API_KEY avant de lancer ce script.")
    print("\nExemple PowerShell:")
    print('   $env:RUNPOD_ENDPOINT_ID = "wupg1xsork5mk7"')
    print('   $env:RUNPOD_API_KEY = "votre_clé"')
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
STATUS_URL_TEMPLATE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{{job_id}}"


def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def monitor_job(job_id, timeout_seconds=3600, poll_interval=5):
    """
    Monitor a RunPod job until completion or timeout
    
    Args:
        job_id: Job ID to monitor
        timeout_seconds: Maximum time to wait (default 1h)
        poll_interval: Seconds between status checks (default 5s)
    
    Returns:
        Final job status dict or None if timeout/error
    """
    status_url = STATUS_URL_TEMPLATE.format(job_id=job_id)
    start_time = time.time()
    timeout_time = start_time + timeout_seconds
    last_status = None
    iteration = 0
    
    print("\n" + "=" * 70)
    print(f"🔍 MONITORING JOB: {job_id}")
    print("=" * 70)
    print(f"⏱️  Timeout: {format_duration(timeout_seconds)}")
    print(f"🔄 Intervalle de polling: {poll_interval}s")
    print(f"🌐 Status URL: {status_url}")
    print("=" * 70 + "\n")
    
    while time.time() < timeout_time:
        iteration += 1
        elapsed = time.time() - start_time
        
        try:
            response = requests.get(status_url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️  [{format_duration(elapsed)}] HTTP {response.status_code}: {response.text}")
                time.sleep(poll_interval)
                continue
            
            result = response.json()
            current_status = result.get("status", "UNKNOWN")
            
            # Print status update if changed or every 10 iterations
            if current_status != last_status or iteration % 10 == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] [{format_duration(elapsed)}] Status: {current_status}")
                
                # Print additional info if available
                if "output" in result:
                    output = result["output"]
                    if isinstance(output, dict):
                        if "error" in output:
                            print(f"   ⚠️  Error: {output['error']}")
                        if "message" in output:
                            print(f"   💬 Message: {output['message']}")
                
                # Print delayTime if job is queued
                if current_status == "IN_QUEUE" and "delayTime" in result:
                    delay = result["delayTime"]
                    print(f"   ⏳ Delay time: {delay}ms")
                
                # Print executionTime if job is running/completed
                if "executionTime" in result:
                    exec_time = result["executionTime"]
                    print(f"   ⚡ Execution time: {exec_time}ms")
            
            last_status = current_status
            
            # Terminal states
            if current_status == "COMPLETED":
                print("\n" + "✅" * 35)
                print("✅ JOB COMPLÉTÉ AVEC SUCCÈS!")
                print("✅" * 35 + "\n")
                print(f"🕐 Durée totale: {format_duration(elapsed)}")
                
                # Print full output
                if "output" in result:
                    print("\n📦 OUTPUT:")
                    print(json.dumps(result["output"], indent=2, ensure_ascii=False))
                
                return result
            
            elif current_status == "FAILED":
                print("\n" + "❌" * 35)
                print("❌ JOB ÉCHOUÉ")
                print("❌" * 35 + "\n")
                print(f"🕐 Durée avant échec: {format_duration(elapsed)}")
                
                # Print error details
                if "output" in result:
                    print("\n⚠️  ERREUR:")
                    print(json.dumps(result["output"], indent=2, ensure_ascii=False))
                
                return result
            
            elif current_status == "CANCELLED":
                print("\n⚠️  Job annulé par l'utilisateur")
                return result
            
            # Continue monitoring
            time.sleep(poll_interval)
            
        except requests.RequestException as e:
            print(f"⚠️  [{format_duration(elapsed)}] Erreur réseau: {e}")
            time.sleep(poll_interval)
        except Exception as e:
            print(f"⚠️  [{format_duration(elapsed)}] Erreur: {type(e).__name__}: {e}")
            time.sleep(poll_interval)
    
    # Timeout reached
    print("\n" + "⏱️ " * 35)
    print(f"⏱️  TIMEOUT ATTEINT ({format_duration(timeout_seconds)})")
    print("⏱️ " * 35 + "\n")
    print(f"📊 Dernier status connu: {last_status}")
    print("💡 Le job continue à s'exécuter sur RunPod. Utilisez la console web pour suivre l'avancement.")
    
    return None


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python monitor_jobs.py <job_id> [--timeout SECONDS]")
        print("\nExemple:")
        print('   python monitor_jobs.py "fdb73da3-d662-42ec-98e1-bc4eaf5529e3-e2"')
        print('   python monitor_jobs.py "job-id" --timeout 7200  # 2 heures')
        sys.exit(1)
    
    job_id = sys.argv[1]
    timeout = 3600  # Default 1 hour
    
    # Parse timeout argument
    if "--timeout" in sys.argv:
        try:
            idx = sys.argv.index("--timeout")
            timeout = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("⚠️  Argument --timeout invalide, utilisation du timeout par défaut (3600s)")
    
    result = monitor_job(job_id, timeout_seconds=timeout)
    
    if result:
        final_status = result.get("status")
        sys.exit(0 if final_status == "COMPLETED" else 1)
    else:
        sys.exit(2)  # Timeout


if __name__ == "__main__":
    main()
