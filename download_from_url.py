"""
Télécharger un corpus depuis une URL vers le volume RunPod
Utile pour les très gros fichiers (> 30 MB compressés)
"""
import requests
import json
import os
import sys

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("❌ ERREUR: Variables d'environnement manquantes")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

def download_corpus(url, remote_path, timeout=600):
    """Télécharger corpus depuis URL vers volume RunPod"""
    print(f"\n{'=' * 70}")
    print("🌐 DOWNLOAD CORPUS FROM URL")
    print("=" * 70)
    print(f"🔗 URL: {url}")
    print(f"🎯 Destination: {remote_path}")
    print(f"⏱️  Timeout: {timeout}s")
    
    # Créer payload
    payload = {
        "input": {
            "action": "download_from_url",
            "url": url,
            "remote_path": remote_path,
            "timeout": timeout
        }
    }
    
    print(f"\n📤 Envoi à l'endpoint...")
    
    try:
        response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("id")
            
            print(f"\n✅ Job soumis!")
            print(f"🆔 Job ID: {job_id}")
            print(f"\n💡 Suivez le job avec:")
            print(f"   python monitor_jobs.py \"{job_id}\"")
            
            return job_id
        else:
            print(f"\n❌ Erreur HTTP {response.status_code}")
            print(f"📋 Réponse: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n❌ Exception: {type(e).__name__}: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python download_from_url.py <url> <chemin_remote> [timeout]")
        print("\nExemples:")
        print("  # GitHub Release:")
        print('  python download_from_url.py https://github.com/user/repo/releases/download/v1.0/corpus.txt /workspace/corpus.txt')
        print("\n  # Dropbox (use dl=1 at end):")
        print('  python download_from_url.py "https://www.dropbox.com/s/xxx/corpus.txt?dl=1" /workspace/corpus.txt')
        print("\n  # Google Drive (use export link):")
        print('  python download_from_url.py "https://drive.google.com/uc?export=download&id=FILE_ID" /workspace/corpus.txt')
        print("\n  # File.io (temporary, expire after 1 download):")
        print('  python download_from_url.py https://file.io/xxx /workspace/corpus.txt')
        sys.exit(1)
    
    url = sys.argv[1]
    remote_path = sys.argv[2]
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    
    job_id = download_corpus(url, remote_path, timeout)
    
    if job_id:
        print("\n" + "🎉" * 35)
        print("DOWNLOAD INITIÉ AVEC SUCCÈS")
        print("🎉" * 35)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
