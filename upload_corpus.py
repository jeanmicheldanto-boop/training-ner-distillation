"""
Script pour uploader le corpus directement via un job RunPod
Ce script crée le corpus en mémoire et le sauvegarde sur le volume
"""
import requests
import json
import os
import sys
import base64

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("❌ ERREUR: Variables d'environnement manquantes")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

def upload_corpus_via_job(local_path, remote_path):
    """
    Upload corpus by creating a job that writes the content to disk
    """
    print(f"\n{'=' * 70}")
    print("📤 UPLOAD CORPUS VIA JOB RUNPOD")
    print("=" * 70)
    print(f"📁 Fichier local: {local_path}")
    print(f"🎯 Destination: {remote_path}")
    
    # Read corpus content
    print("\n📖 Lecture du fichier local...")
    try:
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.count('\n')
        size_kb = len(content.encode('utf-8')) / 1024
        print(f"✅ Fichier lu: {lines} lignes, {size_kb:.1f} KB")
        
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return None
    
    # Encode content as base64 to avoid JSON escaping issues
    content_b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
    
    # Create job payload
    payload = {
        "input": {
            "action": "upload_corpus",
            "remote_path": remote_path,
            "content_b64": content_b64
        }
    }
    
    print(f"\n📤 Envoi du corpus à l'endpoint...")
    print(f"📦 Taille payload: {len(json.dumps(payload)) / 1024:.1f} KB")
    
    try:
        response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("id")
            
            print(f"\n✅ Job d'upload soumis!")
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
        print("Usage: python upload_corpus.py <fichier_local> <chemin_remote>")
        print("\nExemples:")
        print('  python upload_corpus.py corpus\\corpus_test_100.txt /workspace/corpus_test_100.txt')
        print('  python upload_corpus.py corpus\\corpus_fr_100k_medico_FINAL.txt /workspace/corpus_fr_100k_medico_FINAL.txt')
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_path = sys.argv[2]
    
    if not os.path.exists(local_path):
        print(f"❌ Fichier introuvable: {local_path}")
        sys.exit(1)
    
    job_id = upload_corpus_via_job(local_path, remote_path)
    
    if job_id:
        print("\n" + "🎉" * 35)
        print("UPLOAD INITIÉ AVEC SUCCÈS")
        print("🎉" * 35)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
