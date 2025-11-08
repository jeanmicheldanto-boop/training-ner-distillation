"""
Upload corpus compressé en gzip pour contourner la limite de 10MB
Le handler décompresse automatiquement
"""
import requests
import json
import os
import sys
import base64
import gzip

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("❌ ERREUR: Variables d'environnement manquantes")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

def upload_corpus_compressed(local_path, remote_path):
    """Upload corpus compressé en gzip"""
    print(f"\n{'=' * 70}")
    print("📦 UPLOAD CORPUS COMPRESSÉ (GZIP)")
    print("=" * 70)
    print(f"📁 Fichier local: {local_path}")
    print(f"🎯 Destination: {remote_path}")
    
    # Lire et compresser
    print("\n📖 Lecture et compression...")
    with open(local_path, 'rb') as f:
        original_data = f.read()
    
    compressed_data = gzip.compress(original_data, compresslevel=9)
    
    original_size_kb = len(original_data) / 1024
    compressed_size_kb = len(compressed_data) / 1024
    ratio = (1 - compressed_size_kb / original_size_kb) * 100
    
    print(f"✅ Taille originale: {original_size_kb:.1f} KB")
    print(f"✅ Taille compressée: {compressed_size_kb:.1f} KB (réduction: {ratio:.1f}%)")
    
    # Encoder en base64
    content_b64 = base64.b64encode(compressed_data).decode('ascii')
    payload_size_kb = len(content_b64.encode('utf-8')) / 1024
    
    print(f"📦 Taille payload (base64): {payload_size_kb:.1f} KB")
    
    if payload_size_kb > 10000:
        print(f"\n❌ ERREUR: Même compressé, le fichier dépasse 10 MB!")
        print(f"   Taille actuelle: {payload_size_kb:.1f} KB")
        print(f"   Maximum: 10000 KB")
        return None
    
    # Créer payload avec action upload_corpus_gzip
    payload = {
        "input": {
            "action": "upload_corpus_gzip",  # Handler décompresse automatiquement
            "remote_path": remote_path,
            "content_b64": content_b64
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
        print("Usage: python upload_corpus_gzip.py <fichier_local> <chemin_remote>")
        print("\nExemple:")
        print('  python upload_corpus_gzip.py corpus\\corpus_fr_100k_medico_FINAL.txt /workspace/corpus_fr_100k_medico_FINAL.txt')
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_path = sys.argv[2]
    
    if not os.path.exists(local_path):
        print(f"❌ Fichier introuvable: {local_path}")
        sys.exit(1)
    
    job_id = upload_corpus_compressed(local_path, remote_path)
    
    if job_id:
        print("\n" + "🎉" * 35)
        print("UPLOAD INITIÉ AVEC SUCCÈS")
        print("🎉" * 35)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
