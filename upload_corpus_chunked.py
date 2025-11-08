"""
Upload corpus en chunks pour éviter la limite de 10MB de RunPod
"""
import requests
import json
import os
import sys
import base64
import math

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
API_KEY = os.environ.get("RUNPOD_API_KEY")

if not ENDPOINT_ID or not API_KEY:
    print("❌ ERREUR: Variables d'environnement manquantes")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{{job_id}}"

# Taille max par chunk en KB (7 MB pour être sûr, avec marge pour base64)
MAX_CHUNK_SIZE_KB = 7000

def wait_for_job(job_id, timeout=300):
    """Attendre qu'un job se termine"""
    import time
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(STATUS_URL.format(job_id=job_id), headers=HEADERS, timeout=10)
        if response.status_code == 200:
            result = response.json()
            status = result.get("status")
            if status == "COMPLETED":
                return True
            elif status == "FAILED":
                print(f"❌ Job {job_id} failed: {result.get('output')}")
                return False
        time.sleep(2)
    print(f"⏱️ Timeout waiting for job {job_id}")
    return False

def upload_chunk(lines, chunk_num, total_chunks, remote_base_path):
    """Upload un chunk de lignes"""
    content = "\n".join(lines)
    content_b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
    
    # Nom du fichier chunk
    remote_path = f"{remote_base_path}.part{chunk_num}"
    
    payload = {
        "input": {
            "action": "upload_corpus",
            "remote_path": remote_path,
            "content_b64": content_b64
        }
    }
    
    size_kb = len(content.encode('utf-8')) / 1024
    print(f"📤 Chunk {chunk_num}/{total_chunks} : {len(lines)} lignes, {size_kb:.1f} KB → {remote_path}")
    
    response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get("id")
        print(f"   Job ID: {job_id}")
        
        # Attendre que le chunk soit uploadé
        if wait_for_job(job_id):
            print(f"   ✅ Chunk {chunk_num} uploaded")
            return True
        else:
            return False
    else:
        print(f"   ❌ Erreur HTTP {response.status_code}: {response.text}")
        return False

def merge_chunks(remote_base_path, total_chunks):
    """Fusionner les chunks sur le serveur via un job Python"""
    merge_code = f"""
import os
output_path = "{remote_base_path}"
chunks = []
for i in range(1, {total_chunks + 1}):
    chunk_path = output_path + f".part{{i}}"
    if os.path.exists(chunk_path):
        with open(chunk_path, 'r', encoding='utf-8') as f:
            chunks.append(f.read())
        os.remove(chunk_path)  # Supprimer le chunk après lecture
    else:
        raise FileNotFoundError(f"Chunk {{chunk_path}} not found")

# Écrire le fichier final
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("\\n".join(chunks))

lines = sum(chunk.count("\\n") for chunk in chunks) + 1
size_kb = os.path.getsize(output_path) / 1024
print(f"✅ Merged {{len(chunks)}} chunks into {{output_path}}: {{lines}} lines, {{size_kb:.1f}} KB")
"""
    
    # Encoder le code Python en base64
    code_b64 = base64.b64encode(merge_code.encode('utf-8')).decode('ascii')
    
    payload = {
        "input": {
            "action": "upload_corpus",
            "remote_path": f"/tmp/merge_chunks.py",
            "content_b64": code_b64
        }
    }
    
    print(f"\n🔗 Fusion des {total_chunks} chunks...")
    response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get("id")
        print(f"   Job ID: {job_id}")
        
        if wait_for_job(job_id, timeout=600):
            print(f"   ✅ Chunks fusionnés dans {remote_base_path}")
            return True
    
    print(f"   ❌ Erreur lors de la fusion")
    return False

def upload_corpus_chunked(local_path, remote_path):
    """Upload corpus en chunks"""
    print(f"\n{'=' * 70}")
    print("📦 UPLOAD CORPUS PAR CHUNKS")
    print("=" * 70)
    print(f"📁 Fichier local: {local_path}")
    print(f"🎯 Destination: {remote_path}")
    
    # Lire le fichier
    print(f"\n📖 Lecture du fichier...")
    with open(local_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    total_lines = len(lines)
    total_kb = sum(len(line.encode('utf-8')) for line in lines) / 1024
    print(f"✅ {total_lines} lignes, {total_kb:.1f} KB")
    
    # Calculer le nombre de chunks nécessaires
    avg_line_size = total_kb / total_lines
    lines_per_chunk = int(MAX_CHUNK_SIZE_KB / avg_line_size * 0.9)  # Marge de sécurité
    total_chunks = math.ceil(total_lines / lines_per_chunk)
    
    print(f"\n📊 Découpage en {total_chunks} chunks de ~{lines_per_chunk} lignes")
    print(f"   Taille cible par chunk: < {MAX_CHUNK_SIZE_KB} KB")
    
    # Upload chaque chunk
    print(f"\n{'=' * 70}")
    for i in range(total_chunks):
        start_idx = i * lines_per_chunk
        end_idx = min((i + 1) * lines_per_chunk, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        
        if not upload_chunk(chunk_lines, i + 1, total_chunks, remote_path):
            print(f"\n❌ Échec upload chunk {i + 1}")
            return False
    
    print(f"\n{'=' * 70}")
    
    # Fusionner les chunks sur le serveur
    print(f"\n🔗 Fusion des {total_chunks} chunks en 1 fichier...")
    
    merge_script = f"""
import os
import glob

# Trouver tous les chunks
chunk_pattern = "{remote_path}.part*"
chunk_files = sorted(glob.glob(chunk_pattern))

if len(chunk_files) != {total_chunks}:
    raise Exception(f"Expected {total_chunks} chunks, found {{len(chunk_files)}}")

print(f"Merging {{len(chunk_files)}} chunks...")

# Fusionner
with open("{remote_path}", 'w', encoding='utf-8') as outfile:
    for i, chunk_file in enumerate(chunk_files, 1):
        print(f"  Reading chunk {{i}}/{{len(chunk_files)}}: {{chunk_file}}")
        with open(chunk_file, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())
            if i < len(chunk_files):
                outfile.write("\\n")  # Ajouter newline entre chunks
        os.remove(chunk_file)  # Supprimer chunk après fusion

# Vérifier
size_kb = os.path.getsize("{remote_path}") / 1024
with open("{remote_path}", 'r', encoding='utf-8') as f:
    line_count = sum(1 for _ in f)

print(f"✅ Merged file: {remote_path}")
print(f"   Lines: {{line_count}}")
print(f"   Size: {{size_kb:.1f}} KB")
"""
    
    merge_script_b64 = base64.b64encode(merge_script.encode('utf-8')).decode('ascii')
    
    # Uploader le script de fusion comme un fichier Python temporaire
    script_path = "/tmp/merge_corpus.py"
    payload = {
        "input": {
            "action": "upload_corpus",
            "remote_path": script_path,
            "content_b64": merge_script_b64
        }
    }
    
    response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=120)
    if response.status_code != 200:
        print(f"❌ Erreur upload script de fusion: {response.text}")
        return False
    
    script_job_id = response.json().get("id")
    print(f"   Script de fusion uploadé (job {script_job_id})")
    
    if not wait_for_job(script_job_id, timeout=60):
        print(f"❌ Échec upload script de fusion")
        return False
    
    # Maintenant, exécuter le script via Python
    # On utilise l'action upload_corpus pour créer un petit lanceur
    exec_script = f"""
import subprocess
result = subprocess.run(['python3', '{script_path}'], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise Exception(f"Merge failed with code {{result.returncode}}")
"""
    
    exec_script_b64 = base64.b64encode(exec_script.encode('utf-8')).decode('ascii')
    
    exec_payload = {
        "input": {
            "action": "upload_corpus",
            "remote_path": "/tmp/exec_merge.py",
            "content_b64": exec_script_b64
        }
    }
    
    # Note: Cette approche ne marche pas car upload_corpus n'exécute pas le code
    # On va plutôt créer un script Python qui fait tout en une fois
    
    print(f"⚠️  Fusion manuelle requise sur le serveur")
    print(f"   Les chunks sont dans: {remote_path}.part1 à .part{total_chunks}")
    print(f"\n💡 Utilisez ce code Python dans un notebook ou job:")
    print(f"""
import glob
chunks = sorted(glob.glob("{remote_path}.part*"))
with open("{remote_path}", 'w') as out:
    for chunk in chunks:
        with open(chunk, 'r') as f:
            out.write(f.read())
            out.write("\\n")
""")
    
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python upload_corpus_chunked.py <fichier_local> <chemin_remote>")
        print("\nExemple:")
        print('  python upload_corpus_chunked.py corpus\\corpus_fr_100k_medico_FINAL.txt /workspace/corpus_fr_100k_medico_FINAL.txt')
        sys.exit(1)
    
    local_path = sys.argv[1]
    remote_path = sys.argv[2]
    
    if not os.path.exists(local_path):
        print(f"❌ Fichier introuvable: {local_path}")
        sys.exit(1)
    
    if upload_corpus_chunked(local_path, remote_path):
        print("\n" + "🎉" * 35)
        print("TOUS LES CHUNKS UPLOADÉS AVEC SUCCÈS")
        print("🎉" * 35)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
