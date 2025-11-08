"""
Script d'orchestration complète pour le workflow NER Distillation sur RunPod

Ce script orchestre:
1. Vérification du corpus uploadé
2. Lancement de l'annotation (avec le gros corpus)
3. Attente de fin de l'annotation
4. Lancement du training
5. Attente de fin du training
6. Affichage des résultats

Usage:
    python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt
    python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt --skip-annotation
"""
import requests
import json
import time
import os
import sys
import argparse
from datetime import datetime

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
RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
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


def submit_job(action, payload):
    """Submit a job to the RunPod endpoint"""
    print(f"\n{'=' * 70}")
    print(f"📤 SOUMISSION JOB: {action.upper()}")
    print("=" * 70)
    print(f"📦 Payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(RUN_URL, json=payload, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("id")
            status = result.get("status")
            
            print(f"\n✅ Job soumis avec succès!")
            print(f"🆔 Job ID: {job_id}")
            print(f"📊 Status initial: {status}")
            
            return job_id
        else:
            print(f"\n❌ Erreur HTTP {response.status_code}")
            print(f"📋 Réponse: {response.text}")
            return None
            
    except Exception as e:
        print(f"\n❌ Exception lors de la soumission: {type(e).__name__}: {e}")
        return None


def wait_for_job(job_id, timeout_seconds=7200, poll_interval=10):
    """
    Wait for a job to complete with progress updates
    
    Args:
        job_id: Job ID to monitor
        timeout_seconds: Maximum wait time (default 2h)
        poll_interval: Seconds between checks (default 10s)
    
    Returns:
        Final job result dict or None if timeout/error
    """
    status_url = STATUS_URL_TEMPLATE.format(job_id=job_id)
    start_time = time.time()
    timeout_time = start_time + timeout_seconds
    last_status = None
    iteration = 0
    
    print(f"\n{'=' * 70}")
    print(f"⏳ ATTENTE COMPLETION: {job_id}")
    print("=" * 70)
    print(f"⏱️  Timeout: {format_duration(timeout_seconds)}")
    print(f"🔄 Intervalle: {poll_interval}s")
    
    while time.time() < timeout_time:
        iteration += 1
        elapsed = time.time() - start_time
        
        try:
            response = requests.get(status_url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️  [{format_duration(elapsed)}] HTTP {response.status_code}")
                time.sleep(poll_interval)
                continue
            
            result = response.json()
            current_status = result.get("status", "UNKNOWN")
            
            # Progress update
            if current_status != last_status or iteration % 6 == 0:  # Every minute
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] [{format_duration(elapsed)}] 📊 {current_status}", end="")
                
                if "executionTime" in result:
                    exec_ms = result["executionTime"]
                    print(f" (exec: {exec_ms}ms)", end="")
                
                print()  # Newline
            
            last_status = current_status
            
            # Terminal states
            if current_status == "COMPLETED":
                print(f"\n✅ Job complété en {format_duration(elapsed)}")
                return result
            
            elif current_status == "FAILED":
                print(f"\n❌ Job échoué après {format_duration(elapsed)}")
                if "output" in result:
                    print(f"⚠️  Erreur: {json.dumps(result['output'], indent=2, ensure_ascii=False)}")
                return result
            
            elif current_status == "CANCELLED":
                print(f"\n⚠️  Job annulé après {format_duration(elapsed)}")
                return result
            
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"⚠️  [{format_duration(elapsed)}] Erreur: {e}")
            time.sleep(poll_interval)
    
    print(f"\n⏱️  Timeout atteint ({format_duration(timeout_seconds)})")
    print(f"📊 Dernier status: {last_status}")
    return None


def run_workflow(corpus_path, skip_annotation=False, annotation_timeout=7200, training_timeout=14400):
    """
    Run the complete NER distillation workflow
    
    Args:
        corpus_path: Path to corpus file on RunPod volume
        skip_annotation: Skip annotation step (use existing data)
        annotation_timeout: Max time for annotation (default 2h)
        training_timeout: Max time for training (default 4h)
    """
    print("\n" + "🚀" * 35)
    print("WORKFLOW NER DISTILLATION - RUNPOD")
    print("🚀" * 35)
    print(f"\n📅 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Corpus: {corpus_path}")
    print(f"🔀 Skip annotation: {skip_annotation}")
    print(f"⏱️  Timeout annotation: {format_duration(annotation_timeout)}")
    print(f"⏱️  Timeout training: {format_duration(training_timeout)}")
    
    workflow_start = time.time()
    
    # Step 1: Annotation (if not skipped)
    if not skip_annotation:
        print("\n" + "📝" * 35)
        print("ÉTAPE 1/2: ANNOTATION DU CORPUS")
        print("📝" * 35)
        
        annotation_payload = {
            "input": {
                "action": "annotate",
                "corpus_path": corpus_path,
                "output_dir": "/app/training_ner/data"
            }
        }
        
        annotation_job_id = submit_job("annotation", annotation_payload)
        
        if not annotation_job_id:
            print("❌ Échec de soumission du job d'annotation")
            return False
        
        print(f"\n💡 Console: https://console.runpod.io/jobs?id={annotation_job_id}")
        
        annotation_result = wait_for_job(annotation_job_id, timeout_seconds=annotation_timeout)
        
        if not annotation_result or annotation_result.get("status") != "COMPLETED":
            print("\n❌ L'annotation a échoué ou timeout. Arrêt du workflow.")
            return False
        
        print("\n✅ Annotation terminée avec succès!")
        if "output" in annotation_result:
            print(f"📋 Output: {json.dumps(annotation_result['output'], indent=2, ensure_ascii=False)}")
    else:
        print("\n⏭️  Annotation skippée (utilisation des données existantes)")
    
    # Step 2: Training
    print("\n" + "🎓" * 35)
    print("ÉTAPE 2/2: TRAINING & DISTILLATION")
    print("🎓" * 35)
    
    training_payload = {
        "input": {
            "action": "train",
            "config": "/app/training_ner/configs/kd_camembert.yaml",
            "output_dir": "/app/artifacts"
        }
    }
    
    training_job_id = submit_job("training", training_payload)
    
    if not training_job_id:
        print("❌ Échec de soumission du job de training")
        return False
    
    print(f"\n💡 Console: https://console.runpod.io/jobs?id={training_job_id}")
    
    training_result = wait_for_job(training_job_id, timeout_seconds=training_timeout)
    
    if not training_result or training_result.get("status") != "COMPLETED":
        print("\n❌ Le training a échoué ou timeout.")
        return False
    
    print("\n✅ Training terminé avec succès!")
    if "output" in training_result:
        print(f"📋 Output: {json.dumps(training_result['output'], indent=2, ensure_ascii=False)}")
    
    # Workflow summary
    workflow_duration = time.time() - workflow_start
    
    print("\n" + "🎉" * 35)
    print("WORKFLOW TERMINÉ AVEC SUCCÈS!")
    print("🎉" * 35)
    print(f"\n⏱️  Durée totale: {format_duration(workflow_duration)}")
    print(f"📅 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Prochaines étapes:")
    print("   1. Télécharger les artefacts depuis /app/artifacts sur le volume RunPod")
    print("   2. Évaluer le modèle distillé sur un jeu de test")
    print("   3. Comparer les performances teacher vs student")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Orchestration du workflow NER Distillation sur RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Workflow complet (annotation + training)
  python workflow.py --corpus-path /runpod-volume/corpus_fr_100k_medico_FINAL.txt
  
  # Training uniquement (données déjà annotées)
  python workflow.py --corpus-path /runpod-volume/corpus.txt --skip-annotation
  
  # Avec timeouts personnalisés (en secondes)
  python workflow.py --corpus-path /runpod-volume/corpus.txt --annotation-timeout 3600 --training-timeout 7200
        """
    )
    
    parser.add_argument(
        "--corpus-path",
        required=True,
        help="Chemin du corpus sur le volume RunPod (ex: /runpod-volume/corpus_fr_100k_medico_FINAL.txt)"
    )
    
    parser.add_argument(
        "--skip-annotation",
        action="store_true",
        help="Skipper l'annotation (utiliser des données existantes)"
    )
    
    parser.add_argument(
        "--annotation-timeout",
        type=int,
        default=7200,
        help="Timeout pour l'annotation en secondes (défaut: 7200 = 2h)"
    )
    
    parser.add_argument(
        "--training-timeout",
        type=int,
        default=14400,
        help="Timeout pour le training en secondes (défaut: 14400 = 4h)"
    )
    
    args = parser.parse_args()
    
    success = run_workflow(
        corpus_path=args.corpus_path,
        skip_annotation=args.skip_annotation,
        annotation_timeout=args.annotation_timeout,
        training_timeout=args.training_timeout
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
