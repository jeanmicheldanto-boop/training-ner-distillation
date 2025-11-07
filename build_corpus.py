#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_corpus_fr_stratified.py

Construction d'un corpus français stratifié pour distillation de modèle NER.
Combine Wikipedia, sources administratives, narratifs et données locales.

Usage:
    python build_corpus_fr_stratified.py --target-size 100000 \\
        --wiki-max-pages 6000 \\
        --local-dir data_local \\
        --admin-dir sources_admin \\
        --narr-dir sources_narr \\
        --quota "wiki=0.35,admin=0.45,narr=0.20,other=0.00" \\
        --out corpus/corpus_fr_100k_medico.txt
"""

import argparse
import random
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sentence splitting (optionnel : blingfire)
try:
    from blingfire import text_to_sentences
    BLINGFIRE_AVAILABLE = True
except ImportError:
    BLINGFIRE_AVAILABLE = False

# Filtres de qualité
MIN_CHARS = 15
MAX_CHARS = 350
MIN_ALPHA_RATIO = 0.6

# Mots français fréquents (heuristique simple)
FRENCH_HINTS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "à", "au", "aux",
    "ce", "cette", "ces", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "est", "sont", "a", "ont", "être", "avoir", "que", "qui", "quoi", "où"
}

# User-Agent respectueux
USER_AGENT = "Mozilla/5.0 (compatible; CorpusBuilder/1.0; +https://github.com/yourproject)"

# ============================================================================
# SOURCES MÉDICO-SOCIALES (URLs PUBLIQUES)
# ============================================================================

# Sources administratives médico-sociales françaises
ADMIN_URLS = [
    # DREES (Études et statistiques)
    "https://drees.solidarites-sante.gouv.fr/publications",
    "https://drees.solidarites-sante.gouv.fr/etudes-et-statistiques",
    # IRDES (Institut de recherche en santé)
    "https://www.irdes.fr/recherche/questions-d-economie-de-la-sante.html",
    # Santé Publique France
    "https://www.santepubliquefrance.fr/revues-et-ouvrages",
    # HAS - Recommandations publiques
    "https://www.has-sante.fr/jcms/fc_2875473/fr/toutes-nos-publications",
    # CNSA - Documentation
    "https://www.cnsa.fr/documentation",
    # IGAS - Rapports publics
    "https://www.igas.gouv.fr/spip.php?rubrique7",
    # Vie publique - Rapports santé/social
    "https://www.vie-publique.fr/fiches/268992-quest-ce-que-le-secteur-medico-social",
    # Wikisource - Textes législatifs santé
    "https://fr.wikisource.org/wiki/Cat%C3%A9gorie:Sant%C3%A9",
    # HAL - Archive ouverte (santé publique)
    "https://hal.science/search/index/?q=*&fq=docType_s:ART&fq=domain_s:sdv.spee",
]

# Sources narratives médico-sociales françaises
NARR_URLS = [
    # Associations handicap
    "https://www.unapei.org/actualites/",
    "https://www.fondation-groupama.com/nos-actions/sante-handicap/",
    # Témoignages et blogs médicosociaux
    "https://www.hizy.org/fr",
    "https://blog.helpy.fr/",
    # Revues et magazines médico-sociaux
    "https://www.lien-social.com/",
    "https://www.ash.tm.fr/",
    # Medisite - Articles santé grand public
    "https://www.medisite.fr/",
    # Doctissimo - Forums santé (modération stricte)
    "https://www.doctissimo.fr/sante",
    # Ameli - Assurance Maladie (FAQ publiques)
    "https://www.ameli.fr/assure/sante",
]

# Corpus pré-existants (téléchargement direct) - DÉSACTIVÉ
# Utiliser les PDFs locaux à la place
CORPUS_DOWNLOADS = []

# ============================================================================
# SENTENCE SPLITTING
# ============================================================================

def sent_split(text: str) -> List[str]:
    """Découpe le texte en phrases (blingfire ou regex fallback)."""
    if BLINGFIRE_AVAILABLE:
        return text_to_sentences(text).split("\n")
    else:
        # Fallback regex basique
        return re.split(r'(?<=[\.\!\?…:;])\s+', text)

# ============================================================================
# FILTRES DE QUALITÉ
# ============================================================================

def looks_french(s: str) -> bool:
    """Filtre heuristique : longueur, ratio alpha, présence mots français."""
    s_clean = s.strip()
    if len(s_clean) < MIN_CHARS or len(s_clean) > MAX_CHARS:
        return False
    
    alpha_count = sum(c.isalpha() for c in s_clean)
    if alpha_count / max(len(s_clean), 1) < MIN_ALPHA_RATIO:
        return False
    
    words_lower = set(re.findall(r'\b\w+\b', s_clean.lower()))
    if not (words_lower & FRENCH_HINTS):
        return False
    
    return True

# ============================================================================
# LECTURE FICHIERS TXT (RÉCURSIF)
# ============================================================================

def read_txt_dir(root: Path) -> List[str]:
    """
    Lit TOUS les fichiers .txt du dossier (récursif).
    Retourne une liste de phrases filtrées.
    """
    sentences = []
    if not root.exists():
        return sentences
    
    txt_files = list(root.rglob("*.txt"))
    for fpath in tqdm(txt_files, desc=f"   Lecture TXT {root.name}"):
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            for s in sent_split(content):
                if looks_french(s):
                    sentences.append(s.strip())
        except Exception as e:
            # Ignorer fichiers corrompus
            continue
    
    return sentences

# ============================================================================
# LECTURE FICHIERS PDF (RÉCURSIF)
# ============================================================================

def read_pdf_dir(root: Path) -> List[str]:
    """
    Lit TOUS les fichiers .pdf du dossier (récursif).
    Retourne une liste de phrases filtrées.
    """
    sentences = []
    if not root.exists():
        return sentences
    
    try:
        import pdfplumber
    except ImportError:
        print(f"   ⚠️  pdfplumber non installé, PDF ignorés")
        return sentences
    
    pdf_files = list(root.rglob("*.pdf"))
    
    if not pdf_files:
        return sentences
    
    print(f"   📄 {len(pdf_files)} PDF(s) trouvé(s)")
    
    for fpath in tqdm(pdf_files, desc=f"   Lecture PDF {root.name}"):
        page_count = 0
        pdf_sentences = []
        
        try:
            with pdfplumber.open(fpath) as pdf:
                total_pages = min(len(pdf.pages), 100)  # Limiter à 100 pages max
                
                for page_num, page in enumerate(pdf.pages[:100]):
                    try:
                        text = page.extract_text()
                        if text:
                            page_count += 1
                            for s in sent_split(text):
                                if looks_french(s):
                                    pdf_sentences.append(s.strip())
                    except Exception as e_page:
                        # Erreur sur une page spécifique, continuer avec les autres
                        continue
                
                sentences.extend(pdf_sentences)
                
        except Exception as e:
            print(f"\n   ⚠️  PDF {fpath.name}: {str(e)[:100]} - ignoré")
            continue
    
    return sentences

# ============================================================================
# ROBOTS.TXT CHECK (OPTIONNEL)
# ============================================================================

def check_robots_txt(base_url: str, user_agent: str = USER_AGENT) -> bool:
    """
    Vérifie si robots.txt autorise le scraping (basique).
    Retourne True si autorisé ou absent.
    """
    try:
        robots_url = f"{base_url}/robots.txt"
        resp = requests.get(robots_url, timeout=5, headers={"User-Agent": user_agent})
        if resp.status_code == 404:
            return True  # Pas de robots.txt = autorisé
        
        # Parsing basique (simplifié)
        if "Disallow: /" in resp.text:
            return False
        return True
    except:
        return True  # En cas d'erreur, on considère autorisé

# ============================================================================
# SCRAPING URL (AVEC RESPECT ROBOTS.TXT)
# ============================================================================

def scrape_url(url: str, sleep_time: float = 0.5) -> List[str]:
    """
    Récupère le contenu HTML d'une URL, extrait le texte et découpe en phrases.
    Respecte robots.txt et ajoute un délai.
    """
    sentences = []
    
    try:
        # Vérifier robots.txt
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        if not check_robots_txt(base_url):
            print(f"   ⚠️  Robots.txt interdit {url}, ignoré")
            return sentences
        
        # Requête HTTP
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        
        # Parsing HTML
        soup = BeautifulSoup(resp.content, "lxml")
        
        # Retirer scripts, styles, etc.
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        # Extraire texte
        text = soup.get_text(separator=" ", strip=True)
        
        # Découper en phrases
        for s in sent_split(text):
            if looks_french(s):
                sentences.append(s.strip())
        
        # Délai anti-DoS
        time.sleep(sleep_time)
    
    except Exception as e:
        print(f"   ❌ Erreur URL {url}: {e}")
    
    return sentences

# ============================================================================
# TÉLÉCHARGEMENT CORPUS PRÉ-EXISTANTS
# ============================================================================

def download_corpus(corpus_info: dict) -> List[str]:
    """
    Télécharge et parse un corpus pré-existant.
    Retourne une liste de phrases.
    """
    sentences = []
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(corpus_info["url"], timeout=30, headers=headers)
        
        if resp.status_code != 200:
            print(f"   ⚠️  Corpus {corpus_info['name']} inaccessible (code {resp.status_code})")
            return sentences
        
        # Décompression si .gz
        content = resp.content
        if corpus_info["url"].endswith(".gz"):
            import gzip
            try:
                content = gzip.decompress(content)
            except:
                pass
        
        # Parsing selon le type de contenu
        content_type = resp.headers.get("Content-Type", "")
        
        if "json" in content_type or corpus_info["url"].endswith((".json", ".jsonl")):
            # Corpus JSON/JSONL (ex: HuggingFace datasets)
            import json
            try:
                text_content = content.decode("utf-8", errors="ignore")
                
                # Gérer JSONL (une ligne = un JSON)
                if corpus_info["url"].endswith(".jsonl"):
                    for line in text_content.split("\n")[:5000]:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                if isinstance(item, dict):
                                    for key in ["text", "question", "answer", "content", "sentence", "context", "review", "summary"]:
                                        if key in item and item[key]:
                                            text = item[key]
                                            for s in sent_split(str(text)):
                                                if looks_french(s):
                                                    sentences.append(s.strip())
                            except:
                                continue
                else:
                    # JSON classique
                    data = json.loads(text_content)
                    # Extraire textes selon structure
                    if isinstance(data, list):
                        for item in data[:5000]:  # Augmenté à 5000 items
                            if isinstance(item, dict):
                                # Chercher champs textuels
                                for key in ["text", "question", "answer", "content", "sentence", "context", "review", "summary", "body"]:
                                    if key in item and item[key]:
                                        text = item[key]
                                        for s in sent_split(str(text)):
                                            if looks_french(s):
                                                sentences.append(s.strip())
                    elif isinstance(data, dict):
                        # Naviguer dans la structure
                        for value in list(data.values())[:5000]:
                            if isinstance(value, str):
                                for s in sent_split(value):
                                    if looks_french(s):
                                        sentences.append(s.strip())
                            elif isinstance(value, list):
                                for item in value[:5000]:
                                    if isinstance(item, str):
                                        for s in sent_split(item):
                                            if looks_french(s):
                                                sentences.append(s.strip())
                                    elif isinstance(item, dict):
                                        for key in ["text", "question", "answer", "content", "sentence", "context"]:
                                            if key in item and item[key]:
                                                text = item[key]
                                                for s in sent_split(str(text)):
                                                    if looks_french(s):
                                                        sentences.append(s.strip())
            except json.JSONDecodeError:
                pass
        
        else:
            # Corpus HTML ou texte brut
            try:
                text_content = content.decode("utf-8", errors="ignore")
                # Si c'est du texte brut ligne par ligne
                if not text_content.strip().startswith("<"):
                    for line in text_content.split("\n")[:5000]:  # Max 5000 lignes
                        for s in sent_split(line):
                            if looks_french(s):
                                sentences.append(s.strip())
                else:
                    # Parse HTML
                    soup = BeautifulSoup(content, "lxml")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                    for s in sent_split(text):
                        if looks_french(s):
                            sentences.append(s.strip())
            except:
                pass
        
        time.sleep(1.0)  # Délai respectueux
    
    except Exception as e:
        print(f"   ❌ Erreur téléchargement {corpus_info['name']}: {e}")
    
    return sentences

# ============================================================================
# CRÉATION SOURCES MÉDICO-SOCIALES (AUTO-GÉNÉRATION)
# ============================================================================

def create_medsoc_sources(admin_dir: Path, narr_dir: Path, force: bool = False) -> Tuple[Path, Path]:
    """
    Crée automatiquement les sources admin et narratives médico-sociales
    si elles n'existent pas ou sont vides.
    
    SIMPLIFIÉ : Scraping web uniquement, pas de téléchargement HuggingFace.
    Les PDFs locaux seront traités dans data_local/admin et data_local/narratif.
    
    Retourne: (admin_dir, narr_dir) créés
    """
    admin_dir.mkdir(parents=True, exist_ok=True)
    narr_dir.mkdir(parents=True, exist_ok=True)
    
    # Scraping URLs si dossiers vides (comportement existant)
    admin_scraped = admin_dir / "medsoc_admin_scraped.txt"
    narr_scraped = narr_dir / "medsoc_narr_scraped.txt"
    
    if not admin_scraped.exists() or force:
        print(f"🏥 Scraping sources ADMIN web...")
        admin_sentences = []
        for url in tqdm(ADMIN_URLS, desc="   URLs ADMIN"):
            admin_sentences.extend(scrape_url(url, sleep_time=1.0))
        
        if admin_sentences:
            with open(admin_scraped, "w", encoding="utf-8") as f:
                for s in admin_sentences:
                    f.write(s + "\n")
            print(f"   ✅ {len(admin_sentences):,} phrases ADMIN web → {admin_scraped}\n")
    else:
        print(f"   ℹ️  Sources ADMIN web déjà scrapées\n")
    
    if not narr_scraped.exists() or force:
        print(f"📖 Scraping sources NARR web...")
        narr_sentences = []
        for url in tqdm(NARR_URLS, desc="   URLs NARR"):
            narr_sentences.extend(scrape_url(url, sleep_time=1.0))
        
        if narr_sentences:
            with open(narr_scraped, "w", encoding="utf-8") as f:
                for s in narr_sentences:
                    f.write(s + "\n")
            print(f"   ✅ {len(narr_sentences):,} phrases NARR web → {narr_scraped}\n")
    else:
        print(f"   ℹ️  Sources NARR web déjà scrapées\n")
    
    return admin_dir, narr_dir

# ============================================================================
# WIKIPEDIA RANDOM PAGES (CORRIGÉ)
# ============================================================================

def fetch_wiki_random_pages(
    max_pages: int = 1000,
    batch: int = 50,
    sleep_time: float = 0.5,
    lang: str = "fr"
) -> List[str]:
    """
    Récupère des pages Wikipedia aléatoires via l'API.
    Retourne une liste de phrases françaises filtrées.
    
    Paramètres optimisés:
    - batch=50 (vs 10) : meilleure efficacité API
    - sleep_time=0.5s (vs 0.2s) : rate limiting sécurisé
    - Utilise un User-Agent personnalisé pour éviter 403
    """
    print(f"📚 Récupération Wikipedia ({max_pages} pages max, batch={batch})...")
    
    sentences = []
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    
    num_batches = (max_pages + batch - 1) // batch
    
    # Headers avec User-Agent personnalisé
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "fr-FR,fr;q=0.9"
    }
    
    for i in tqdm(range(num_batches), desc="   Pages Wiki"):
        try:
            params = {
                "action": "query",
                "format": "json",
                "generator": "random",
                "grnnamespace": 0,  # Articles uniquement
                "grnlimit": min(batch, 10),  # Wikipedia limite à 10 max pour random
                "prop": "extracts",
                "exintro": False,
                "explaintext": True
            }
            
            resp = requests.get(api_url, params=params, headers=headers, timeout=15)
            
            # Gestion erreur 403 : fallback sur méthode alternative
            if resp.status_code == 403:
                print(f"   ⚠️  API Wikipedia bloquée (403), passage à méthode alternative...")
                break
            
            resp.raise_for_status()
            data = resp.json()
            
            if "query" in data and "pages" in data["query"]:
                for page_id, page_data in data["query"]["pages"].items():
                    if "extract" in page_data:
                        text = page_data["extract"]
                        for s in sent_split(text):
                            if looks_french(s):
                                sentences.append(s.strip())
            
            time.sleep(sleep_time)
        
        except Exception as e:
            print(f"   ❌ Erreur batch {i}: {e}")
            continue
    
    # Fallback : scraper catégories Wikipedia Santé si API bloquée
    if len(sentences) < 100:
        print(f"   🔄 Fallback: scraping catégories Wikipedia Santé...")
        wiki_cats = [
            "https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Sant%C3%A9",
            "https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:M%C3%A9decine",
            "https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Action_sociale",
            "https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Handicap",
        ]
        for cat_url in wiki_cats[:2]:  # Limiter à 2 catégories
            sentences.extend(scrape_url(cat_url, sleep_time=1.0))
    
    print(f"   ✅ {len(sentences):,} phrases extraites de Wikipedia\n")
    return sentences

# ============================================================================
# PARSING QUOTA
# ============================================================================

def parse_quota(quota_str: str) -> Dict[str, float]:
    """
    Parse une chaîne quota type "wiki=0.35,admin=0.45,narr=0.20,other=0.00"
    Normalise automatiquement pour que la somme = 1.0
    """
    parts = [p.strip() for p in quota_str.split(",")]
    quotas = {}
    
    for part in parts:
        if "=" not in part:
            continue
        key, val = part.split("=")
        quotas[key.strip()] = float(val.strip())
    
    # Normaliser pour somme = 1.0
    total = sum(quotas.values())
    if total > 0:
        quotas = {k: v / total for k, v in quotas.items()}
    
    return quotas

# ============================================================================
# SAMPLING HELPER
# ============================================================================

def take_up_to(sentences: List[str], n: int) -> List[str]:
    """Mélange et prend jusqu'à n phrases."""
    random.shuffle(sentences)
    return sentences[:n]

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Construction corpus français stratifié pour distillation NER"
    )
    
    # Taille cible
    ap.add_argument("--target-size", type=int, default=100000,
                    help="Nombre de phrases cibles (stratifié, avant injection locale)")
    
    # Sources Wikipedia
    ap.add_argument("--wiki-max-pages", type=int, default=6000,
                    help="Nombre max de pages Wikipedia (batch=50, sleep=0.5s)")
    
    # Sources locales (NOUVEAU)
    ap.add_argument("--local-dir", type=Path, default=Path("data_local"),
                    help="Dossier contenant TOUS vos fichiers locaux (data_test, input, etc.)")
    
    # Sources admin/narr (optionnelles)
    ap.add_argument("--admin-dir", type=Path, default=None,
                    help="Dossier sources administratives (optionnel)")
    ap.add_argument("--narr-dir", type=Path, default=None,
                    help="Dossier sources narratifs (optionnel)")
    
    # Quotas stratifiés (RECOMMANDATION APPLIQUÉE)
    ap.add_argument("--quota", type=str, default="wiki=0.35,admin=0.45,narr=0.20,other=0.00",
                    help="Quotas stratifiés (admin=45% pour biais médico-social)")
    
    # Fichier URLs (optionnel)
    ap.add_argument("--urls-file", type=Path, default=None,
                    help="Fichier texte avec URLs à scraper (1/ligne, optionnel)")
    
    # Sortie
    ap.add_argument("--out", type=Path, default=Path("corpus/corpus_fr_100k_medico.txt"),
                    help="Fichier de sortie (1 phrase/ligne)")
    
    # Seed reproductibilité
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed pour reproductibilité")
    
    args = ap.parse_args()
    
    # Seed global
    random.seed(args.seed)
    
    # Parse quotas
    quotas = parse_quota(args.quota)
    
    print("=" * 70)
    print("🏗️  CONSTRUCTION CORPUS STRATIFIÉ FRANÇAIS")
    print("=" * 70)
    print(f"Cible         : {args.target_size:,} phrases (stratifié)")
    print(f"Wikipedia     : {args.wiki_max_pages:,} pages max")
    print(f"Local         : {args.local_dir}")
    print(f"Quotas        : {quotas}")
    print(f"Seed          : {args.seed}")
    print(f"Sortie        : {args.out}")
    print("=" * 70)
    print()
    
    # ========================================================================
    # PHASE 0: AUTO-CRÉATION SOURCES MÉDICO-SOCIALES (SI ABSENTES)
    # ========================================================================
    
    # Définir les dossiers par défaut si non spécifiés
    if args.admin_dir is None:
        args.admin_dir = Path("sources_admin")
    if args.narr_dir is None:
        args.narr_dir = Path("sources_narr")
    
    # Créer automatiquement les sources si absentes/vides
    args.admin_dir, args.narr_dir = create_medsoc_sources(
        args.admin_dir, 
        args.narr_dir, 
        force=False  # Mettre True pour forcer la régénération
    )
    
    # ========================================================================
    # PHASE 1: COLLECTE SOURCES STRATIFIÉES
    # ========================================================================
    
    buckets = defaultdict(list)
    
    # 1.1 Wikipedia
    if quotas.get("wiki", 0) > 0:
        wiki_sentences = fetch_wiki_random_pages(
            max_pages=args.wiki_max_pages,
            batch=50,
            sleep_time=0.5
        )
        buckets["wiki"] = wiki_sentences
    
    # 1.2 Admin (optionnel)
    if args.admin_dir and quotas.get("admin", 0) > 0:
        print(f"📁 Lecture sources ADMIN depuis {args.admin_dir}...")
        admin_sentences = read_txt_dir(args.admin_dir)
        
        # Lire aussi les PDFs dans sources_admin
        pdf_sentences = read_pdf_dir(args.admin_dir)
        if pdf_sentences:
            print(f"   📄 {len(pdf_sentences):,} phrases extraites des PDFs")
            admin_sentences.extend(pdf_sentences)
        
        # BONUS: Lire aussi data_local/admin si présent (rapports PDF)
        admin_local = args.local_dir / "admin"
        if admin_local.exists():
            print(f"📑 Lecture rapports depuis {admin_local}...")
            rapports_txt = read_txt_dir(admin_local)
            rapports_pdf = read_pdf_dir(admin_local)
            if rapports_pdf:
                print(f"   📄 {len(rapports_pdf):,} phrases extraites des rapports PDF")
            admin_sentences.extend(rapports_txt)
            admin_sentences.extend(rapports_pdf)
        
        buckets["admin"] = admin_sentences
        print(f"   ✅ {len(admin_sentences):,} phrases ADMIN TOTAL (TXT+PDF+Rapports)\n")
    
    # 1.3 Narratifs (optionnel)
    if args.narr_dir and quotas.get("narr", 0) > 0:
        print(f"📖 Lecture sources NARR depuis {args.narr_dir}...")
        narr_sentences = read_txt_dir(args.narr_dir)
        
        # Lire aussi les PDFs dans sources_narr
        pdf_sentences = read_pdf_dir(args.narr_dir)
        if pdf_sentences:
            print(f"   📄 {len(pdf_sentences):,} phrases extraites des PDFs NARR")
            narr_sentences.extend(pdf_sentences)
        
        # BONUS: Lire aussi data_local/narratif si présent (romans)
        narratif_local = args.local_dir / "narratif"
        if narratif_local.exists():
            print(f"📚 Lecture romans depuis {narratif_local}...")
            romans_txt = read_txt_dir(narratif_local)
            romans_pdf = read_pdf_dir(narratif_local)
            if romans_pdf:
                print(f"   📄 {len(romans_pdf):,} phrases extraites des romans PDF")
            narr_sentences.extend(romans_txt)
            narr_sentences.extend(romans_pdf)
        
        buckets["narr"] = narr_sentences
        print(f"   ✅ {len(narr_sentences):,} phrases NARR TOTAL (TXT+PDF+Romans)\n")
    
    # 1.4 URLs (optionnel)
    if args.urls_file and args.urls_file.exists() and quotas.get("other", 0) > 0:
        print(f"🌐 Scraping URLs depuis {args.urls_file}...")
        urls = args.urls_file.read_text(encoding="utf-8").strip().split("\n")
        other_sentences = []
        
        for url in tqdm(urls, desc="   URLs"):
            other_sentences.extend(scrape_url(url.strip(), sleep_time=0.5))
        
        buckets["other"] = other_sentences
        print(f"   ✅ {len(other_sentences):,} phrases URLs\n")
    
    # ========================================================================
    # PHASE 2: ÉCHANTILLONNAGE STRATIFIÉ
    # ========================================================================
    
    print("🎯 Échantillonnage stratifié...")
    samples = defaultdict(list)
    
    for bucket_name, quota in quotas.items():
        if quota <= 0 or bucket_name not in buckets:
            continue
        
        target_count = int(args.target_size * quota)
        available = buckets[bucket_name]
        
        if len(available) == 0:
            print(f"   ⚠️  {bucket_name.upper()}: quota {quota:.1%} mais aucune phrase disponible")
            continue
        
        sampled = take_up_to(available, target_count)
        samples[bucket_name] = sampled
        
        print(f"   {bucket_name.upper():6s}: {len(sampled):6,} / {target_count:6,} (quota {quota:.1%})")
    
    # Fusion stratifiée
    merged = []
    for bucket_name in sorted(samples.keys()):
        merged.extend(samples[bucket_name])
    
    random.shuffle(merged)
    print(f"\n   ✅ Corpus stratifié: {len(merged):,} phrases\n")
    
    # ========================================================================
    # PHASE 3: INJECTION PHRASES LOCALES (NOUVEAU)
    # ========================================================================
    
    local_sentences = []
    
    if args.local_dir.exists():
        print(f"💉 Injection phrases LOCAL depuis {args.local_dir}...")
        
        # Déduplication basée sur normalisation lowercase
        norm_seen = {s.lower() for s in merged}
        
        # Lire UNIQUEMENT data_test et input (pas admin/narratif, déjà lus en Phase 1)
        local_candidates = []
        
        for subdir in ["data_test", "input"]:
            subdir_path = args.local_dir / subdir
            if subdir_path.exists():
                print(f"   📂 Lecture {subdir}...")
                local_candidates.extend(read_txt_dir(subdir_path))
                pdf_sub = read_pdf_dir(subdir_path)
                if pdf_sub:
                    print(f"      📄 {len(pdf_sub):,} phrases PDF")
                    local_candidates.extend(pdf_sub)
        
        for s in local_candidates:
            key = s.lower()
            if key not in norm_seen:
                norm_seen.add(key)
                local_sentences.append(s)
        
        # Injection aléatoire dans le corpus
        print(f"   📌 Injection aléatoire de {len(local_sentences):,} phrases locales (data_test + input)...")
        for s in tqdm(local_sentences, desc="   Insertion"):
            # Insertion à position aléatoire
            pos = random.randint(0, len(merged))
            merged.insert(pos, s)
        
        print(f"   ✅ {len(local_sentences):,} phrases locales injectées\n")
    else:
        print(f"⚠️  Dossier local {args.local_dir} introuvable, injection ignorée\n")
    
    # ========================================================================
    # PHASE 4: ÉCRITURE CORPUS FINAL
    # ========================================================================
    
    print("💾 Écriture corpus final...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.out, "w", encoding="utf-8") as f:
        for s in merged:
            f.write(s + "\n")
    
    print(f"   ✅ Corpus écrit: {args.out}\n")
    
    # ========================================================================
    # PHASE 5: STATISTIQUES FINALES
    # ========================================================================
    
    print("=" * 70)
    print("📊 STATISTIQUES FINALES")
    print("=" * 70)
    print(f"Total phrases (final)         : {len(merged):,}")
    print(f"Phrases locales injectées     : {len(local_sentences):,}")
    print(f"Corpus stratifié (avant inj.) : {len(merged) - len(local_sentences):,}")
    print()
    print("Répartition stratifiée (avant injection locale):")
    for bucket_name in sorted(samples.keys()):
        count = len(samples[bucket_name])
        pct = count / max(len(merged) - len(local_sentences), 1) * 100
        print(f"  {bucket_name.upper():6s}: {count:6,} phrases ({pct:5.1f}%)")
    print()
    print(f"Fichier sortie : {args.out}")
    print(f"Taille fichier : {args.out.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 70)
    print("✅ CORPUS CONSTRUIT AVEC SUCCÈS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
