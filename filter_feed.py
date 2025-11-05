#!/usr/bin/env python3
"""
filter_feed.py - simplified version with automatic GitHub titles fetch
"""

import os
import re
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "titles.txt"
REFERENCE_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"
REF_EMB_NPY = "ref_embeddings.npy"
OUTPUT_FILE = "filtered.xml"

MODEL_PATH = "models/all-mpnet-base-v2"
USE_SMALL_MODEL = False

ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_MED = 7
HYBRID_PATTERN_HIGH = 8

CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 75

DBSCAN_EPS = 0.22
DBSCAN_MIN_SAMPLES = 1

SOURCE_WEIGHTS = {
    "reuters.com": 2.0,
    "nytimes.com": 2.0,
    "washingtonpost.com": 1.8,
    "cnn.com": 1.6,
    "aljazeera.com": 1.6,
    "bbc.co.uk": 1.8,
}

# ===== UTIL =====
def clean_title(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r'["""\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_iso_to_utc_naive(s: str):
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None

def timestamp_from_pubdate(s: str) -> int:
    dt = parse_iso_to_utc_naive(s)
    if dt is None:
        return 0
    return int(dt.timestamp())

def domain_from_url(url: str) -> str:
    try:
        m = re.search(r"https?://([^/]+)/?", url)
        if m:
            return m.group(1).lower()
    except Exception:
        pass
    return ""

def source_weight_from_url(url: str) -> float:
    dom = domain_from_url(url)
    for k, v in SOURCE_WEIGHTS.items():
        if k in dom:
            return v
    return 1.0

# ===== FETCH TITLES FROM GITHUB =====
def fetch_reference_titles():
    try:
        print("[fetch_reference_titles] fetching titles.txt from GitHub...")
        r = requests.get(REFERENCE_URL)
        if r.status_code == 200:
            with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("[fetch_reference_titles] ✓ downloaded titles.txt")
        else:
            print(f"[fetch_reference_titles] ERROR: status {r.status_code}")
    except Exception as e:
        print(f"[fetch_reference_titles] ERROR: {e}")

# ===== LOAD REFERENCE TITLES =====
def load_reference_titles():
    if not os.path.exists(REFERENCE_FILE):
        print(f"[load_reference_titles] WARNING: {REFERENCE_FILE} not found")
        return []
    try:
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            titles = [clean_title(line) for line in f if clean_title(line)]
        print(f"[load_reference_titles] ✓ loaded {len(titles)} titles from {REFERENCE_FILE}")
        return titles
    except Exception as e:
        print(f"[load_reference_titles] ERROR: {e}")
        return []

# ===== PATTERN SCORING =====
def calculate_analytical_score(title: str) -> int:
    if not title:
        return 0
    tl = title.lower()
    score = 0

    crisis_terms = [
        'war','conflict','crisis','collapse','violence','attack','strikes','invasion','sanctions',
        'blockade','ceasefire','offensive','bombing','insurgency','coup','protests','unrest','genocide',
        'massacre','refugees','displacement','humanitarian','famine','drought','terror','clashes',
        'shelling','uprising','hostilities','tensions','retaliation','raid','mobilization','escalation',
        'hostage','airstrike','militia','civil war','rebels','armed group'
    ]
    score += min(sum(1 for term in crisis_terms if term in tl) * 2, 6)

    if any(p in tl for p in ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']):
        score += 2
    if "'s " in tl or "'" in tl:
        score += 2

    question_starters = ['why ','how ','what ','can ','will ','should ','is ','are ','do ','does ','could ','may ','might ','would ']
    if any(tl.startswith(q) for q in question_starters):
        score += 2

    geo_terms = [
        'economic','trade','debt','inflation','currency','military','nuclear','election','government','regime',
        'policy','treaty','agreement','dispute','territorial','sovereignty','peacekeeping','intervention','occupation',
        'embargo','tariffs','bilateral','multilateral','alliance','nato','un','security','defense','border','independence',
        'autonomy','separatist','geopolitics','foreign policy','diplomacy','strategy','strategic','realignment','deterrence',
        'containment','bloc','regional order','sanction','power balance','influence','proxy','geostrategic'
    ]
    score += min(sum(1 for term in geo_terms if term in tl), 3)

    outcome_terms = ['faces','threatens','undermines','deepens','escalates','intensifies','worsens','persists','continues','remains','struggles','fails']
    if any(term in tl for term in outcome_terms):
        score += 1

    explanatory = ['reveals','shows','exposes','impact on','impact of','implications','consequences','amid','despite','analysis','explained','lessons from']
    if any(exp in tl for exp in explanatory):
        score += 1

    comparative = ['between','versus','against','compared to','the paradox','the dilemma','the challenge','crossroads']
    if any(comp in tl for comp in comparative):
        score += 1

    future = ['future of','will ','could ','may ','might ','outlook','prospects','next phase','ahead','forecast','trajectory']
    if any(fut in tl for fut in future):
        score += 1

    regions = ['middle east','south asia','east asia','southeast asia','central asia','africa','europe','gaza','israel','palestine','ukraine','russia','china','india','pakistan','iran','turkey','yemen','syria','afghanistan','bangladesh','myanmar','taiwan','korea','japan','ethiopia','sudan','nigeria']
    if any(region in tl for region in regions):
        score += 1

    news_indicators = ['announces','launches','opens','celebrates','wins award','ceremony','appointed today','signs deal today','breaks record','new product','birthday','gala','sports','match','tournament','cup','music','film','startup','company','brand','fashion']
    if any(ind in tl for ind in news_indicators):
        score -= 3

    return score

# ===== MAIN =====
def main():
    print("[main] starting filter_feed.py")

    # Fetch reference titles
    fetch_reference_titles()
    ref_titles = load_reference_titles()

    # Load model
    model_name = "sentence-transformers/all-MiniLM-L6-v2" if USE_SMALL_MODEL else MODEL_PATH
    if not os.path.exists(model_name) and model_name == MODEL_PATH:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = SentenceTransformer(model_name)

    # Compute/load reference embeddings
    ref_embeddings = None
    if ref_titles:
        if os.path.exists(REF_EMB_NPY):
            emb_cache = np.load(REF_EMB_NPY)
            if emb_cache.shape[0] == len(ref_titles):
                ref_embeddings = emb_cache
            else:
                ref_embeddings = model.encode(ref_titles, convert_to_numpy=True)
                np.save(REF_EMB_NPY, ref_embeddings)
        else:
            ref_embeddings = model.encode(ref_titles, convert_to_numpy=True)
            np.save(REF_EMB_NPY, ref_embeddings)

    # Load feeds
    if not os.path.exists(FEEDS_FILE):
        raise SystemExit(f"{FEEDS_FILE} missing")
    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        feed_urls = [l.strip() for l in f if l.strip()]

    feed_articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for e in feed.entries:
            if getattr(e, "title", None) and getattr(e, "link", None):
                feed_articles.append({
                    "title": clean_title(e.title),
                    "link": e.link,
                    "published": getattr(e, "published", "") or getattr(e, "updated", ""),
                    "feed_source": url
                })

    if not feed_articles:
        print("[main] no feed articles found; exiting")
        return

    # Encode articles
    article_emb = model.encode([a["title"] for a in feed_articles], convert_to_numpy=True)

    # Hybrid filter
    candidates = []
    for idx, a in enumerate(feed_articles):
        t = a["title"]
        pat = calculate_analytical_score(t)
        max_sim = 0.0
        if ref_embeddings is not None and ref_embeddings.size > 0:
            sims = cosine_similarity([article_emb[idx]], ref_embeddings)[0]
            max_sim = float(np.max(sims))
        accept = (max_sim >= ENGLISH_SIM_THRESHOLD) or \
                 (pat >= HYBRID_PATTERN_MED and max_sim >= 0.33) or \
                 (pat >= (HYBRID_PATTERN_MED - 2) and max_sim >= 0.38) or \
                 (pat >= HYBRID_PATTERN_HIGH and max_sim >= 0.30)
        if accept:
            meta = a.copy()
            meta.update({
                "pattern_score": pat,
                "sim_to_refs": max_sim,
                "embedding": article_emb[idx],
                "timestamp": timestamp_from_pubdate(a.get("published","")),
                "source_weight": source_weight_from_url(a.get("link",""))
            })
            candidates.append(meta)

    if not candidates:
        print("[main] no candidates passed filter; exiting")
        return

    # ===== TIME CUTOFF =====
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)).timestamp())
    candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
    print(f"[main] {len(candidates)} candidates after {CUTOFF_HOURS}h cutoff")

    if not candidates:
        print("[main] no recent candidates; exiting")
        return

    # ===== CLUSTERING (DBSCAN) =====
    X = np.vstack([c["embedding"] for c in candidates])
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(X)
    labels = clustering.labels_

    for lbl, c in zip(labels, candidates):
        c["cluster"] = int(lbl)

    clusters = {}
    for c in candidates:
        clusters.setdefault(c["cluster"], []).append(c)

    # ===== CLUSTER SCORING =====
    def cluster_score(cluster):
        size = len(cluster)
        avg_pat = sum(x["pattern_score"] for x in cluster) / max(1, size)
        sum_source = sum(x["source_weight"] for x in cluster)
        max_ts = max(x["timestamp"] for x in cluster)
        age_hours = max(1, (datetime.now(timezone.utc).timestamp() - max_ts) / 3600.0)
        recency_boost = 1.0 / (1.0 + age_hours / 24.0)
        return size * (1 + avg_pat / 5.0) * (1 + (sum_source - size) / 5.0) * (1 + recency_boost)

    cluster_list = []
    for lbl, items in clusters.items():
        cluster_list.append({"label": lbl, "items": items, "score": cluster_score(items)})
    cluster_list.sort(key=lambda x: x["score"], reverse=True)

    # ===== REGION DETECTION =====
    REGION_KEYWORDS = {
        "middle east": ["gaza","israel","palestine","yemen","syria","iran","turkey","saudi","egypt"],
        "europe": ["ukraine","russia","uk","france","germany","balkans","georgia","armenia","azerbaijan"],
        "south asia": ["india","pakistan","bangladesh","afghanistan","sri lanka","nepal"],
        "east asia": ["china","taiwan","korea","japan"],
        "africa": ["ethiopia","sudan","nigeria","kenya","somalia","mali","libya"],
        "latin america": ["venezuela","colombia","brazil","mexico"],
    }

    def detect_region(title):
        tl = title.lower()
        for r, kws in REGION_KEYWORDS.items():
            for kw in kws:
                if kw in tl:
                    return r
        return "global"

    # ===== DIVERSITY & FINAL SELECTION =====
    selected = []
    used_regions = set()
    max_consider = min(len(cluster_list), MAX_OUTPUT_ITEMS * 3)

    for cl in cluster_list[:max_consider]:
        items = cl["items"]
        best = None
        best_score = -1

        for it in items:
            recency_hours = max(1, (datetime.now(timezone.utc).timestamp() - it["timestamp"]) / 3600.0)
            recency_score = 1.0 / (1.0 + recency_hours / 24.0)
            item_score = it["pattern_score"] * 2.0 + it["sim_to_refs"] * 5.0 + it["source_weight"] * 1.5 + recency_score
            if item_score > best_score:
                best_score = item_score
                best = it

        if not best:
            continue

        region = detect_region(best["title"])
        priority = cl["score"] * (1.2 if region not in used_regions else 0.8)
        selected.append((priority, best, region))
        used_regions.add(region)

    selected.sort(key=lambda x: x[0], reverse=True)

    # ===== FINALIZE OUTPUT =====
    final = []
    seen_titles = set()
    for _, art, _ in selected:
        if art["title"] in seen_titles:
            continue
        final.append(art)
        seen_titles.add(art["title"])
        if len(final) >= MAX_OUTPUT_ITEMS:
            break

    if not final:
        for cl in cluster_list[:MAX_OUTPUT_ITEMS]:
            final.append(cl["items"][0])

    print(f"[main] selected {len(final)} final articles")

    # ===== WRITE XML =====
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Filtered Feed"
    ET.SubElement(channel, "link").text = "https://example.com"
    ET.SubElement(channel, "description").text = "Filtered feed articles"

    for art in final:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = art["title"]
        ET.SubElement(item, "link").text = art["link"]
        ET.SubElement(item, "pubDate").text = datetime.utcfromtimestamp(art["timestamp"]).strftime("%a, %d %b %Y %H:%M:%S GMT")
        ET.SubElement(item, "source").text = art.get("feed_source", "")

    tree = ET.ElementTree(root)
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    print(f"[main] written {len(final)} articles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()