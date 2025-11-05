#!/usr/bin/env python3
"""
filter_feed.py
- Pulls remote titles (public) from REMOTE_TITLES_URL (append-only expected)
- Maintains reference_titles.txt using last-seen incremental append
- Keeps meta in ref_titles_cache.json (added_at, match_count)
- Uses local model at models/all-mpnet-base-v2 to compute embeddings
- Hybrid filtering (pattern score + semantic similarity)
- Clusters accepted articles, scores clusters, selects diverse top items
- Writes filtered.xml (keeps existing items and prepends new ones)
"""

import os
import re
import json
import requests
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dateutil import parser as dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG (file names unchanged) =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"

REMOTE_TITLES_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"
REF_META_JSON = "ref_titles_cache.json"

MODEL_PATH = "models/all-mpnet-base-v2"  # must exist (workflow downloads it)
USE_SMALL_MODEL = False  # set True to use a smaller HF model if you change code

# thresholds & limits
ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_MED = 7
HYBRID_PATTERN_HIGH = 8

REF_MATCH_SIM = 0.65  # to increment match_count in meta
MAX_REF_TITLES = 2000
MAX_REF_DAYS = 90

CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 75

DBSCAN_EPS = 0.22  # cosine distance threshold
DBSCAN_MIN_SAMPLES = 1

# optional source authority weights
SOURCE_WEIGHTS = {
    "reuters.com": 2.0,
    "nytimes.com": 2.0,
    "washingtonpost.com": 1.8,
    "cnn.com": 1.6,
    "aljazeera.com": 1.6,
    "bbc.co.uk": 1.8,
}

# ===== UTIL =====
def clean_title(t):
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r'["“”‘’\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def md5(s):
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def timestamp_from_pubdate(s):
    try:
        dt = dateparser.parse(s, fuzzy=True)
        if dt.tzinfo:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        return int(dt.timestamp())
    except Exception:
        return 0

def domain_from_url(url):
    try:
        m = re.search(r"https?://([^/]+)/?", url)
        if m:
            return m.group(1).lower()
    except:
        pass
    return ""

def source_weight_from_url(url):
    d = domain_from_url(url)
    for dom, w in SOURCE_WEIGHTS.items():
        if dom in d:
            return w
    return 1.0

# ===== REMOTE TITLE PULL + LAST-SEEN APPEND + META =====
def _read_remote_titles(url, timeout=15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    lines = [clean_title(l) for l in r.text.splitlines() if clean_title(l)]
    return lines

def load_ref_meta():
    if not os.path.exists(REF_META_JSON):
        return []
    try:
        with open(REF_META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_ref_meta(meta):
    with open(REF_META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def prune_ref_meta(meta):
    # 1) time-based prune
    cutoff = datetime.utcnow() - timedelta(days=MAX_REF_DAYS)
    kept = []
    for m in meta:
        try:
            added = dateparser.parse(m.get("added_at"))
            if added.tzinfo is not None:
                added = added.astimezone(tz=None).replace(tzinfo=None)  # convert to naive UTC
        except Exception:
            added = datetime.utcnow()
        if added >= cutoff:
            kept.append(m)
    meta = kept
    # 2) if still too large, keep top by match_count + recency factor
    if len(meta) > MAX_REF_TITLES:
        now_ts = datetime.utcnow().timestamp()
        scored = []
        for m in meta:
            try:
                added_ts = dateparser.parse(m.get("added_at"))
                if added_ts.tzinfo is not None:
                    added_ts = added_ts.astimezone(tz=None).replace(tzinfo=None)
                added_ts = added_ts.timestamp()
            except:
                added_ts = now_ts
            age_days = max(0.0, (now_ts - added_ts) / 86400.0)
            recency_boost = max(0.0, 1.0 - (age_days / 365.0))
            score = m.get("match_count", 0) + recency_boost
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        meta = [m for m, _ in scored[:MAX_REF_TITLES]]
    return meta
        

def incremental_pull_and_append():
    """
    - Pull remote titles (public).
    - If cached meta is prefix -> append only new lines.
    - If no cache -> bootstrap using tail of remote up to MAX_REF_TITLES.
    - On mismatch, rebuild from remote tail (safe).
    - Always maintain/reference REFERENCE_FILE (lines) and REF_META_JSON (meta).
    """
    remote = _read_remote_titles(REMOTE_TITLES_URL)
    meta = load_ref_meta()
    cached_titles = [m["title"] for m in meta] if meta else []

    # fast path: cached is prefix
    if cached_titles and len(remote) >= len(cached_titles) and remote[:len(cached_titles)] == cached_titles:
        new = remote[len(cached_titles):]
        if new:
            new_meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in new]
            meta.extend(new_meta)
            # append to local reference file
            with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
                for t in new:
                    f.write(t + "\n")
        meta = prune_ref_meta(meta)
        save_ref_meta(meta)
        return [m["title"] for m in meta]

    # bootstrap if empty
    if not cached_titles:
        boot = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
        meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in boot]
        with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
            for t in boot:
                f.write(t + "\n")
        meta = prune_ref_meta(meta)
        save_ref_meta(meta)
        return [m["title"] for m in meta]

    # fallback: rebuild from remote tail (safe)
    rebuild = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
    meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in rebuild]
    with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
        for t in rebuild:
            f.write(t + "\n")
    meta = prune_ref_meta(meta)
    save_ref_meta(meta)
    return [m["title"] for m in meta]

# ===== PATTERN SCORE =====
def calculate_analytical_score(title):
    if not title:
        return 0
    tl = title.lower()
    score = 0
    crisis_terms = [
        'war','conflict','crisis','collapse','violence','attack','strikes',
        'invasion','sanctions','blockade','ceasefire','offensive','bombing',
        'insurgency','coup','protests','unrest','genocide','massacre',
        'refugees','displacement','humanitarian','famine','drought','terror',
        'clashes','shelling','uprising','hostilities','tensions','retaliation',
        'raid','mobilization','escalation','hostage','airstrike','militia',
        'civil war','rebels','armed group'
    ]
    crisis_count = sum(1 for term in crisis_terms if term in tl)
    score += min(crisis_count * 2, 6)

    relation_patterns = ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']
    if any(p in tl for p in relation_patterns):
        score += 2

    if "'s " in tl or "'" in tl:
        score += 2

    question_starters = ['why ', 'how ', 'what ', 'can ', 'will ', 'should ', 'is ', 'are ', 'do ', 'does ', 'could ', 'may ', 'might ', 'would ']
    if any(tl.startswith(q) for q in question_starters):
        score += 2

    geo_terms = [
        'economic', 'trade', 'debt', 'inflation', 'currency', 'military', 'nuclear',
        'election', 'government', 'regime', 'policy', 'treaty', 'agreement', 'dispute',
        'territorial', 'sovereignty', 'peacekeeping', 'intervention', 'occupation',
        'embargo', 'tariffs', 'bilateral', 'multilateral', 'alliance', 'nato', 'un',
        'security', 'defense', 'border', 'independence', 'autonomy', 'separatist',
        'geopolitics', 'foreign policy', 'diplomacy', 'strategy', 'strategic',
        'realignment', 'deterrence', 'containment', 'bloc', 'regional order',
        'sanction', 'power balance', 'influence', 'proxy', 'geostrategic',
        'foreign affairs', 'statecraft', 'hegemony', 'multipolar', 'unipolar',
        'arms race', 'militarization', 'deterrent', 'security council'
    ]
    geo_count = sum(1 for term in geo_terms if term in tl)
    score += min(geo_count, 3)

    outcome_terms = [
        'faces', 'threatens', 'undermines', 'deepens', 'escalates', 'intensifies',
        'worsens', 'persists', 'continues', 'remains', 'struggles', 'fails',
        'succeeds', 'achieves', 'drives', 'fuels', 'prevents', 'blocks', 'enables',
        'forces', 'triggers', 'sparks', 'creates', 'causes', 'leads to', 'results in',
        'complicates', 'reshapes', 'transforms', 'revives', 'shifts', 'redefines',
        'contributes to', 'signals', 'marks', 'heightens', 'deteriorates'
    ]
    if any(term in tl for term in outcome_terms):
        score += 1

    explanatory = [
        'reveals', 'shows', 'exposes', 'means for', 'means to', 'impact on', 'impact of',
        'effect on', 'effect of', 'implications', 'consequences', 'amid', 'despite',
        'after', 'before', 'during', 'following', 'since', 'until',
        'significance', 'context', 'analysis', 'explained', 'lessons from'
    ]
    if any(exp in tl for exp in explanatory):
        score += 1

    comparative = [
        'between', 'versus', 'against', 'compared to', 'the paradox',
        'the dilemma', 'the challenge', 'the struggle', 'crossroads', 'balance of power'
    ]
    if any(comp in tl for comp in comparative):
        score += 1

    future = [
        'future of', 'will ', 'could ', 'may ', 'might ', 'outlook', 'prospects',
        'next phase', 'ahead', 'forecast', 'trajectory'
    ]
    if any(fut in tl for fut in future):
        score += 1

    regions = [
        'middle east', 'south asia', 'east asia', 'southeast asia', 'central asia',
        'africa', 'europe', 'balkans', 'sahel', 'caucasus', 'latin america',
        'gaza', 'israel', 'palestine', 'ukraine', 'russia', 'china', 'india',
        'pakistan', 'iran', 'saudi', 'turkey', 'egypt', 'syria', 'yemen',
        'afghanistan', 'bangladesh', 'myanmar', 'taiwan', 'korea', 'japan',
        'ethiopia', 'sudan', 'nigeria', 'kenya', 'somalia', 'mali', 'libya',
        'venezuela', 'colombia', 'brazil', 'mexico', 'indonesia', 'philippines',
        'thailand', 'vietnam', 'azerbaijan', 'armenia', 'georgia', 'kosovo', 'usa'
    ]
    if any(region in tl for region in regions):
        score += 1

    news_indicators = [
        'announces', 'launches', 'opens', 'celebrates', 'wins award', 'ceremony',
        'appointed today', 'signs deal today', 'breaks record', 'new product',
        'birthday', 'anniversary celebration', 'gala', 'festival begins',
        'sports', 'match', 'tournament', 'cup', 'music', 'film', 'concert',
        'startup', 'company', 'brand', 'fashion', 'award show'
    ]
    if any(ind in tl for ind in news_indicators):
        score -= 3

    return score

# ===== BOOTSTRAP / UPDATE REF TITLES & EMBEDDINGS =====
print("[main] updating reference titles from remote...")
REF_TITLES = incremental_pull_and_append()

# load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please ensure the model is downloaded at this path.")
model = SentenceTransformer(MODEL_PATH)

# encode references
if REF_TITLES:
    print(f"[main] encoding {len(REF_TITLES)} reference titles...")
    ref_embeddings = model.encode(REF_TITLES, show_progress_bar=False, convert_to_numpy=True)
else:
    ref_embeddings = None

# ===== LOAD FEEDS =====
if not os.path.exists(FEEDS_FILE):
    print("[main] feeds.txt missing; exiting.")
    raise SystemExit(1)

with open(FEEDS_FILE, "r", encoding="utf-8") as f:
    feed_urls = [l.strip() for l in f if l.strip()]

feed_articles = []
for url in feed_urls:
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if not getattr(entry, "title", None) or not getattr(entry, "link", None):
                continue
            feed_articles.append({
                "title": clean_title(entry.title),
                "link": entry.link,
                "published": getattr(entry, "published", "") or getattr(entry, "updated", ""),
                "feed_source": url
            })
    except Exception as e:
        print("[main] feed parse error:", url, e)

if not feed_articles:
    print("[main] no feed articles found; exiting.")
    raise SystemExit(0)

# ===== ENCODE ARTICLES =====
print(f"[main] encoding {len(feed_articles)} feed titles...")
titles = [a["title"] for a in feed_articles]
article_emb = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)

# ===== HYBRID FILTER =====
print("[main] applying hybrid filter...")
candidates = []
ref_meta = load_ref_meta()

for idx, a in enumerate(feed_articles):
    t = a["title"]
    pat = calculate_analytical_score(t)
    max_sim = 0.0
    best_ref_idx = None
    if ref_embeddings is not None and ref_embeddings.size > 0:
        sims = cosine_similarity([article_emb[idx]], ref_embeddings)[0]
        max_sim = float(np.max(sims))
        best_ref_idx = int(np.argmax(sims))
    accept = False
    if max_sim >= ENGLISH_SIM_THRESHOLD:
        accept = True
    elif pat >= HYBRID_PATTERN_MED and max_sim >= HYBRID_MIN_SIM_LOW:
        accept = True
    elif pat >= (HYBRID_PATTERN_MED - 2) and max_sim >= HYBRID_MIN_SIM_HIGH:
        accept = True
    elif pat >= HYBRID_PATTERN_HIGH and max_sim >= 0.30:
        accept = True

    if not accept:
        continue

    meta = a.copy()
    meta.update({
        "pattern_score": pat,
        "sim_to_refs": max_sim,
        "embedding": article_emb[idx],
        "timestamp": timestamp_from_pubdate(a.get("published", "")),
        "source_weight": source_weight_from_url(a.get("link", ""))
    })
    # increment match_count in ref_meta for matches above REF_MATCH_SIM
    if ref_meta and ref_embeddings is not None and ref_embeddings.size > 0:
        sims = cosine_similarity([article_emb[idx]], ref_embeddings)[0]
        matched_idxs = [i for i, s in enumerate(sims) if s >= REF_MATCH_SIM]
        for mi in matched_idxs:
            try:
                ref_meta[mi]["match_count"] = ref_meta[mi].get("match_count", 0) + 1
            except Exception:
                pass
    candidates.append(meta)

# save updated ref meta (persistence for eviction)
if ref_meta:
    ref_meta = prune_ref_meta(ref_meta)
    save_ref_meta(ref_meta)

# ===== TIME CUTOFF =====
cutoff_ts = int((datetime.utcnow() - timedelta(hours=CUTOFF_HOURS)).timestamp())
candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
if not candidates:
    print("[main] no recent candidates after cutoff.")
    raise SystemExit(0)

# ===== CLUSTERING (DBSCAN) =====
print("[main] clustering candidates...")
X = np.vstack([c["embedding"] for c in candidates])
clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(X)
labels = clustering.labels_
for lbl, c in zip(labels, candidates):
    c["cluster"] = int(lbl)

# group clusters
clusters = {}
for c in candidates:
    clusters.setdefault(c["cluster"], []).append(c)

def cluster_score(cluster):
    size = len(cluster)
    avg_pat = sum(x["pattern_score"] for x in cluster) / max(1, size)
    sum_source = sum(x["source_weight"] for x in cluster)
    max_ts = max(x["timestamp"] for x in cluster)
    age_hours = max(1, (datetime.utcnow().timestamp() - max_ts) / 3600.0)
    recency_boost = 1.0 / (1.0 + age_hours / 24.0)
    return size * (1 + avg_pat / 5.0) * (1 + (sum_source - size) / 5.0) * (1 + recency_boost)

cluster_list = []
for lbl, items in clusters.items():
    cluster_list.append({"label": lbl, "items": items, "score": cluster_score(items), "size": len(items)})
cluster_list.sort(key=lambda x: x["score"], reverse=True)

# ===== REGION DETECTION (simple) =====
REGION_KEYWORDS = {
    "middle east": ["gaza","israel","palestine","saudi","yemen","syria","iran","turkey","egypt"],
    "europe": ["ukraine","russia","uk","france","germany","europe","balkans","georgia","armenia","azerbaijan"],
    "south asia": ["india","pakistan","bangladesh","afghanistan","sri lanka","nepal"],
    "east asia": ["china","taiwan","korea","japan","north korea","south korea"],
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

# ===== DIVERSITY PASS & SELECT REPRESENTATIVES =====
selected = []
used_regions = set()
max_consider = min(len(cluster_list), MAX_OUTPUT_ITEMS * 3)
for cl in cluster_list[:max_consider]:
    items = cl["items"]
    best = None
    best_score = -1
    for it in items:
        recency_hours = max(1, (datetime.utcnow().timestamp() - it["timestamp"]) / 3600.0)
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
final = []
seen_titles = set()
for _, art, region in selected:
    if art["title"] in seen_titles:
        continue
    final.append(art)
    seen_titles.add(art["title"])
    if len(final) >= MAX_OUTPUT_ITEMS:
        break

# fallback
if not final:
    for cl in cluster_list[:MAX_OUTPUT_ITEMS]:
        final.append(cl["items"][0])

# ===== WRITE OUTPUT XML (merge with existing) =====
if os.path.exists(OUTPUT_FILE):
    try:
        tree = ET.parse(OUTPUT_FILE)
        root = tree.getroot()
        channel = root.find("channel")
        if channel is None:
            root = ET.Element("rss", version="2.0")
            channel = ET.SubElement(root, "channel")
    except Exception:
        root = ET.Element("rss", version="2.0")
        channel = ET.SubElement(root, "channel")
else:
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Filtered Feed"
    ET.SubElement(channel, "link").text = "https://yourrepo.github.io/"
    ET.SubElement(channel, "description").text = "Filtered articles"

existing_titles = set()
for item in channel.findall("item"):
    t = item.find("title")
    if t is not None:
        existing_titles.add(t.text)

for a in reversed(final):
    if a["title"] in existing_titles:
        continue
    item = ET.Element("item")
    ET.SubElement(item, "title").text = a["title"]
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a.get("published", "")
    ET.SubElement(item, "source").text = a.get("feed_source", "")
    channel.insert(0, item)
    existing_titles.add(a["title"])

# cap XML items
all_items = channel.findall("item")
if len(all_items) > MAX_OUTPUT_ITEMS:
    for item in all_items[MAX_OUTPUT_ITEMS:]:
        channel.remove(item)

ET.ElementTree(root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
print(f"[main] wrote {min(MAX_OUTPUT_ITEMS, len(final))} items to {OUTPUT_FILE}")