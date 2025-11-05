#!/usr/bin/env python3
"""
filter_feed.py - final version

Features:
- Pulls remote titles (public) from REMOTE_TITLES_URL (append-only expected)
- Maintains reference_titles.txt using last-seen incremental append
- Keeps meta in ref_titles_cache.json (title, added_at ISO UTC, match_count)
- Writes reference_titles.txt after every meta update (so your workflow can commit it)
- Handles timezone-aware vs naive datetimes safely
- Uses local SentenceTransformer model at models/all-mpnet-base-v2
- Hybrid filtering (pattern score + semantic similarity)
- DBSCAN clustering -> hot-topic detection
- Diversity pass and compact filtered.xml output
"""

import os
import re
import json
import requests
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG (keep these filenames unchanged) =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
REF_META_JSON = "ref_titles_cache.json"      # meta: list of {"title","added_at","match_count"}
REF_EMB_NPY = "ref_embeddings.npy"           # optional embedding cache
OUTPUT_FILE = "filtered.xml"

REMOTE_TITLES_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"

MODEL_PATH = "models/all-mpnet-base-v2"      # workflow downloads this path
USE_SMALL_MODEL = False                      # toggle if you change model

# Filtering & clustering thresholds
ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_MED = 7
HYBRID_PATTERN_HIGH = 8
REF_MATCH_SIM = 0.65  # used to increment match_count in meta

MAX_REF_TITLES = 2000
MAX_REF_DAYS = 90

CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 75

DBSCAN_EPS = 0.22
DBSCAN_MIN_SAMPLES = 1

# Source authority weights (optional)
SOURCE_WEIGHTS = {
    "reuters.com": 2.0,
    "nytimes.com": 2.0,
    "washingtonpost.com": 1.8,
    "cnn.com": 1.6,
    "aljazeera.com": 1.6,
    "bbc.co.uk": 1.8,
}

# ===== HELPERS =====
def clean_title(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r'["“”‘’\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def parse_iso_to_utc_naive(s: str):
    """Parse an ISO or fuzzy datetime string, return naive UTC datetime (tz info removed)."""
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        # convert to UTC and drop tzinfo (naive)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            # consider it naive local -> treat as UTC naive
            # We standardize: interpret naive as UTC (consistent with now() in UTC)
            pass
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
    domain = domain_from_url(url)
    for d, w in SOURCE_WEIGHTS.items():
        if d in domain:
            return w
    return 1.0

# ===== REMOTE PULL + LAST-SEEN META MANAGEMENT =====
def read_remote_titles(raw_url: str, timeout: int = 15):
    r = requests.get(raw_url, timeout=timeout)
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
    # also write plain reference_titles.txt so other parts (and commit) see it
    write_reference_txt(meta)

def write_reference_txt(meta):
    """Write the reference_titles.txt from meta (one title per line)."""
    try:
        with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(m["title"] + "\n")
    except Exception as e:
        print("[write_reference_txt] error:", e)

def prune_ref_meta(meta):
    """Eviction: 1) remove entries older than MAX_REF_DAYS; 2) if still > MAX_REF_TITLES keep top by (match_count + recency)."""
    if not meta:
        return meta
    # time cutoff (UTC naive)
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=MAX_REF_DAYS)
    kept = []
    for m in meta:
        added = None
        try:
            added = parse_iso_to_utc_naive(m.get("added_at", ""))
        except Exception:
            added = None
        if added is None:
            # keep if cannot parse, but set added_at to now to avoid accidental deletion
            m["added_at"] = now_iso_utc()
            kept.append(m)
        else:
            if added >= cutoff:
                kept.append(m)
    meta = kept
    if len(meta) > MAX_REF_TITLES:
        # score = match_count + recency_boost (recent -> larger)
        now_ts = datetime.now(timezone.utc).timestamp()
        scored = []
        for m in meta:
            try:
                added_ts = parse_iso_to_utc_naive(m.get("added_at")).timestamp()
            except Exception:
                added_ts = now_ts
            age_days = max(0.0, (now_ts - added_ts) / 86400.0)
            recency_boost = max(0.0, 1.0 - (age_days / 365.0))  # decays over a year
            score = m.get("match_count", 0) + recency_boost
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        meta = [m for m, _ in scored[:MAX_REF_TITLES]]
    return meta

def incremental_pull_and_append():
    """
    Pull remote titles and update local ref meta and reference_titles.txt.
    Logic:
     - If local meta exists and is a prefix of remote -> append only new titles.
     - If no local meta -> bootstrap from tail of remote up to MAX_REF_TITLES.
     - Otherwise fallback: rebuild from tail of remote (safe).
    """
    print("[refs] fetching remote titles...")
    remote = read_remote_titles(REMOTE_TITLES_URL)
    meta = load_ref_meta()
    cached_titles = [m["title"] for m in meta] if meta else []

    # fast path - cached is exact prefix of remote (append-only)
    if cached_titles and len(remote) >= len(cached_titles) and remote[:len(cached_titles)] == cached_titles:
        new = remote[len(cached_titles):]
        if not new:
            print("[refs] no new remote titles to append")
            # ensure reference_titles.txt exists and matches meta
            write_reference_txt(meta)
            return [m["title"] for m in meta]
        print(f"[refs] appending {len(new)} new titles")
        new_meta = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in new]
        meta.extend(new_meta)
        # prune then save (save writes reference_titles.txt)
        meta = prune_ref_meta(meta)
        save_ref_meta(meta)
        return [m["title"] for m in meta]

    # bootstrap if empty
    if not cached_titles:
        print("[refs] no cache present - bootstrapping from remote tail")
        boot = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
        meta = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in boot]
        meta = prune_ref_meta(meta)
        save_ref_meta(meta)
        return [m["title"] for m in meta]

    # Fall-back: remote changed non-trivially -> rebuild from remote tail
    print("[refs] remote changed - rebuilding local references from remote tail")
    rebuild = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
    meta = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in rebuild]
    meta = prune_ref_meta(meta)
    save_ref_meta(meta)
    return [m["title"] for m in meta]

# ===== PATTERN SCORE (same scoring logic) =====
def calculate_analytical_score(title: str) -> int:
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
    score += min(sum(1 for term in crisis_terms if term in tl) * 2, 6)

    if any(p in tl for p in ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']):
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
    score += min(sum(1 for term in geo_terms if term in tl), 3)

    outcome_terms = [
        'faces','threatens','undermines','deepens','escalates','intensifies',
        'worsens','persists','continues','remains','struggles','fails',
        'succeeds','achieves','drives','fuels','prevents','blocks','enables',
        'forces','triggers','sparks','creates','causes','leads to','results in',
        'complicates','reshapes','transforms','revives','shifts','redefines',
        'contributes to','signals','marks','heightens','deteriorates'
    ]
    if any(term in tl for term in outcome_terms):
        score += 1

    explanatory = [
        'reveals','shows','exposes','means for','means to','impact on','impact of',
        'effect on','effect of','implications','consequences','amid','despite',
        'after','before','during','following','since','until','significance','context','analysis','explained','lessons from'
    ]
    if any(exp in tl for exp in explanatory):
        score += 1

    comparative = ['between','versus','against','compared to','the paradox','the dilemma','the challenge','the struggle','crossroads','balance of power']
    if any(comp in tl for comp in comparative):
        score += 1

    future = ['future of','will ','could ','may ','might ','outlook','prospects','next phase','ahead','forecast','trajectory']
    if any(fut in tl for fut in future):
        score += 1

    regions = ['middle east','south asia','east asia','southeast asia','central asia','africa','europe','balkans','sahel','caucasus','latin america','gaza','israel','palestine','ukraine','russia','china','india','pakistan','iran','saudi','turkey','egypt','syria','yemen','afghanistan','bangladesh','myanmar','taiwan','korea','japan','ethiopia','sudan','nigeria','kenya','somalia','mali','libya','venezuela','colombia','brazil','mexico','indonesia','philippines','thailand','vietnam','azerbaijan','armenia','georgia','kosovo','usa']
    if any(region in tl for region in regions):
        score += 1

    news_indicators = ['announces','launches','opens','celebrates','wins award','ceremony','appointed today','signs deal today','breaks record','new product','birthday','anniversary celebration','gala','festival begins','sports','match','tournament','cup','music','film','concert','startup','company','brand','fashion','award show']
    if any(ind in tl for ind in news_indicators):
        score -= 3

    return score

# ===== MAIN PROCESS =====
def main():
    print("[main] start")

    # 1) update references (pull + append + write reference_titles.txt)
    try:
        ref_titles = incremental_pull_and_append()
    except Exception as e:
        print("[main] error updating references:", e)
        ref_titles = []
    print(f"[main] reference titles count: {len(ref_titles)}")

    # 2) load model
    if USE_SMALL_MODEL:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        # using local path; workflow should ensure downloaded model at MODEL_PATH
        model_name = MODEL_PATH

    if not os.path.exists(model_name) and model_name == MODEL_PATH:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Ensure workflow downloaded it.")

    print("[main] loading model...")
    model = SentenceTransformer(model_name)

    # 3) compute (or load) reference embeddings
    ref_embeddings = None
    try:
        if ref_titles:
            # Use saved embeddings if shape matches titles length
            if os.path.exists(REF_EMB_NPY):
                try:
                    emb_cache = np.load(REF_EMB_NPY)
                    if emb_cache.shape[0] == len(ref_titles):
                        ref_embeddings = emb_cache
                        print("[main] loaded cached ref embeddings")
                    else:
                        print("[main] ref_embeddings.npy length mismatch; recomputing embeddings")
                        ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                        np.save(REF_EMB_NPY, ref_embeddings)
                except Exception:
                    print("[main] failed to load ref embeddings; computing fresh")
                    ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                    np.save(REF_EMB_NPY, ref_embeddings)
            else:
                print("[main] computing ref embeddings...")
                ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                np.save(REF_EMB_NPY, ref_embeddings)
    except Exception as e:
        print("[main] reference embeddings error:", e)
        ref_embeddings = None

    # 4) load feeds
    if not os.path.exists(FEEDS_FILE):
        raise SystemExit("[main] feeds.txt not found")
    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        feed_urls = [l.strip() for l in f if l.strip()]

    print(f"[main] parsing {len(feed_urls)} feeds...")
    feed_articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                if not getattr(e, "title", None) or not getattr(e, "link", None):
                    continue
                feed_articles.append({
                    "title": clean_title(getattr(e, "title", "")),
                    "link": e.link,
                    "published": getattr(e, "published", "") or getattr(e, "updated", ""),
                    "feed_source": url
                })
        except Exception as ex:
            print("[main] feed parse error:", url, ex)

    if not feed_articles:
        print("[main] no feed articles found; exiting")
        return

    # 5) encode feed titles
    titles = [a["title"] for a in feed_articles]
    print(f"[main] encoding {len(titles)} article titles...")
    article_emb = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)

    # 6) hybrid filter
    print("[main] applying hybrid filter...")
    candidates = []
    ref_meta = load_ref_meta()  # to increment match_count
    for idx, a in enumerate(feed_articles):
        t = a["title"]
        pat = calculate_analytical_score(t)
        max_sim = 0.0
        if ref_embeddings is not None and ref_embeddings.size > 0:
            sims = cosine_similarity([article_emb[idx]], ref_embeddings)[0]
            max_sim = float(np.max(sims))
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

        meta_item = a.copy()
        meta_item.update({
            "pattern_score": pat,
            "sim_to_refs": max_sim,
            "embedding": article_emb[idx],
            "timestamp": timestamp_from_pubdate(a.get("published", "")),
            "source_weight": source_weight_from_url(a.get("link", ""))
        })

        # increment match_count for strongly matching refs (helps eviction)
        if ref_meta and ref_embeddings is not None and ref_embeddings.size > 0:
            sims = cosine_similarity([article_emb[idx]], ref_embeddings)[0]
            for mi, s in enumerate(sims):
                if s >= REF_MATCH_SIM:
                    try:
                        ref_meta[mi]["match_count"] = ref_meta[mi].get("match_count", 0) + 1
                    except Exception:
                        pass

        candidates.append(meta_item)

    # persist updated ref_meta and write reference_titles.txt (so the repo commit sees it)
    try:
        if ref_meta:
            ref_meta = prune_ref_meta(ref_meta)
            save_ref_meta(ref_meta)
    except Exception as e:
        print("[main] error saving ref meta:", e)

    # 7) time cutoff
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)).timestamp())
    candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
    if not candidates:
        print("[main] no candidates after cutoff")
        return

    # 8) clustering
    print("[main] clustering candidates...")
    X = np.vstack([c["embedding"] for c in candidates])
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(X)
    labels = clustering.labels_
    for lbl, c in zip(labels, candidates):
        c["cluster"] = int(lbl)

    clusters = {}
    for c in candidates:
        clusters.setdefault(c["cluster"], []).append(c)

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
        cluster_list.append({"label": lbl, "items": items, "score": cluster_score(items), "size": len(items)})
    cluster_list.sort(key=lambda x: x["score"], reverse=True)

    # 9) region detection (simple keyword)
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

    # 10) diversity pass & select representatives
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
    final = []
    seen_titles = set()
    for _, art, region in selected:
        if art["title"] in seen_titles:
            continue
        final.append(art)
        seen_titles.add(art["title"])
        if len(final) >= MAX_OUTPUT_ITEMS:
            break

    if not final:
        for cl in cluster_list[:MAX_OUTPUT_ITEMS]:
            final.append(cl["items"][0])

    # 11) write XML (merge with existing)
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

    # cap items
    all_items = channel.findall("item")
    if len(all_items) > MAX_OUTPUT_ITEMS:
        for item in all_items[MAX_OUTPUT_ITEMS:]:
            channel.remove(item)

    ET.ElementTree(root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    print(f"[main] wrote {min(MAX_OUTPUT_ITEMS, len(final))} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()