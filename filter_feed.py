#!/usr/bin/env python3
"""
filter_feed.py - final robust version

Features:
- Pull remote titles (public) from REMOTE_TITLES_URL (append-only expected)
- Incremental last-seen append logic (fast path + overlap detection)
- Maintain ref meta in REF_META_JSON with added_at (ISO UTC) and match_count
- Save JSON meta and write reference_titles.txt atomically (backup)
- Prune to MAX_REF_TITLES (time-first, then frequency+recency)
- Use local SentenceTransformer model (models/all-mpnet-base-v2)
- Use ref_embeddings.npy cache if shape matches
- Hybrid filtering: pattern score + semantic similarity
- DBSCAN clustering -> hot-topic selection with diversity pass
"""

import os
import re
import json
import shutil
import tempfile
import requests
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ===== CONFIG (KEEP FILE NAMES UNCHANGED) =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
REF_META_JSON = "ref_titles_cache.json"
REF_EMB_NPY = "ref_embeddings.npy"
OUTPUT_FILE = "filtered.xml"
REMOTE_TITLES_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"

MODEL_PATH = "models/all-mpnet-base-v2"  # workflow should ensure this exists
USE_SMALL_MODEL = False  # if you change to mini model, toggle here

# thresholds and caps
ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_MED = 7
HYBRID_PATTERN_HIGH = 8
REF_MATCH_SIM = 0.65

MAX_REF_TITLES = 2000
MAX_REF_DAYS = 90
CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 75

DBSCAN_EPS = 0.22
DBSCAN_MIN_SAMPLES = 1

# optional source weighting (tweak as needed)
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
    t = re.sub(r'["“”‘’\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def now_iso_utc() -> str:
    # returns ISO with timezone info (UTC)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def parse_iso_to_utc_naive(s: str):
    """Parse any string to a timezone-naive UTC datetime (or None)."""
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            # assume naive times are already UTC; standardize by leaving tzinfo off
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
    dom = domain_from_url(url)
    for k, v in SOURCE_WEIGHTS.items():
        if k in dom:
            return v
    return 1.0

def atomic_write_text(path: str, lines):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line.rstrip("\n") + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def atomic_write_json(path: str, obj):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

# ===== REF META MANAGEMENT =====
def load_ref_meta_from_disk():
    if not os.path.exists(REF_META_JSON):
        return []
    try:
        with open(REF_META_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[load_ref_meta_from_disk] error:", e)
        return []

def merge_meta_lists(existing, incoming):
    """
    Merge existing and incoming meta lists (list of dicts with 'title','added_at','match_count').
    Keep existing order and update items that match. Append new incoming items at end.
    """
    if not existing:
        # dedupe incoming by title preserving order
        seen = set()
        out = []
        for m in incoming:
            t = m.get("title","").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(m)
        return out

    idx = {m.get("title",""): i for i, m in enumerate(existing)}
    merged = list(existing)
    for m in incoming:
        t = m.get("title","").strip()
        if not t:
            continue
        if t in idx:
            i = idx[t]
            # update match_count conservatively (max)
            try:
                merged[i]["match_count"] = max(merged[i].get("match_count",0), m.get("match_count",0))
            except Exception:
                merged[i]["match_count"] = m.get("match_count",0)
            # keep earliest added_at where possible
            try:
                old_dt = parse_iso_to_utc_naive(merged[i].get("added_at",""))
                new_dt = parse_iso_to_utc_naive(m.get("added_at",""))
                if new_dt and old_dt and new_dt < old_dt:
                    merged[i]["added_at"] = m.get("added_at")
                elif new_dt and not old_dt:
                    merged[i]["added_at"] = m.get("added_at")
            except Exception:
                pass
        else:
            merged.append(m)
    return merged

def prune_ref_meta(meta):
    """Evict by age, then by match_count+recency if over cap."""
    if not meta:
        return meta
    # 1) time-based
    cutoff = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=MAX_REF_DAYS))
    kept = []
    for m in meta:
        added = parse_iso_to_utc_naive(m.get("added_at",""))
        if added is None:
            # if can't parse, keep and set added_at now
            m["added_at"] = now_iso_utc()
            kept.append(m)
        else:
            if added >= cutoff:
                kept.append(m)
    meta = kept
    # 2) frequency+recency if still too large
    if len(meta) > MAX_REF_TITLES:
        now_ts = datetime.now(timezone.utc).timestamp()
        scored = []
        for m in meta:
            try:
                added_ts = parse_iso_to_utc_naive(m.get("added_at","")).timestamp()
            except Exception:
                added_ts = now_ts
            age_days = max(0.0, (now_ts - added_ts) / 86400.0)
            recency_boost = max(0.0, 1.0 - (age_days / 365.0))
            score = m.get("match_count", 0) + recency_boost
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        meta = [m for m, _ in scored[:MAX_REF_TITLES]]
    return meta

def backup_file(path):
    if not os.path.exists(path):
        return
    try:
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}")
    except Exception as e:
        print("[backup] failed:", e)

def save_ref_meta_safe(incoming_meta):
    """
    Robust save: load disk meta, merge with incoming_meta, prune, backup text file, then atomically write JSON and TXT.
    incoming_meta: list of dicts (may be small or full rebuild)
    """
    try:
        existing = load_ref_meta_from_disk()
        merged = merge_meta_lists(existing, incoming_meta or [])
        merged = prune_ref_meta(merged)
        print(f"[save_ref_meta_safe] existing={len(existing)} incoming={len(incoming_meta or [])} merged={len(merged)}")
        # backup text file before overwrite
        if os.path.exists(REFERENCE_FILE):
            backup_file(REFERENCE_FILE)
        # write files atomically
        atomic_write_json(REF_META_JSON, merged)
        titles = [m.get("title","") for m in merged]
        atomic_write_text(REFERENCE_FILE, titles)
        print(f"[save_ref_meta_safe] wrote {REF_META_JSON} and {REFERENCE_FILE} ({len(titles)} titles)")
    except Exception as e:
        print("[save_ref_meta_safe] error:", e)

# ===== REMOTE PULL + INCREMENTAL APPEND =====
def read_remote_titles(raw_url, timeout=20):
    r = requests.get(raw_url, timeout=timeout)
    r.raise_for_status()
    return [clean_title(line) for line in r.text.splitlines() if clean_title(line)]

def find_overlap_len(existing, remote, max_check=300):
    """Find largest k such that existing[-k:] == remote[:k]."""
    max_k = min(len(existing), len(remote), max_check)
    for k in range(max_k, 0, -1):
        if existing[-k:] == remote[:k]:
            return k
    return 0

def incremental_pull_and_update():
    """
    Fast path:
      - if cached_titles is prefix of remote -> append remote tail
      - else try overlap detection -> append remainder
      - else bootstrap or rebuild from remote tail (safe)
    Always calls save_ref_meta_safe to ensure TXT sync.
    """
    print("[refs] fetching remote titles...")
    remote = read_remote_titles(REMOTE_TITLES_URL)
    disk_meta = load_ref_meta_from_disk()
    cached_titles = [m["title"] for m in disk_meta] if disk_meta else []

    # fast prefix append
    if cached_titles and len(remote) >= len(cached_titles) and remote[:len(cached_titles)] == cached_titles:
        new = remote[len(cached_titles):]
        if not new:
            print("[refs] no new titles to append")
            # ensure TXT matches disk_meta
            save_ref_meta_safe(disk_meta)
            return [m["title"] for m in disk_meta]
        print(f"[refs] appending {len(new)} titles")
        new_meta = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in new]
        incoming = new_meta
        save_ref_meta_safe(incoming)
        # after save: return merged list from disk
        merged = load_ref_meta_from_disk()
        return [m["title"] for m in merged]

    # try overlap detection
    if cached_titles:
        k = find_overlap_len(cached_titles, remote, max_check=300)
        if k > 0:
            print(f"[refs] overlap detected k={k}, appending remote[{k}:]")
            append_part = remote[k:]
            incoming = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in append_part]
            save_ref_meta_safe(incoming)
            merged = load_ref_meta_from_disk()
            return [m["title"] for m in merged]

    # if no cache, bootstrap
    if not cached_titles:
        print("[refs] bootstrapping from remote tail")
        boot = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
        incoming = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in boot]
        save_ref_meta_safe(incoming)
        merged = load_ref_meta_from_disk()
        return [m["title"] for m in merged]

    # fallback: remote changed significantly -> add any remote titles not present (union)
    print("[refs] remote mismatch - merging union (safe fallback)")
    present = set(cached_titles)
    new_titles = [t for t in remote if t not in present]
    incoming = [{"title": t, "added_at": now_iso_utc(), "match_count": 0} for t in new_titles]
    save_ref_meta_safe(incoming)
    merged = load_ref_meta_from_disk()
    return [m["title"] for m in merged]

# ===== PATTERN SCORING (same logic) =====
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
    print("[main] starting")

    # 1) update reference titles from remote (incremental)
    try:
        ref_titles = incremental_pull_and_update()
    except Exception as e:
        print("[main] error updating references:", e)
        ref_titles = []
    print(f"[main] reference_titles count after update: {len(ref_titles)}")

    # 2) load model
    if USE_SMALL_MODEL:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        model_name = MODEL_PATH
    if not os.path.exists(model_name) and model_name == MODEL_PATH:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Ensure your workflow downloaded the model to this path.")
    print("[main] loading model...")
    model = SentenceTransformer(model_name)

    # 3) compute or load reference embeddings
    ref_embeddings = None
    if ref_titles:
        try:
            if os.path.exists(REF_EMB_NPY):
                try:
                    emb_cache = np.load(REF_EMB_NPY)
                    if emb_cache.shape[0] == len(ref_titles):
                        ref_embeddings = emb_cache
                        print("[main] loaded cached ref embeddings")
                    else:
                        print("[main] cached ref embeddings count mismatch; recomputing")
                        ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                        np.save(REF_EMB_NPY, ref_embeddings)
                except Exception as e:
                    print("[main] failed loading ref embeddings:", e, "computing fresh")
                    ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                    np.save(REF_EMB_NPY, ref_embeddings)
            else:
                print("[main] computing ref embeddings...")
                ref_embeddings = model.encode(ref_titles, show_progress_bar=False, convert_to_numpy=True)
                np.save(REF_EMB_NPY, ref_embeddings)
        except Exception as e:
            print("[main] error computing/loading ref embeddings:", e)
            ref_embeddings = None

    # 4) load feeds
    if not os.path.exists(FEEDS_FILE):
        raise SystemExit("[main] feeds.txt missing")
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
        except Exception as e:
            print("[main] feed parse error:", url, e)
    if not feed_articles:
        print("[main] no feed articles; exiting")
        return

    # 5) encode article titles
    titles = [a["title"] for a in feed_articles]
    print(f"[main] encoding {len(titles)} article titles...")
    article_emb = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)

    # 6) hybrid filter: pattern + semantic
    print("[main] applying hybrid filter...")
    candidates = []
    ref_meta_disk = load_ref_meta_from_disk()
    for idx, a in enumerate(feed_articles):
        t = a["title"]
        pat = calculate_analytical_score(t)
        max_sim = 0.0
        sims = None
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
        meta = a.copy()
        meta.update({
            "pattern_score": pat,
            "sim_to_refs": max_sim,
            "embedding": article_emb[idx],
            "timestamp": timestamp_from_pubdate(a.get("published","")),
            "source_weight": source_weight_from_url(a.get("link",""))
        })
        # increment match_count for strongly matched refs
        if sims is not None and ref_meta_disk:
            for mi, s in enumerate(sims):
                if s >= REF_MATCH_SIM:
                    try:
                        ref_meta_disk[mi]["match_count"] = ref_meta_disk[mi].get("match_count", 0) + 1
                    except Exception:
                        pass
        candidates.append(meta)

    # persist updated match counts safely
    if ref_meta_disk:
        try:
            ref_meta_disk = prune_ref_meta(ref_meta_disk)
            save_incoming = ref_meta_disk  # we want to update disk file with these modifications
            save_ref_meta_safe(save_incoming)  # merge-safe write
        except Exception as e:
            print("[main] saving ref meta failed:", e)

    # 7) time cutoff
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)).timestamp())
    candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
    if not candidates:
        print("[main] no recent candidates after cutoff; exiting")
        return

    # 8) clustering (DBSCAN)
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
        cluster_list.append({"label": lbl, "items": items, "score": cluster_score(items)})
    cluster_list.sort(key=lambda x: x["score"], reverse=True)

    # 9) region detection (simple keywords)
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

    # 11) write XML output (merge safe)
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