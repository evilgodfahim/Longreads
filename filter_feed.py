#!/usr/bin/env python3
"""
hot_topics_feed_full.py
Lifetime-ready RSS filter:
 - incremental last-seen reference-title fetch from GitHub raw file
 - embedding cache (only new titles encoded)
 - hybrid eviction (time + frequency) when MAX_REF_TITLES exceeded
 - hybrid pattern + semantic filtering
 - clustering -> hot-topic detection
 - diversity-aware selection
 - outputs compact filtered.xml
"""

import os
import re
import json
import time
import hashlib
import requests
import feedparser
import numpy as np
import xml.etree.ElementTree as ET
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# ===== CONFIG =====
GITHUB_RAW_URL = "https://raw.githubusercontent.com/evilgodfahim/ref/main/titles.txt"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional if repo private
FEEDS_FILE = "feeds.txt"
OUTPUT_FILE = "filtered.xml"

REF_CACHE_JSON = "ref_titles_cache.json"   # stores list of dicts: {title, added_at, match_count}
REF_EMB_NPY = "ref_embeddings.npy"
REF_CHECKPOINT = "ref_checkpoint.json"     # stores last_seen_idx & prefix_hash

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # change to MiniLM for speed
USE_SMALL_MODEL = False

# Limits & thresholds
MAX_REF_TITLES = 2000      # hard cap on stored reference titles
MAX_REF_DAYS = 90          # remove refs older than this (time-based)
REF_MATCH_SIM = 0.65       # similarity threshold to increment match_count
ENGLISH_SIM_THRESHOLD = 0.60
HYBRID_MIN_SIM_LOW = 0.33
HYBRID_MIN_SIM_HIGH = 0.38
HYBRID_PATTERN_HIGH = 8
HYBRID_PATTERN_MED = 7

CUTOFF_HOURS = 36
MAX_OUTPUT_ITEMS = 75

DBSCAN_EPS = 0.22          # cosine distance threshold (0.22 ~ similarity 0.78)
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
def http_get(url, token=None, timeout=15):
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def clean_title(t):
    if not t:
        return ""
    t = t.strip()
    t = re.sub(r'["“”‘’\'`]', "", t)
    t = re.sub(r"\s+", " ", t)
    return t

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
    domain = domain_from_url(url)
    for d, w in SOURCE_WEIGHTS.items():
        if d in domain:
            return w
    return 1.0

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ===== PATTERN SCORE (your function, same logic) =====
def calculate_analytical_score(title):
    if not title:
        return 0
    tl = title.lower()
    score = 0
    crisis_terms = [ 'war','conflict','crisis','collapse','violence','attack','strikes','invasion','sanctions','blockade','ceasefire','offensive','bombing','insurgency','coup','protests','unrest','genocide','massacre','refugees','displacement','humanitarian','famine','drought','terror','clashes','shelling','uprising','hostilities','tensions','retaliation','raid','mobilization','escalation','hostage','airstrike','militia','civil war','rebels','armed group']
    crisis_count = sum(1 for term in crisis_terms if term in tl)
    score += min(crisis_count * 2, 6)

    relation_patterns = ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']
    if any(p in tl for p in relation_patterns):
        score += 2

    if "'s " in tl or "'" in tl:
        score += 2

    question_starters = ['why ','how ','what ','can ','will ','should ','is ','are ','do ','does ','could ','may ','might ','would ']
    if any(tl.startswith(q) for q in question_starters):
        score += 2

    geo_terms = ['economic','trade','debt','inflation','currency','military','nuclear','election','government','regime','policy','treaty','agreement','dispute','territorial','sovereignty','peacekeeping','intervention','occupation','embargo','tariffs','bilateral','multilateral','alliance','nato','un','security','defense','border','independence','autonomy','separatist','geopolitics','foreign policy','diplomacy','strategy','strategic','realignment','deterrence','containment','bloc','regional order','sanction','power balance','influence','proxy','geostrategic','foreign affairs','statecraft','hegemony','multipolar','unipolar','arms race','militarization','deterrent','security council']
    geo_count = sum(1 for term in geo_terms if term in tl)
    score += min(geo_count, 3)

    outcome_terms = ['faces','threatens','undermines','deepens','escalates','intensifies','worsens','persists','continues','remains','struggles','fails','succeeds','achieves','drives','fuels','prevents','blocks','enables','forces','triggers','sparks','creates','causes','leads to','results in','complicates','reshapes','transforms','revives','shifts','redefines','contributes to','signals','marks','heightens','deteriorates']
    if any(term in tl for term in outcome_terms):
        score += 1

    explanatory = ['reveals','shows','exposes','means for','means to','impact on','impact of','effect on','effect of','implications','consequences','amid','despite','after','before','during','following','since','until','significance','context','analysis','explained','lessons from']
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

# ===== REFERENCE CACHE I/O =====
def load_ref_cache():
    titles_meta = []
    emb = None
    checkpoint = {"last_seen_idx": -1, "prefix_hash": ""}
    if os.path.exists(REF_CACHE_JSON):
        try:
            with open(REF_CACHE_JSON, "r", encoding="utf-8") as f:
                titles_meta = json.load(f)
        except Exception:
            titles_meta = []
    if os.path.exists(REF_EMB_NPY):
        try:
            emb = np.load(REF_EMB_NPY)
        except Exception:
            emb = None
    if os.path.exists(REF_CHECKPOINT):
        try:
            with open(REF_CHECKPOINT, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            checkpoint = {"last_seen_idx": -1, "prefix_hash": ""}
    return titles_meta, emb, checkpoint

def save_ref_cache(titles_meta, emb, last_seen_idx, prefix_hash):
    with open(REF_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(titles_meta, f, ensure_ascii=False, indent=2)
    if emb is not None:
        np.save(REF_EMB_NPY, emb)
    with open(REF_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"last_seen_idx": last_seen_idx, "prefix_hash": prefix_hash}, f)

# ===== FETCH & INCREMENTAL UPDATE =====
def fetch_remote_titles(raw_url, token=None):
    text = http_get(raw_url, token)
    lines = [clean_title(line) for line in text.splitlines() if line.strip()]
    return lines

def prefix_hash(titles, n=30):
    # small hash of first n lines; helps detect rotation/truncation quickly
    head = "\n".join(titles[:n])
    return md5(head)

def incremental_update_refs(raw_url, model, token=None):
    remote = fetch_remote_titles(raw_url, token)
    remote = remote  # list of strings
    titles_meta, emb, checkpoint = load_ref_cache()
    cached_titles = [m["title"] for m in titles_meta] if titles_meta else []
    last_seen_idx = checkpoint.get("last_seen_idx", -1)
    cached_prefix_hash = checkpoint.get("prefix_hash", "")

    # Normal append-only fast path: cached_titles is exact prefix of remote
    if cached_titles and len(remote) >= len(cached_titles) and remote[:len(cached_titles)] == cached_titles:
        new_titles = remote[len(cached_titles):]
        if not new_titles:
            # nothing new
            return titles_meta, emb
        # create new meta entries
        new_meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in new_titles]
        # encode only new titles
        print(f"[refs] {len(new_titles)} new reference titles -> encoding")
        new_emb = model.encode(new_titles, show_progress_bar=False, convert_to_numpy=True)
        if emb is None:
            emb = new_emb
        else:
            emb = np.vstack([emb, new_emb])
        titles_meta = titles_meta + new_meta
        # prune to MAX_REF_TITLES (by recency first)
        titles_meta, emb = prune_ref_meta_and_emb(titles_meta, emb)
        # update checkpoint
        last_seen_idx = len(titles_meta) - 1
        p_hash = prefix_hash([m["title"] for m in titles_meta])
        save_ref_cache(titles_meta, emb, last_seen_idx, p_hash)
        return titles_meta, emb

    # If cache empty -> compute fresh (but keep only last MAX_REF_TITLES)
    if not cached_titles:
        to_use = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
        if to_use:
            print(f"[refs] cache empty -> encoding {len(to_use)} refs")
            emb = model.encode(to_use, show_progress_bar=False, convert_to_numpy=True)
            titles_meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in to_use]
            last_seen_idx = len(titles_meta) - 1
            p_hash = prefix_hash([m["title"] for m in titles_meta])
            save_ref_cache(titles_meta, emb, last_seen_idx, p_hash)
        return titles_meta, emb

    # Otherwise remote changed non-trivially (rotation/truncation). Try to align tails.
    # If remote shorter and remote tail matches cached tail -> keep matched tail.
    tail_len = min(len(remote), len(cached_titles), 300)
    if tail_len > 0 and cached_titles[-tail_len:] == remote[-tail_len:]:
        # find overlap start index in cached_titles where remote's tail aligns
        overlap_start = len(cached_titles) - tail_len
        # rebuild new titles_meta from matched tail or remote entire
        # We'll re-use embeddings for the overlapped tail portion and recompute for any earlier remote entries
        keep = cached_titles[overlap_start:]
        keep_meta = titles_meta[overlap_start:]
        # remote may have some earlier lines we need to compute
        prefix_part = remote[:-tail_len] if len(remote) > tail_len else []
        new_meta_prefix = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in prefix_part]
        # encode prefix_part
        prefix_emb = None
        if prefix_part:
            print(f"[refs] file rotated/truncated -> encoding prefix {len(prefix_part)}")
            prefix_emb = model.encode(prefix_part, show_progress_bar=False, convert_to_numpy=True)
        # now combine prefix_emb + overlapped embeddings (from cached)
        overlapped_emb = emb[overlap_start:] if emb is not None else None
        if prefix_emb is None and overlapped_emb is None:
            combined_emb = None
        elif overlapped_emb is None:
            combined_emb = prefix_emb
        elif prefix_emb is None:
            combined_emb = overlapped_emb
        else:
            combined_emb = np.vstack([prefix_emb, overlapped_emb])
        titles_meta = new_meta_prefix + keep_meta
        emb = combined_emb
        titles_meta, emb = prune_ref_meta_and_emb(titles_meta, emb)
        last_seen_idx = len(titles_meta) - 1
        p_hash = prefix_hash([m["title"] for m in titles_meta])
        save_ref_cache(titles_meta, emb, last_seen_idx, p_hash)
        return titles_meta, emb

    # fallback: remote changed significantly -> recompute using last MAX_REF_TITLES
    to_use = remote[-MAX_REF_TITLES:] if len(remote) > MAX_REF_TITLES else remote
    if to_use:
        print("[refs] remote changed significantly -> full recompute of embeddings")
        emb = model.encode(to_use, show_progress_bar=False, convert_to_numpy=True)
        titles_meta = [{"title": t, "added_at": now_iso(), "match_count": 0} for t in to_use]
        last_seen_idx = len(titles_meta) - 1
        p_hash = prefix_hash([m["title"] for m in titles_meta])
        save_ref_cache(titles_meta, emb, last_seen_idx, p_hash)
    return titles_meta, emb

# ===== PRUNING / EVICTION =====
def prune_ref_meta_and_emb(titles_meta, emb):
    # Remove by age first (older than MAX_REF_DAYS)
    if not titles_meta:
        return titles_meta, emb
    cutoff_dt = datetime.utcnow() - timedelta(days=MAX_REF_DAYS)
    keep_mask = []
    for m in titles_meta:
        try:
            added = dateparser.parse(m.get("added_at"))
        except Exception:
            added = datetime.utcnow()
        keep_mask.append(added >= cutoff_dt)
    if any(keep_mask) and not all(keep_mask):
        # apply mask
        new_meta = [m for m, k in zip(titles_meta, keep_mask) if k]
        if emb is not None:
            emb = emb[[i for i, k in enumerate(keep_mask) if k], :]
        titles_meta = new_meta

    # If still too large, keep top by match_count then recency
    if len(titles_meta) > MAX_REF_TITLES:
        # compute score = match_count + recency_factor (recent -> higher)
        scored = []
        now_ts = datetime.utcnow().timestamp()
        for i, m in enumerate(titles_meta):
            try:
                added_ts = dateparser.parse(m.get("added_at")).timestamp()
            except:
                added_ts = now_ts
            age_days = max(0.0, (now_ts - added_ts) / 86400.0)
            recency_boost = max(0.0, 1.0 - (age_days / 365.0))  # decays over a year
            score = m.get("match_count", 0) + recency_boost
            scored.append((i, score))
        # keep indices of top MAX_REF_TITLES
        scored.sort(key=lambda x: x[1], reverse=True)
        keep_indices = set(i for i, _ in scored[:MAX_REF_TITLES])
        new_meta = [m for idx, m in enumerate(titles_meta) if idx in keep_indices]
        if emb is not None:
            emb = emb[[idx for idx, _ in scored[:MAX_REF_TITLES]], :]
        titles_meta = new_meta

    return titles_meta, emb

# ===== REGION DETECTION (simple keywords) =====
REGION_KEYWORDS = {
    "middle east": ["gaza","israel","palestine","saudi","yemen","syria","iran","turkey","egypt"],
    "europe": ["ukraine","russia","uk","france","germany","europe","balkans","georgia","armenia","azerbaijan"],
    "south asia": ["india","pakistan","bangladesh","afghanistan","sri lanka","nepal"],
    "east asia": ["china","taiwan","korea","japan","north korea","south korea"],
    "africa": ["ethiopia","sudan","nigeria","kenya","somalia","mali","libya"],
    "latin america": ["venezuela","colombia","brazil","mexico"],
    "global": []
}
def detect_region(title):
    tl = title.lower()
    for r, kws in REGION_KEYWORDS.items():
        for kw in kws:
            if kw in tl:
                return r
    return "global"

# ===== MAIN =====
def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2" if USE_SMALL_MODEL else MODEL_NAME
    model = SentenceTransformer(model_name)

    if not os.path.exists(FEEDS_FILE):
        raise SystemExit("feeds.txt not found")

    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        feed_urls = [l.strip() for l in f if l.strip()]

    # 1) update reference titles & embeddings incrementally (last-seen friendly)
    print("[main] updating reference titles from GitHub...")
    ref_meta, ref_emb = incremental_update_refs(GITHUB_RAW_URL, model, GITHUB_TOKEN)
    ref_titles = [m["title"] for m in ref_meta] if ref_meta else []
    print(f"[main] loaded {len(ref_titles)} reference titles")

    # 2) parse feeds
    print("[main] parsing feeds...")
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
        print("[main] no feed articles found")
        return

    # 3) encode feed titles
    print("[main] encoding feed titles...")
    titles = [a["title"] for a in feed_articles]
    article_emb = model.encode(titles, show_progress_bar=False, convert_to_numpy=True)

    # 4) hybrid filter (pattern + semantic)
    print("[main] hybrid filtering...")
    candidates = []
    for idx, a in enumerate(feed_articles):
        t = a["title"]
        pat = calculate_analytical_score(t)
        max_sim = 0.0
        if ref_emb is not None and ref_emb.size > 0:
            sims = cosine_similarity([article_emb[idx]], ref_emb)[0]
            max_sim = float(np.max(sims))
            best_ref_idx = int(np.argmax(sims))
        else:
            sims = None
            best_ref_idx = None

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

        a_meta = a.copy()
        a_meta.update({
            "pattern_score": pat,
            "sim_to_refs": max_sim,
            "embedding": article_emb[idx],
            "timestamp": timestamp_from_pubdate(a.get("published","")),
            "source_weight": source_weight_from_url(a.get("link",""))
        })
        # increment match_count for matched reference indices above threshold (helps eviction)
        if sims is not None:
            matched_idxs = [i for i, s in enumerate(sims) if s >= REF_MATCH_SIM]
            for mi in matched_idxs:
                try:
                    ref_meta[mi]["match_count"] = ref_meta[mi].get("match_count", 0) + 1
                except Exception:
                    pass

        candidates.append(a_meta)

    # save ref cache after updating match_counts (so frequency persists)
    # ensure emb and meta lengths match
    if ref_meta and ref_emb is not None and len(ref_meta) == ref_emb.shape[0]:
        p_hash = prefix_hash([m["title"] for m in ref_meta])
        last_idx = len(ref_meta) - 1
        save_ref_cache(ref_meta, ref_emb, last_idx, p_hash)

    # 5) time cutoff
    cutoff_ts = int((datetime.utcnow() - timedelta(hours=CUTOFF_HOURS)).timestamp())
    candidates = [c for c in candidates if c["timestamp"] >= cutoff_ts]
    if not candidates:
        print("[main] no recent candidates after cutoff")
        return

    # 6) clustering (DBSCAN on embeddings)
    print("[main] clustering candidates...")
    X = np.vstack([c["embedding"] for c in candidates])
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit(X)
    labels = clustering.labels_
    for lbl, c in zip(labels, candidates):
        c["cluster"] = int(lbl)

    # 7) group clusters and score them
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

    # 8) diversity pass - pick best representative per cluster, prefer unseen regions
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

    # fallback: if empty, take top items from cluster_list
    if not final:
        for cl in cluster_list[:MAX_OUTPUT_ITEMS]:
            final.append(cl["items"][0])

    # 9) write XML output
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

if __name__ == "__main__":
    main()