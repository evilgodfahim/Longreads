import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os
import json
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
LAST_SEEN_FILE = "last_seen.json"
ENGLISH_THRESHOLD = 0.65
MAX_NEW_TITLES = 2
REFERENCE_MAX = 1000  # stop adding completely at 1000
HOURS_LIMIT = 36      # only articles not older than 36 hours

# ===== MODEL CONFIG =====
MODEL_PATH = "models/paraphrase-MiniLM-L6-v2"
MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[“”‘’\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_pubdate(pubdate_str):
    """Convert pubDate string to timezone-aware UTC datetime"""
    if not pubdate_str:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = parsedate_to_datetime(pubdate_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except:
        return datetime.min.replace(tzinfo=timezone.utc)

# ===== LOAD MODEL =====
if os.path.exists(MODEL_PATH):
    print(f"Loading local model from {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
else:
    print(f"Local model not found. Downloading {MODEL_NAME}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_PATH)
    print(f"Model downloaded and cached at {MODEL_PATH}")

# ===== LOAD FEEDS =====
with open(FEEDS_FILE, "r") as f:
    feed_urls = [line.strip() for line in f if line.strip()]

feed_articles = []
time_limit = datetime.now(timezone.utc) - timedelta(hours=HOURS_LIMIT)

for url in feed_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        pub_dt = parse_pubdate(getattr(entry, "published", ""))
        if pub_dt < time_limit:
            continue  # skip old articles
        feed_articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", ""),
            "feed_source": url
        })

# ===== LOAD REFERENCE TITLES =====
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    REF_TITLES = [clean_title(line) for line in f if line.strip()]

# ===== EMBEDDINGS =====
ref_embeddings = model.encode(REF_TITLES)

# ===== FILTER ARTICLES =====
filtered_articles = []
eligible_titles = []

for article in feed_articles:
    title_clean = clean_title(article["title"])
    emb = model.encode([title_clean])
    sim_scores = cosine_similarity(emb, ref_embeddings)
    if sim_scores.max() >= ENGLISH_THRESHOLD:
        filtered_articles.append(article)
        if title_clean not in REF_TITLES:
            eligible_titles.append((title_clean, article.get("feed_source", "unknown")))

# ===== WRITE OUTPUT XML =====
if os.path.exists(OUTPUT_FILE):
    existing_tree = ET.parse(OUTPUT_FILE)
    existing_root = existing_tree.getroot()
    existing_channel = existing_root.find("channel")
    existing_items = existing_channel.findall("item")
else:
    existing_root = ET.Element("rss", version="2.0")
    existing_channel = ET.SubElement(existing_root, "channel")
    ET.SubElement(existing_channel, "title").text = "Filtered Feed"
    ET.SubElement(existing_channel, "link").text = "https://yourrepo.github.io/"
    ET.SubElement(existing_channel, "description").text = "Filtered articles"
    existing_items = []

existing_titles = {item.find("title").text for item in existing_items if item.find("title") is not None}

new_items = []
for a in filtered_articles:
    t = a["title"]
    if t in existing_titles:
        continue
    item = ET.Element("item")
    ET.SubElement(item, "title").text = t
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a["published"]
    new_items.append(item)

all_items = existing_items + new_items
all_items_sorted = sorted(all_items, key=lambda x: parse_pubdate(x.find("pubDate").text if x.find("pubDate") is not None else ""), reverse=True)

for item in existing_channel.findall("item"):
    existing_channel.remove(item)
for item in all_items_sorted:
    existing_channel.append(item)

ET.ElementTree(existing_root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

# ===== UPDATE REFERENCE TITLES =====
if len(REF_TITLES) >= REFERENCE_MAX:
    print("Reference titles reached max. No new titles added.")
else:
    random.shuffle(eligible_titles)
    to_add = eligible_titles[:MAX_NEW_TITLES]
    if to_add:
        with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
            for t, _ in to_add:
                f.write(t + "\n")
