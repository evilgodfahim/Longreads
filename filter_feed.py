import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os
from datetime import datetime, timedelta
from dateutil import parser

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
TEMP_FILE = "temp.xml"
ENGLISH_THRESHOLD = 0.50
MAX_NEW_TITLES = 2
REFERENCE_MAX = 2000
MAX_XML_ITEMS = 200
CUT_OFF_HOURS = 36

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[“”‘’\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def detect_language(title):
    return "english"

def pubdate_to_minutes(pubdate_str):
    """
    Convert any pubDate string to a UTC timestamp in minutes.
    Handles date-only, datetime with timezone, and malformed strings.
    """
    try:
        dt = parser.parse(pubdate_str, fuzzy=True)
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        return int(dt.timestamp() // 60)
    except Exception:
        return 0  # very old value for invalid dates

# ===== LOAD FEEDS =====
with open(FEEDS_FILE, "r") as f:
    feed_urls = [line.strip() for line in f if line.strip()]

feed_articles = []
for url in feed_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        if not getattr(entry, "title", None) or not getattr(entry, "link", None):
            continue
        feed_articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", ""),
            "feed_source": url
        })

# ===== LOAD REFERENCE TITLES =====
if os.path.exists(REFERENCE_FILE):
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        REF_TITLES = [clean_title(line) for line in f if line.strip()]
else:
    REF_TITLES = []

# ===== LOAD LOCAL MODEL =====
MODEL_PATH = "models/paraphrase-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run workflow to download first.")

model = SentenceTransformer(MODEL_PATH)
ref_embeddings = model.encode(REF_TITLES)

# ===== FILTER ARTICLES BY SIMILARITY =====
filtered_articles = []
eligible_titles = []

for article in feed_articles:
    title_clean = clean_title(article["title"])
    emb = model.encode([title_clean])
    sim_scores = cosine_similarity(emb, ref_embeddings) if ref_embeddings.size else []

    if len(sim_scores) == 0 or sim_scores.max() >= ENGLISH_THRESHOLD:
        filtered_articles.append(article)
        if title_clean not in REF_TITLES:
            eligible_titles.append((title_clean, article.get("feed_source", "unknown"), "english"))

# ===== FILTER BY LAST 36 HOURS USING MINUTES =====
cutoff_minutes = int((datetime.utcnow() - timedelta(hours=CUT_OFF_HOURS)).timestamp() // 60)
recent_articles = []
for a in filtered_articles:
    minutes = pubdate_to_minutes(a.get("published", ""))
    if minutes > cutoff_minutes:
        recent_articles.append(a)

filtered_articles = recent_articles

# ===== SORT BY PUBLISH DATE DESC =====
filtered_articles.sort(key=lambda x: pubdate_to_minutes(x.get("published", "")), reverse=True)

# ===== CREATE TEMPORARY XML WITH STRICT CHRONOLOGICAL ORDER =====
temp_root = ET.Element("rss", version="2.0")
temp_channel = ET.SubElement(temp_root, "channel")
ET.SubElement(temp_channel, "title").text = "Temporary Chronological Feed"
ET.SubElement(temp_channel, "link").text = "https://yourrepo.github.io/temp"
ET.SubElement(temp_channel, "description").text = "Temporary chronological staging feed"

# Write all filtered articles (already sorted by pubDate desc)
for a in filtered_articles:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = a["title"]
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a["published"]
    temp_channel.append(item)

# Overwrite temp.xml every run
ET.ElementTree(temp_root).write(TEMP_FILE, encoding="utf-8", xml_declaration=True)

# ===== LOAD TEMP.XML BACK AND USE FOR FINAL FILTERED.XML =====
tree_temp = ET.parse(TEMP_FILE)
root_temp = tree_temp.getroot()
channel_temp = root_temp.find("channel")

sorted_temp_articles = []
for item in channel_temp.findall("item"):
    title_el = item.find("title")
    link_el = item.find("link")
    date_el = item.find("pubDate")
    if title_el is not None and link_el is not None:
        sorted_temp_articles.append({
            "title": title_el.text,
            "link": link_el.text,
            "published": date_el.text if date_el is not None else ""
        })

# ===== LOAD EXISTING FINAL XML =====
if os.path.exists(OUTPUT_FILE):
    tree = ET.parse(OUTPUT_FILE)
    root = tree.getroot()
    channel = root.find("channel")
else:
    root = ET.Element("rss", version="2.0")
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = "Filtered Feed"
    ET.SubElement(channel, "link").text = "https://yourrepo.github.io/"
    ET.SubElement(channel, "description").text = "Filtered articles"

# ===== BUILD EXISTING TITLES SET =====
existing_titles = set()
for item in channel.findall("item"):
    t = item.find("title")
    if t is not None:
        existing_titles.add(t.text)

# ===== ADD NEW ITEMS AT THE TOP FROM TEMP.XML =====
for a in reversed(sorted_temp_articles):  # strictly sorted from temp.xml
    t = a["title"]
    if t in existing_titles:
        continue
    item = ET.Element("item")
    ET.SubElement(item, "title").text = t
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a["published"]
    channel.insert(0, item)
    existing_titles.add(t)

# ===== LIMIT XML ITEMS =====
all_items = channel.findall("item")
if len(all_items) > MAX_XML_ITEMS:
    for item in all_items[MAX_XML_ITEMS:]:
        channel.remove(item)

# ===== WRITE FINAL FILTERED.XML =====
ET.ElementTree(root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

# ===== UPDATE REFERENCE TITLES =====
if len(REF_TITLES) < REFERENCE_MAX:
    random.shuffle(eligible_titles)
    to_add = eligible_titles[:MAX_NEW_TITLES]
    if to_add:
        with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
            for t, _, _ in to_add:
                f.write(t + "\n")
