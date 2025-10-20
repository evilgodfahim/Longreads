import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
ENGLISH_THRESHOLD = 0.65
MAX_NEW_TITLES = 2
REFERENCE_MAX = 2000

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[“”‘’\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def detect_language(title):
    return "english"  # Only English now

# ===== LOAD FEEDS =====
with open(FEEDS_FILE, "r") as f:
    feed_urls = [line.strip() for line in f if line.strip()]

feed_articles = []
for url in feed_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
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

# ===== FILTER ARTICLES =====
filtered_articles = []
eligible_titles = []

for article in feed_articles:
    title_clean = clean_title(article["title"])
    threshold = ENGLISH_THRESHOLD
    emb = model.encode([title_clean])
    sim_scores = cosine_similarity(emb, ref_embeddings) if ref_embeddings.size else []

    if len(sim_scores) == 0 or sim_scores.max() >= threshold:
        filtered_articles.append(article)
        if title_clean not in REF_TITLES:
            eligible_titles.append((title_clean, article.get("feed_source", "unknown"), "english"))

# ===== WRITE OUTPUT XML =====
rss = ET.Element("rss", version="2.0")
channel = ET.SubElement(rss, "channel")
ET.SubElement(channel, "title").text = "Filtered Feed"
ET.SubElement(channel, "link").text = "https://yourrepo.github.io/"
ET.SubElement(channel, "description").text = "Filtered articles"

seen_titles = set()
for a in filtered_articles:
    t = a["title"]
    if t in seen_titles:
        continue
    seen_titles.add(t)
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = t
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a["published"]

ET.ElementTree(rss).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

# ===== UPDATE REFERENCE TITLES =====
if len(REF_TITLES) < REFERENCE_MAX:
    random.shuffle(eligible_titles)
    to_add = eligible_titles[:MAX_NEW_TITLES]
    if to_add:
        with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
            for t, _, _ in to_add:
                f.write(t + "\n")
