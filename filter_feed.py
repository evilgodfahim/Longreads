import feedparser
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
BANGla_THRESHOLD = 0.75  # strict for Bangla
ENGLISH_THRESHOLD = 0.65  # current/default for English
MAX_NEW_TITLES = 2

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[“”‘’\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def dedupe(articles):
    seen, out = set(), []
    for a in articles:
        h = hashlib.md5(a["title"].encode()).hexdigest()
        if h not in seen:
            out.append(a)
            seen.add(h)
    return out

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
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    REF_TITLES = [clean_title(line) for line in f if line.strip()]

# ===== EMBEDDINGS =====
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
ref_embeddings = model.encode(REF_TITLES)

# ===== FILTER ARTICLES =====
filtered_articles = []
for article in feed_articles:
    title_clean = clean_title(article["title"])

    # Simple language detection
    if re.search(r'[\u0980-\u09FF]', title_clean):
        lang = "bangla"
        threshold = BANGla_THRESHOLD  # strict
    else:
        lang = "english"
        threshold = ENGLISH_THRESHOLD  # original threshold

    emb = model.encode([title_clean])
    sim_scores = cosine_similarity(emb, ref_embeddings)

    if sim_scores.max() >= threshold:
        filtered_articles.append(article)

# Deduplicate
filtered_articles = dedupe(filtered_articles)

# ===== WRITE OUTPUT XML =====
rss = ET.Element("rss", version="2.0")
channel = ET.SubElement(rss, "channel")
ET.SubElement(channel, "title").text = "Filtered Feed"
ET.SubElement(channel, "link").text = "https://yourrepo.github.io/"
ET.SubElement(channel, "description").text = "Filtered articles"

for a in filtered_articles:
    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = a["title"]
    ET.SubElement(item, "link").text = a["link"]
    ET.SubElement(item, "pubDate").text = a["published"]

tree = ET.ElementTree(rss)
tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

# ===== UPDATE REFERENCE TITLES =====
eligible_titles = []
languages_seen = set()
sources_seen = set()

for article in filtered_articles:
    t = clean_title(article["title"])
    if t in REF_TITLES:
        continue

    # Language detection
    if re.search(r'[\u0980-\u09FF]', t):
        lang = "bangla"
    else:
        lang = "english"

    src = article.get("feed_source", "unknown")

    if lang in languages_seen or src in sources_seen:
        continue

    eligible_titles.append((t, src, lang))
    languages_seen.add(lang)
    sources_seen.add(src)

# Randomly pick up to MAX_NEW_TITLES
random.shuffle(eligible_titles)
to_add = eligible_titles[:MAX_NEW_TITLES]

# Append to reference_titles.txt
if to_add:
    with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
        for t, src, lang in to_add:
            f.write(t + "\n")
