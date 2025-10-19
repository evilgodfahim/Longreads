import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import json
import os

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
LAST_SEEN_FILE = "last_seen.json"
BANGla_THRESHOLD = 0.80
ENGLISH_THRESHOLD = 0.65
MAX_NEW_TITLES = 2
REFERENCE_MAX = 1000

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[“”‘’\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def detect_language(title):
    return "bangla" if re.search(r'[\u0980-\u09FF]', title) else "english"

def load_last_seen():
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_last_seen(last_seen):
    with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(last_seen, f, ensure_ascii=False, indent=2)

# ===== LOAD FEEDS =====
with open(FEEDS_FILE, "r") as f:
    feed_urls = [line.strip() for line in f if line.strip()]

last_seen = load_last_seen()
feed_articles = []

for url in feed_urls:
    feed = feedparser.parse(url)
    entries = feed.entries
    last_link = last_seen.get(url, None)

    # Assume entries are newest first
    new_entries = []
    for entry in entries:
        if entry.link == last_link:
            break
        new_entries.append({
            "title": entry.title,
            "link": entry.link,
            "published": getattr(entry, "published", ""),
            "feed_source": url
        })

    if new_entries:
        feed_articles.extend(new_entries)
        # Update last seen link for this feed
        last_seen[url] = new_entries[0]["link"]

# Save updated last_seen
save_last_seen(last_seen)

# ===== LOAD REFERENCE TITLES =====
with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    REF_TITLES = [clean_title(line) for line in f if line.strip()]

# ===== EMBEDDINGS =====
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
ref_embeddings = model.encode(REF_TITLES)

# ===== FILTER ARTICLES =====
filtered_articles = []
eligible_titles = []

for article in feed_articles:
    title_clean = clean_title(article["title"])
    lang = detect_language(title_clean)
    threshold = BANGla_THRESHOLD if lang == "bangla" else ENGLISH_THRESHOLD

    emb = model.encode([title_clean])
    sim_scores = cosine_similarity(emb, ref_embeddings)

    if sim_scores.max() >= threshold:
        filtered_articles.append(article)

        if title_clean not in REF_TITLES:
            eligible_titles.append((title_clean, article.get("feed_source", "unknown"), lang))

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
    # Before 1000: random addition, max 1 Bangla title
    bangla_added = False
    languages_seen = set()
    sources_seen = set()
    random.shuffle(eligible_titles)
    to_add = []
    for t, src, lang in eligible_titles:
        if len(to_add) >= MAX_NEW_TITLES:
            break
        if lang in languages_seen or src in sources_seen:
            continue
        if lang == "bangla" and bangla_added:
            continue
        if lang == "bangla":
            bangla_added = True
        to_add.append((t, src, lang))
        languages_seen.add(lang)
        sources_seen.add(src)

    if to_add:
        with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
            for t, _, _ in to_add:
                f.write(t + "\n")

else:
    # After 1000: English rolling only
    for idx, t in enumerate(REF_TITLES):
        if detect_language(t) == "english":
            REF_TITLES.pop(idx)
            break
    english_candidates = [(t, src, lang) for t, src, lang in eligible_titles if lang == "english"]
    if english_candidates:
        t_new, _, _ = random.choice(english_candidates)
        REF_TITLES.append(t_new)
        with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
            for t in REF_TITLES:
                f.write(t + "\n")
