import feedparser, hashlib, os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET

REF_FILE = "reference_titles.txt"

# Load feeds
with open("feeds.txt") as f:
    FEEDS = [line.strip() for line in f if line.strip()]

# Load reference titles
if os.path.exists(REF_FILE):
    with open(REF_FILE) as f:
        REF_TITLES = [line.strip() for line in f if line.strip()]
else:
    REF_TITLES = []

def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()

def fetch_articles():
    arts = []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = e.get("title", "").strip()
                link = e.get("link", "")
                pub = e.get("published", datetime.utcnow().isoformat())
                if title:
                    arts.append({"title": title, "link": link, "published": pub})
        except Exception as ex:
            print(f"Error reading {url}: {ex}")
    return arts

def dedupe(arts):
    seen, out = set(), []
    for a in arts:
        h = hash_text(a["title"])
        if h not in seen:
            out.append(a)
            seen.add(h)
    return out

def write_rss(arts):
    rss = ET.Element("rss", version="2.0")
    ch = ET.SubElement(rss, "channel")
    ET.SubElement(ch, "title").text = "AI Filtered Editorial Feed"
    ET.SubElement(ch, "link").text = "https://evilgodfahim.github.io/rss-filter/"
    ET.SubElement(ch, "description").text = "Auto-filtered editorial and geopolitics content."

    for a in arts:
        item = ET.SubElement(ch, "item")
        ET.SubElement(item, "title").text = a["title"]
        ET.SubElement(item, "link").text = a["link"]
        ET.SubElement(item, "pubDate").text = a["published"]

    tree = ET.ElementTree(rss)
    tree.write("filtered.xml", encoding="utf-8", xml_declaration=True)

def update_reference_titles(new_titles):
    """Keep up to 1000 titles and add max 2 new ones"""
    global REF_TITLES
    new_to_add = [t for t in new_titles if t not in REF_TITLES][:2]
    if new_to_add:
        REF_TITLES = (REF_TITLES + new_to_add)[-1000:]
        with open(REF_FILE, "w") as f:
            f.write("\n".join(REF_TITLES))
        print(f"Added {len(new_to_add)} new titles to reference list.")

def main():
    print("Fetching feeds...")
    arts = fetch_articles()
    arts = dedupe(arts)
    print(f"Fetched {len(arts)} unique articles")

    if not REF_TITLES:
        print("No reference titles found — aborting filter.")
        return

    print("Applying AI filter...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ref_emb = model.encode(REF_TITLES, convert_to_tensor=True)
    art_titles = [a["title"] for a in arts]
    art_emb = model.encode(art_titles, convert_to_tensor=True)

    scores = cosine_similarity(art_emb, ref_emb).max(axis=1)
    filtered = [a for a, s in zip(arts, scores) if s > 0.55]

    print(f"Filtered down to {len(filtered)} articles")
    write_rss(filtered)

    # Add top 2 new filtered titles to the reference list
    if filtered:
        new_titles = [a["title"] for a in sorted(filtered, key=lambda x: x["published"], reverse=True)]
        update_reference_titles(new_titles)

if __name__ == "__main__":
    main()
