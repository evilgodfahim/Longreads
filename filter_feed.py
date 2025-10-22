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
ENGLISH_THRESHOLD = 0.55   # lowered slightly to catch more phrasing variety
MAX_NEW_TITLES = 2
REFERENCE_MAX = 2000
MAX_XML_ITEMS = 500
CUT_OFF_HOURS = 36

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[""''\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def detect_language(title):
    return "english"

# ===== NEW: ROBUST PATTERN SCORING FUNCTION =====
def calculate_analytical_score(title):
    """Score titles based on crisis/analysis/geopolitical patterns."""
    score = 0
    title_lower = title.lower()
    
    # CRISIS/CONFLICT INDICATORS (very strong signals)
    crisis_terms = [
        'war', 'conflict', 'crisis', 'collapse', 'violence', 'attack', 'strikes',
        'invasion', 'sanctions', 'blockade', 'ceasefire', 'offensive', 'bombing',
        'insurgency', 'coup', 'protests', 'unrest', 'genocide', 'massacre',
        'refugees', 'displacement', 'humanitarian', 'famine', 'drought', 'terror',
        'clashes', 'shelling', 'uprising', 'hostilities', 'tensions', 'retaliation',
        'raid', 'mobilization', 'escalation', 'hostage', 'airstrike', 'militia',
        'civil war', 'rebels', 'armed group'
    ]
    crisis_count = sum(1 for term in crisis_terms if term in title_lower)
    score += min(crisis_count * 2, 6)  # Max 6 points from crisis terms
    
    # GEOPOLITICAL RELATIONS (country-country interactions)
    relation_patterns = ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']
    if any(pattern in title_lower for pattern in relation_patterns):
        score += 2
    
    # POSSESSIVE STRUCTURES (country's/region's X)
    if "'s " in title or "'" in title:
        score += 2
    
    # ANALYTICAL QUESTION WORDS
    question_starters = [
        'why ', 'how ', 'what ', 'can ', 'will ', 'should ', 'is ', 'are ', 
        'do ', 'does ', 'could ', 'may ', 'might ', 'would '
    ]
    if any(title_lower.startswith(q) for q in question_starters):
        score += 2
    
    # GEOPOLITICAL/ECONOMIC KEYWORDS
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
    geo_count = sum(1 for term in geo_terms if term in title_lower)
    score += min(geo_count, 3)  # Max 3 points from geo terms
    
    # OUTCOME/IMPACT LANGUAGE (action verbs)
    outcome_terms = [
        'faces', 'threatens', 'undermines', 'deepens', 'escalates', 'intensifies',
        'worsens', 'persists', 'continues', 'remains', 'struggles', 'fails',
        'succeeds', 'achieves', 'drives', 'fuels', 'prevents', 'blocks', 'enables',
        'forces', 'triggers', 'sparks', 'creates', 'causes', 'leads to', 'results in',
        'complicates', 'reshapes', 'transforms', 'revives', 'shifts', 'redefines',
        'contributes to', 'signals', 'marks', 'heightens', 'deteriorates'
    ]
    if any(term in title_lower for term in outcome_terms):
        score += 1
    
    # EXPLANATORY STRUCTURES
    explanatory = [
        'reveals', 'shows', 'exposes', 'means for', 'means to', 'impact on', 'impact of',
        'effect on', 'effect of', 'implications', 'consequences', 'amid', 'despite',
        'after', 'before', 'during', 'following', 'since', 'until',
        'significance', 'context', 'analysis', 'explained', 'lessons from'
    ]
    if any(exp in title_lower for exp in explanatory):
        score += 1
    
    # COMPARATIVE/DILEMMA STRUCTURES
    comparative = [
        'between', 'versus', 'against', 'compared to', 'the paradox', 
        'the dilemma', 'the challenge', 'the struggle', 'crossroads', 'balance of power'
    ]
    if any(comp in title_lower for comp in comparative):
        score += 1
    
    # FUTURE/PREDICTIVE LANGUAGE
    future = [
        'future of', 'will ', 'could ', 'may ', 'might ', 'outlook', 'prospects',
        'next phase', 'ahead', 'forecast', 'trajectory'
    ]
    if any(fut in title_lower for fut in future):
        score += 1
    
    # REGIONAL/COUNTRY NAMES (indicates geopolitical focus)
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
    if any(region in title_lower for region in regions):
        score += 1
    
    # NEGATIVE INDICATORS (reduce score for pure news/events)
    news_indicators = [
        'announces', 'launches', 'opens', 'celebrates', 'wins award', 'ceremony',
        'appointed today', 'signs deal today', 'breaks record', 'new product',
        'birthday', 'anniversary celebration', 'gala', 'festival begins',
        'sports', 'match', 'tournament', 'cup', 'music', 'film', 'concert',
        'startup', 'company', 'brand', 'fashion', 'award show'
    ]
    if any(indicator in title_lower for indicator in news_indicators):
        score -= 3
    
    return score

def pubdate_to_minutes(pubdate_str):
    try:
        dt = parser.parse(pubdate_str, fuzzy=True)
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        return int(dt.timestamp() // 60)
    except Exception:
        return 0

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
MODEL_PATH = "models/all-mpnet-base-v2"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run workflow to download first.")

model = SentenceTransformer(MODEL_PATH)
ref_embeddings = model.encode(REF_TITLES) if REF_TITLES else None

# ===== FILTER ARTICLES BY HYBRID SCORING =====
filtered_articles = []
eligible_titles = []

for article in feed_articles:
    title_clean = clean_title(article["title"])
    
    # Calculate pattern score
    pattern_score = calculate_analytical_score(title_clean)
    
    # Calculate similarity score
    if ref_embeddings is not None and ref_embeddings.size > 0:
        emb = model.encode([title_clean])
        sim_scores = cosine_similarity(emb, ref_embeddings)
        max_similarity = sim_scores.max()
    else:
        max_similarity = 0.0
    
    # HYBRID DECISION LOGIC
    accept = False
    if max_similarity >= ENGLISH_THRESHOLD:
        accept = True
    elif pattern_score >= 7 and max_similarity >= 0.33:
        accept = True
    elif pattern_score >= 5 and max_similarity >= 0.38:
        accept = True
    
    if accept:
        filtered_articles.append(article)
        if title_clean not in REF_TITLES:
            eligible_titles.append((title_clean, article.get("feed_source", "unknown"), "english"))

# ===== FILTER BY LAST 36 HOURS =====
cutoff_minutes = int((datetime.utcnow() - timedelta(hours=CUT_OFF_HOURS)).timestamp() // 60)
recent_articles = []
for a in filtered_articles:
    minutes = pubdate_to_minutes(a.get("published", ""))
    if minutes > cutoff_minutes:
        recent_articles.append(a)

filtered_articles = recent_articles

# ===== SORT BY PUBLISH DATE DESC =====
filtered_articles.sort(key=lambda x: pubdate_to_minutes(x.get("published", "")), reverse=True)

# ===== LOAD EXISTING XML =====
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

# ===== ADD NEW ITEMS =====
for a in reversed(filtered_articles):
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

# ===== WRITE XML =====
ET.ElementTree(root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)

# ===== UPDATE REFERENCE TITLES =====
if len(REF_TITLES) < REFERENCE_MAX:
    random.shuffle(eligible_titles)
    to_add = eligible_titles[:MAX_NEW_TITLES]
    if to_add:
        with open(REFERENCE_FILE, "a", encoding="utf-8") as f:
            for t, _, _ in to_add:
                f.write(t + "\n")
