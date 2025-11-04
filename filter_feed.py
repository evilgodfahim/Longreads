import feedparser
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from datetime import datetime, timedelta
from dateutil import parser

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
REFERENCE_FILE = "reference_titles.txt"
OUTPUT_FILE = "filtered.xml"
ENGLISH_THRESHOLD = 0.60
MAX_XML_ITEMS = 500
CUT_OFF_HOURS = 36
DESCRIPTION_BOOST = 0.15  # Bonus similarity points if description also matches (0-0.3 recommended)

# ===== UTILS =====
def clean_title(t):
    t = t.strip()
    t = re.sub(r"[""''\"']", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def clean_description(d):
    """Clean description: remove HTML tags, extra whitespace"""
    if not d:
        return ""
    # Remove HTML tags
    d = re.sub(r'<[^>]+>', '', d)
    # Remove extra whitespace
    d = re.sub(r'\s+', ' ', d)
    return d.strip()

def detect_language(title):
    return "english"

# ===== ROBUST PATTERN SCORING FUNCTION =====
def calculate_analytical_score(text):
    """Score text based on crisis/analysis/geopolitical patterns."""
    score = 0
    text_lower = text.lower()

    # CRISIS/CONFLICT INDICATORS
    crisis_terms = [
        'war', 'conflict', 'crisis', 'collapse', 'violence', 'attack', 'strikes',
        'invasion', 'sanctions', 'blockade', 'ceasefire', 'offensive', 'bombing',
        'insurgency', 'coup', 'protests', 'unrest', 'genocide', 'massacre',
        'refugees', 'displacement', 'humanitarian', 'famine', 'drought', 'terror',
        'clashes', 'shelling', 'uprising', 'hostilities', 'tensions', 'retaliation',
        'raid', 'mobilization', 'escalation', 'hostage', 'airstrike', 'militia',
        'civil war', 'rebels', 'armed group'
    ]
    crisis_count = sum(1 for term in crisis_terms if term in text_lower)
    score += min(crisis_count * 2, 6)

    # GEOPOLITICAL RELATIONS
    relation_patterns = ['-', ' v ', ' vs ', ' versus ', ' and ', ' with ']
    if any(pattern in text_lower for pattern in relation_patterns):
        score += 2

    # POSSESSIVE STRUCTURES
    if "'s " in text or "'" in text:
        score += 2

    # ANALYTICAL QUESTION WORDS
    question_starters = [
        'why ', 'how ', 'what ', 'can ', 'will ', 'should ', 'is ', 'are ', 
        'do ', 'does ', 'could ', 'may ', 'might ', 'would '
    ]
    if any(text_lower.startswith(q) for q in question_starters):
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
    geo_count = sum(1 for term in geo_terms if term in text_lower)
    score += min(geo_count, 3)

    # OUTCOME/IMPACT LANGUAGE
    outcome_terms = [
        'faces', 'threatens', 'undermines', 'deepens', 'escalates', 'intensifies',
        'worsens', 'persists', 'continues', 'remains', 'struggles', 'fails',
        'succeeds', 'achieves', 'drives', 'fuels', 'prevents', 'blocks', 'enables',
        'forces', 'triggers', 'sparks', 'creates', 'causes', 'leads to', 'results in',
        'complicates', 'reshapes', 'transforms', 'revives', 'shifts', 'redefines',
        'contributes to', 'signals', 'marks', 'heightens', 'deteriorates'
    ]
    if any(term in text_lower for term in outcome_terms):
        score += 1

    # EXPLANATORY STRUCTURES
    explanatory = [
        'reveals', 'shows', 'exposes', 'means for', 'means to', 'impact on', 'impact of',
        'effect on', 'effect of', 'implications', 'consequences', 'amid', 'despite',
        'after', 'before', 'during', 'following', 'since', 'until',
        'significance', 'context', 'analysis', 'explained', 'lessons from'
    ]
    if any(exp in text_lower for exp in explanatory):
        score += 1

    # COMPARATIVE/DILEMMA STRUCTURES
    comparative = [
        'between', 'versus', 'against', 'compared to', 'the paradox', 
        'the dilemma', 'the challenge', 'the struggle', 'crossroads', 'balance of power'
    ]
    if any(comp in text_lower for comp in comparative):
        score += 1

    # FUTURE/PREDICTIVE LANGUAGE
    future = [
        'future of', 'will ', 'could ', 'may ', 'might ', 'outlook', 'prospects',
        'next phase', 'ahead', 'forecast', 'trajectory'
    ]
    if any(fut in text_lower for fut in future):
        score += 1

    # REGIONAL/COUNTRY NAMES
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
    if any(region in text_lower for region in regions):
        score += 1

    # NEGATIVE INDICATORS
    news_indicators = [
        'announces', 'launches', 'opens', 'celebrates', 'wins award', 'ceremony',
        'appointed today', 'signs deal today', 'breaks record', 'new product',
        'birthday', 'anniversary celebration', 'gala', 'festival begins',
        'sports', 'match', 'tournament', 'cup', 'music', 'film', 'concert',
        'startup', 'company', 'brand', 'fashion', 'award show'
    ]
    if any(indicator in text_lower for indicator in news_indicators):
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
        
        # Get description/summary if available
        description = ""
        if hasattr(entry, "description"):
            description = entry.description
        elif hasattr(entry, "summary"):
            description = entry.summary
        
        feed_articles.append({
            "title": entry.title,
            "description": description,
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

for article in feed_articles:
    title_clean = clean_title(article["title"])
    desc_clean = clean_description(article["description"])
    
    # Calculate pattern score on TITLE ONLY (primary signal)
    pattern_score = calculate_analytical_score(title_clean)
    
    # Calculate similarity score on title primarily
    if ref_embeddings is not None and ref_embeddings.size > 0:
        # Get title similarity (main score)
        title_emb = model.encode([title_clean])
        max_similarity = cosine_similarity(title_emb, ref_embeddings).max()
        
        # Optional: Add small boost if description also matches
        if desc_clean and max_similarity >= 0.45:  # Only check desc if title is promising
            desc_emb = model.encode([desc_clean[:400]])
            desc_sim = cosine_similarity(desc_emb, ref_embeddings).max()
            if desc_sim >= 0.65:  # Description confirms relevance
                max_similarity = min(max_similarity + DESCRIPTION_BOOST, 1.0)
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
    # Optionally include description in output
    if a["description"]:
        ET.SubElement(item, "description").text = a["description"]
    channel.insert(0, item)
    existing_titles.add(t)

# ===== LIMIT XML ITEMS =====
all_items = channel.findall("item")
if len(all_items) > MAX_XML_ITEMS:
    for item in all_items[MAX_XML_ITEMS:]:
        channel.remove(item)

# ===== WRITE XML =====
ET.ElementTree(root).write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)