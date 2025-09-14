import re
import math
import operator
from typing import List, Dict, Any
import pandas as pd
import emoji
from langdetect import detect
from googletrans import Translator
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import en_core_web_sm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

import pandas as pd
import emoji
from langdetect import detect
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import en_core_web_sm
from googletrans import Translator

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# --- Paths to datasets (Please change these 2 lines!) ---
file_path = ".../videos.csv"
file_path_2 = ".../comments1.csv"

# --- Read datasets ---
df = pd.read_csv(file_path)
df2 = pd.read_csv(file_path_2)

# --- Filter for one video ---
df = df[df['videoId'] == 39559]
df2 = df2[df2['videoId'] == 39559]

import pandas as pd
from googletrans import Translator

translator = Translator()

# Translate function: full sentence first, fallback to word-by-word
def translate_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return text
    try:
        # 1️⃣ Full sentence translation
        translated = translator.translate(text, src='auto', dest='en').text

        # 2️⃣ Fallback word-by-word if nothing changed
        if translated.strip().lower() == text.strip().lower():
            words = text.split()
            translated_words = []
            for w in words:
                try:
                    tw = translator.translate(w, src='auto', dest='en').text
                    translated_words.append(tw)
                except:
                    translated_words.append(w)
            translated = " ".join(translated_words)

        return translated
    except:
        return text

# Step 2: Translate normalized text
df["Translated_title"] = df["title"].apply(translate_text)
df["Translated_description"] = df["description"].apply(translate_text)
df2["Translated_textOriginal"] = df2["textOriginal"].apply(translate_text)

# NLP tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()
nlp = en_core_web_sm.load()  # spaCy NER model

# --- Recover emojis ---
def recover_emojis(text):
    if not isinstance(text, str):
        return ""
    for enc in ["latin1", "cp1252"]:
        try:
            fixed = text.encode(enc).decode("utf-8")
            return fixed.replace("\n", " ")
        except Exception:
            continue
    return text.replace("\n", " ")

# --- NLP preprocessing ---
def nlp_preprocess(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = str(text)
    tokens = tokenizer.tokenize(text)

    if not tokens:
        return ""


    # Lowercase
    tokens = [token.lower() for token in tokens if token is not None]
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming and lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(lemmatized_tokens)

# --- NER extraction ---
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def filter_entities_by_type(entities, entity_type):
    return [ent_text for ent_text, ent_label in entities if ent_label == entity_type]

# --- Full processing + NER pipeline ---
def process_and_analyze(df, colname, processed_colname, entities_colname):
    if colname in df.columns:
        df[processed_colname] = (
            df[colname]
            .apply(recover_emojis)
            .apply(lambda x: emoji.demojize(x, language="en") if isinstance(x, str) else "")
            .apply(nlp_preprocess)
        )
        # Extract NER from processed text
        df[entities_colname] = df[processed_colname].apply(extract_entities)

# --- Apply NER preprocessing ---
process_and_analyze(df, "Translated_title", "processed_title", "title_entities")
process_and_analyze(df, "Translated_description", "processed_description", "description_entities")
process_and_analyze(df2, "Translated_textOriginal", "processed_textOriginal", "text_entities")

# --- Show previews ---
print("Videos file preview (processed + slang norm + NER + GoogleTrans):")
print(df[["processed_title", "title_entities","processed_description", "description_entities"]].head(10))

print("\nComments file preview (processed + slang norm + NER + GoogleTrans):")
print(df2[["processed_textOriginal", "text_entities"]].head(10))

# -----------------------
# Custom Concept Utilities
# -----------------------

def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    return str(s).lower()

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def find_phrase_positions(tokens: List[str], phrase: str) -> List[int]:
    phrase_tokens = re.findall(r"\w+", phrase.lower())
    if len(phrase_tokens) == 0:
        return []
    end_len = len(tokens)
    window_len = len(phrase_tokens)
    limit = operator.add(operator.sub(end_len, window_len), 1)
    positions = []
    for i in range(max(0, limit)):
        if tokens[i:i + window_len] == phrase_tokens:
            positions.append(i)
    return positions

def contains_any(text: str, phrases: List[str]) -> List[str]:

    found = []
    text_l = text.lower()
    for p in phrases:
        if re.search(r"\b" + re.escape(p.lower()) + r"\b", text_l):
            found.append(p)
    return found

def not_contains_any(text: str, phrases: List[str]) -> bool:
    return not contains_any(text, phrases)

def within_distance(tokens: List[str], phrases_a: List[str], phrases_b: List[str], n: int) -> bool:
    for a in phrases_a:
        pos_a = find_phrase_positions(tokens, a)
        if not pos_a:
            continue
        for b in phrases_b:
            pos_b = find_phrase_positions(tokens, b)
            if not pos_b:
                continue
            for i in pos_a:
                for j in pos_b:
                    if math.fabs(operator.sub(i, j)) <= n:
                        return True
    return False

# Concept terms dictionary
concept_terms: Dict[str, List[str]] = {
    "brush": ["brush", "brushes"], "lip": ["lip"], "balm": ["balm"], "gloss": ["gloss"],
    "butter": ["butter"], "oil": ["oil"], "tint": ["tint"], "stick": ["stick"],
    "glam": ["glam"], "eye": ["eye", "eyes"], "serum": ["serum"], "essence": ["essence"],
    "scar": ["scar"], "glow": ["glow"], "age": ["age", "aging"], "dewy": ["dewy"],
    "sensitive": ["sensitive"], "oily": ["oily"], "dry": ["dry"], "combination": ["combination"],
    "combo": ["combo"], "flawless": ["flawless"], "beautiful": ["beautiful"], "patch": ["patch"],
    "dull": ["dull"], "soft": ["soft"], "plump": ["plump"], "radiant": ["radiant"],
    "korean": ["korean"], "japanese": ["japanese"], "western": ["western"], "asian": ["asian"],
    "chinese": ["chinese"], "recommend": ["recommend", "recommended"],
    "hydrate": ["hydrate", "hydrating", "hydration"], "mask": ["mask", "masks"],
    "moisturise": ["moisturise", "moisturize"], "exfoliate": ["exfoliate", "exfoliating"],
    "cream": ["cream", "creams"], "vitamin c": ["vitamin c"], "face": ["face"],
    "wrinkle": ["wrinkle", "wrinkles"], "fine line": ["fine line"], "tone": ["tone"],
    "redness": ["redness"], "pore": ["pore"], "blackhead": ["blackhead"], "whitehead": ["whitehead"],
    "sunburn": ["sunburn"], "allergic": ["allergic"], "irritated": ["irritated"],
    "reactive": ["reactive"], "collagen": ["collagen"]
}


# List of brands (from CLASSIFIER rules you provided earlier)
brands = [
    "L'Oreal","loreal","Atelier Cologne","Lancôme","lancome","Yves Saint Laurent Beauté","YSL",
    "Giorgio Armani Beauty","armani","Viktor&Rolf","Mugler Parfums","Valentino Beauty","Valentino",
    "Aesop","Maybelline New York","Maybelline","L’Oréal Paris","NYX Professional Makeup","nyx",
    "Urban Decay","Shu Uemura","IT Cosmetics","Kiehl’s","kiehls","Biotherm","Vichy","La Roche-Posay",
    "la roche posay","SkinCeuticals","Helena Rubinstein","Garnier","CeraVe","Dr.G","dr g","Medic8",
    "Estée Lauder","Revlon","CoverGirl","Neutrogena","Olay","Nivea","Rimmel","E.l.f. Cosmetics",
    "Clinique","Sephora Collection","Bourjois","Ponds","Max Factor","Physicians Formula","Avon",
    "Wet n Wild","Shiseido","Clarins","Dior Beauty","Chanel Beauty","La Mer","Givenchy","Sisley",
    "SK-II","Elizabeth Arden","Guerlain","Bobbi Brown","Origins","Tom Ford Beauty","Aveeno",
    "Herbal Essences","John Frieda","Suave","OGX","The Body Shop","Bioderma","Avene","Cetaphil",
    "Eucerin","Paula’s Choice","The Ordinary","Murad","Dermalogica","Obagi","EltaMD","ROC",
    "Cosrx","Dr. Jart+","Gucci Beauty","Prada Beauty","Hermes Parfums","Burberry Beauty",
    "Dolce & Gabbana Beauty","Marc Jacobs Beauty","Maison Francis Kurkdjian","Byredo","Jo Malone",
    "Creed","Narciso Rodriguez","Versace Beauty","L’Occitane","Caudalie","Sunday Riley",
    "Tata Harper","Drunk Elephant","Jurlique","Milani","Sleek Makeup","Makeup Revolution",
    "Catrice","KIKO Milano","Hard Candy","BoxyCharm","Vanicream","Curel","Mustela","Bvlgari",
    "Montblanc","Paco Rabanne","Le Labo","Diptyque","Augustinus Bader","REN","Herbivore",
    "Malin+Goetz","Le Pere Pelletier","Lush","Biossance","Pai","Odacité","Sensai","Sulwhasoo",
    "Natura Bissé","ColourPop","Morphe","Anastasia Beverly Hills","Huda Beauty","Fenty","Benefit",
    "BH Cosmetics","Inglot","KVD Cosmetics","L.A. Girl","Elemis","ZO Skin Health","PCA Skin",
    "IS Clinical","Medik8","Dr. Dennis Gross","NeoStrata","SkinMedica","Jan Marini","Circadia",
    "Osmosis","Revision","Cosmedix","Alastin","Ultraceuticals","Dolce & Gabbana","Moschino",
    "Cartier","Calvin Klein","Jean Paul Gaultier","Carolina Herrera","Roja Parfums","Bond No.9",
    "Montale","Mancera","Kilian","Serge Lutens","Acqua di Parma","Penhaligon’s","Parfums de Marly",
    "Comme des Garçons","Escentric Molecules","Too Faced","NARS","Tarte","Kat Von D","Stila",
    "Smashbox","Pat McGrath Labs","Lime Crime","etude","Innisfree","Peripera","Missha",
    "Beauty Bakerie","Too Cool For School","Faces Canada","The Face Shop","Holika Holika","Romand",
    "A-Derma","Weleda","Johnson’s Baby","Sebamed","Chicco","Klorane","Bepanthen","Baby Dove",
    "Pigeon","Hada Labo","DHC","Suqqu","Clé de Peau","Tatcha","Dr. Ci:Labo","Rohto","Albion",
    "Koh Gen Do","Kanebo","Kosé","FANCL","HABA","RMK","Kosé Sekkisei","Decorté","Chloe",
    "Rare Beauty","Charlotte Tilbury","elf cosmetics","Summer Fridays","tower28","tower 28",
    "vaseline","beauty of joseon","boj","Jimmy Choo","super goop","Laura Mercier","Loewe",
    "Hourglass","Glossier","Milk Makeup","Ilia","Westman Atelier","Kylie Cosmetics","KKW Beauty",
    "Fenty Skin","Sephora","Make Up For Ever","Kevyn Aucoin","Becca","Juicy Couture","Burberry",
    "Hollister","Lacoste","Michael Kors","Victoria’s Secret","Laneige","Hera","Etude House","3CE",
    "Amorepacific","Banila Co","Clio","Espoir","Mamonde","Hanskin","Huxley","Canmake","Kate Tokyo",
    "Majolica Majorca","POLA","Anessa","Florasis","Perfect Diary","Judydoll","Chando","Carslan",
    "Marubi","Herborist","Pechoin","Winona","ZEESEA","PROYA","Inoherb","Colorkey","Kans",
    "One Leaf","Joyme","MG Mask","Yue Sai","Catkin","Longrich","rhode","haus lab@","r.e.m",
    "rem beauty","AAVRANI","AERIN","ALGENIST","ALPYN","AMIKA","Ami Colé","Aquis","Artist Couture",
    "BeautyBio","bareMinerals","Bondi Sands","Briogeo","BYOMA","By Terry","Chantecaille","COOLA",
    "Colour Wow","Commodity","Curlsmith","Danessa Myricks Beauty","Davines","Drybar","Dyson",
    "Eve Lom","Farmacy","First Aid Beauty","Frank Body","Glow Recipe","Goodal","Heimish","I'M FROM",
    "iUNIK","The INKEY List","Isntree","Klairs","Kosas","L:A BRUKET","La Prairie","Manyo",
    "Melt Cosmetics","Merit","Olaplex","Ouai","Oribe","Paul Mitchell","Perricone MD","Playa","R+Co",
    "Saie","Schwarzkopf Professional","Sol de Janeiro","Tangle Teezer","Thayers","Topicals","Verb",
    "Wander Beauty","Wella Professionals","Zoeva","Dr. Althea","Elizavecca","Jumiso","Klavuu",
    "Make P:rem","Mizon","Round Lab","Secret Key","Some By Mi","SNP","Son & Park","VT Cosmetics",
    "KEEP COOL","GHD","Hot Tools","InStyler"
]

# Classifiers
fragrance_terms = [
    "fragrance", "perfume", "mist", "smell", "spray", "scent",
    "edp", "deo", "deodorant", "eau de parfum",
    # fragrance-only brands
    "Atelier Cologne","Viktor&Rolf","Mugler Parfums","Maison Francis Kurkdjian",
    "Byredo","Jo Malone","Creed","Narciso Rodriguez","Bvlgari","Montblanc",
    "Paco Rabanne","Le Labo","Diptyque","Roja Parfums","Bond No.9","Montale",
    "Mancera","Kilian","Serge Lutens","Acqua di Parma","Penhaligon’s",
    "Parfums de Marly","Comme des Garçons","Escentric Molecules","Chloe",
    "Jimmy Choo","Juicy Couture"
]

hair_terms = [
    "hair", "hair routine", "serum", "mist", "spray",
    "smell", "scent", "scalp", "shampoo", "conditioner"
]

makeup_classifiers = [
    "makeup", "make up", "foundation", "bronzer", "powder",
    "lip", "gloss", "contour", "primer", "concealer",
    "blush", "highlighter", "eyeshadow", "eyeliner",
    "mascara", "brush", "get ready with me",
    # makeup-only brands
    "Maybelline","NYX Professional Makeup","Urban Decay","IT Cosmetics","Revlon",
    "CoverGirl","Rimmel","E.l.f. Cosmetics","Sephora Collection","Bourjois",
    "Max Factor","Physicians Formula","Avon","Wet n Wild","Milani","Sleek Makeup",
    "Makeup Revolution","Catrice","KIKO Milano","Hard Candy","BoxyCharm",
    "ColourPop","Morphe","Anastasia Beverly Hills","Huda Beauty","Fenty",
    "Benefit","BH Cosmetics","Inglot","KVD Cosmetics","L.A. Girl","Too Faced",
    "NARS","Tarte","Kat Von D","Stila","Smashbox","Pat McGrath Labs","Lime Crime",
    "Beauty Bakerie","Too Cool For School","Faces Canada","Rare Beauty",
    "Charlotte Tilbury","elf cosmetics","tower28","tower 28","Laura Mercier",
    "Hourglass","Glossier","Milk Makeup","Ilia","Westman Atelier",
    "Kylie Cosmetics","KKW Beauty","Sephora","Make Up For Ever","Kevyn Aucoin",
    "Becca","Etude","Etude House","Innisfree","Peripera","Missha",
    "The Face Shop","Holika Holika","Romand","3CE","Banila Co","Clio","Espoir",
    "Mamonde","Hanskin","Huxley","Canmake","Kate Tokyo","Majolica Majorca",
    "Florasis","Perfect Diary","Judydoll","Chando","Carslan","Marubi","Herborist",
    "Pechoin","Winona","ZEESEA","PROYA","Inoherb","Colorkey","Kans","One Leaf",
    "Joyme","Catkin","Longrich","rhode","haus lab@","r.e.m","rem beauty",
    "AAVRANI","AERIN","Ami Colé","Artist Couture","By Terry","Chantecaille",
    "Danessa Myricks Beauty","Kosas","Melt Cosmetics","Merit","Saie",
    "Wander Beauty","Zoeva"
]

skincare_classifiers = [
    "skincare", "sunscreen", "spf", "moisturiser", "toner",
    "retinol", "acne", "cleanser", "cream", "face", "serum",
    "get ready with me", "get unready with me","suncream",
    # skincare-only brands
    "Aesop","Kiehl’s","Biotherm","Vichy","La Roche-Posay","SkinCeuticals",
    "CeraVe","Dr.G","Medik8","Aveeno","The Body Shop","Bioderma","Avene",
    "Cetaphil","Eucerin","Paula’s Choice","The Ordinary","Murad","Dermalogica",
    "Obagi","EltaMD","ROC","Cosrx","Dr. Jart+","L’Occitane","Caudalie",
    "Sunday Riley","Tata Harper","Drunk Elephant","Jurlique","Vanicream",
    "Curel","Mustela","Augustinus Bader","REN","Herbivore","Malin+Goetz",
    "Le Pere Pelletier","Lush","Biossance","Pai","Odacité","ZO Skin Health",
    "PCA Skin","IS Clinical","Dr. Dennis Gross","NeoStrata","SkinMedica",
    "Jan Marini","Circadia","Osmosis","Revision","Cosmedix","Alastin",
    "Ultraceuticals","A-Derma","Weleda","Johnson’s Baby","Sebamed","Chicco",
    "Klorane","Bepanthen","Baby Dove","Pigeon","Hada Labo","DHC","Tatcha",
    "Dr. Ci:Labo","Rohto","Albion","FANCL","HABA","Summer Fridays",
    "beauty of joseon","super goop","Fenty Skin","Glow Recipe","Goodal",
    "Heimish","I'M FROM","iUNIK","The INKEY List","Isntree","Klairs",
    "L:A BRUKET","La Prairie","Manyo","Perricone MD","Thayers","Topicals",
    "Dr. Althea","Elizavecca","Jumiso","Klavuu","Make P:rem","Mizon",
    "Round Lab","Secret Key","Some By Mi","SNP","Son & Park","VT Cosmetics",
    "KEEP COOL"
]

def has_concept_token(text: str, concept_key: str) -> bool:
    if concept_key not in concept_terms:
        return False
    return contains_any(text, concept_terms[concept_key])

def classify_text_with_terms(text_raw: Any) -> Dict[str, Any]:
    text = normalize_text(text_raw)
    result = {"category": "other", "detected_terms": []}
    detected = []

    fragrance_hits = contains_any(text, fragrance_terms)
    hair_hits = contains_any(text, hair_terms)
    makeup_hits = contains_any(text, makeup_classifiers)
    skincare_hits = contains_any(text, skincare_classifiers)

    if fragrance_hits:
        result["category"] = "fragrance"
        detected.extend(fragrance_hits)
    if hair_hits:
        result["category"] = "hair"
        detected.extend(hair_hits)
    if makeup_hits:
        result["category"] = "makeup"
        detected.extend(makeup_hits)
    if skincare_hits:
        result["category"] = "skincare"
        detected.extend(skincare_hits)

    if not detected:
        detected.append("none")

    result["detected_terms"] = list(set(detected))  # unique terms
    return result

def classify_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    flags_list = [classify_text_with_terms(t) for t in df[text_column].fillna("").tolist()]
    flags_df = pd.DataFrame(flags_list)
    combined = pd.concat([df.reset_index(drop=True), flags_df.reset_index(drop=True)], axis=1)

    # keep only skincare, makeup, fragrance
    combined = combined[combined["category"].isin(["skincare", "makeup", "fragrance"])]
    return combined


# --- Apply category classification ---
df2["category_flags"] = df2["processed_textOriginal"].apply(classify_text_with_terms)
df2["category"] = df2["category_flags"].apply(lambda x: x["category"])
df2["detected_terms"] = df2["category_flags"].apply(lambda x: ", ".join(x["detected_terms"]))

# filter to only skincare, makeup, fragrance
df2 = df2[df2["category"].isin(["skincare", "makeup", "fragrance"])]

# Precompile regex patterns
brand_patterns = {b: re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE) for b in brands}

# Function to find all brands in a comment
def find_brands(text):
    if not isinstance(text, str):
        return []
    matches = [brand for brand, pattern in brand_patterns.items() if pattern.search(text)]
    return matches

# Apply to comments DataFrame
df2["brands_detected"] = df2["processed_textOriginal"].apply(find_brands)

print("Sample brand detections:")
print(df2[["processed_textOriginal", "brands_detected"]].head(20))

# Sentiment analysis with simple emoji cleaning
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def clean_for_sentiment(text):
    if not isinstance(text, str):
        return ""
    # replace underscore and hyphen with a space
    return text.replace("_", " ").replace("-", " ")

def get_sentiment_from_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    clean = clean_for_sentiment(text)
    scores = sia.polarity_scores(clean)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

df2["sentiment_text"] = df2["processed_textOriginal"].astype(str).apply(clean_for_sentiment)
df2["sentiment"] = df2["sentiment_text"].apply(get_sentiment_from_text)

# --- Final Output ---
print("\n===== Final Classified & Analyzed Comments =====")
print(df2[[
    "processed_textOriginal",
    "category",
    "detected_terms",
    "brands_detected",
    "sentiment"
]].head(50))  

# --- Spam Detection ---
SPAM_REGEXES = [
    re.compile(r"^\s*(?:https?:\/\/[^\s]+|www\.[^\s]+|[A-Za-z0-9.-]+\.(?:com|net|org|io|co|info|biz|top|site|xyz)(?:\/[^\s])?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly)(?:\/[^\s])?\s*$", re.IGNORECASE)
]

SPAM_KEYWORDS = [
    "win","only","offer","off","click","claim","prize","left","limited","time",
    "jesus","bible","commandment","resurrection","christianity","scripture","gospel"
]

SPAM_SWEAR_PATTERNS = [
    re.compile(r"f+[\W_]*u+[\W_]*c+[\W_]*k+", re.IGNORECASE),
    re.compile(r"s+[\W_]*h+[\W_]*i+[\W_]*t+", re.IGNORECASE),
    re.compile(r"b+[\W_]*i+[\W_]*t+[\W_]*c+[\W_]*h+", re.IGNORECASE),
    re.compile(r"a+[\W_]*s+[\W_]*s+[\W_]*h*[\W_]*o*[\W_]*l+[\W_]*e*", re.IGNORECASE)
]

SPAM_EMOJI_LIST = ["😂","👇","🤣","❤️","😍","🔥","💯","👉","💖","✨","⭐","🎉","🚀","😭","👍","😅"]

def is_full_swear_comment(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return False
    for tok in tokens:
        if not any(p.match(tok) for p in SPAM_SWEAR_PATTERNS):
            return False
    return True

def is_spam_row(row):
    text_processed = row.get("processed_textOriginal", "")
    text_raw = row.get("textOriginal", "")

    # --- Checks on processed text (cleaned) ---
    t = str(text_processed).strip().lower()
    if any(r.match(t) for r in SPAM_REGEXES):
        return True
    if any(word in t for word in SPAM_KEYWORDS):
        return True
    if is_full_swear_comment(text_processed):
        return True
    if len(str(text_processed).split()) > 75:
        return True

    # --- Emoji spam check on raw text (keep emojis) ---
    if isinstance(text_raw, str):
        if len([c for c in text_raw if c in SPAM_EMOJI_LIST]) > 5:
            return True

    return False

df2["spam"] = df2.apply(is_spam_row, axis=1)
print(df2[["processed_textOriginal", "sentiment", "spam"]])

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load XLM-R SentenceTransformer (pre-finetuned for semantic similarity)
model_name = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
embedder = SentenceTransformer(model_name)

# Embedding function (handles empty strings)
def get_sentence_embedding(text, embedding_dim=768):
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(embedding_dim)
    return embedder.encode(text)

# Embed your dataframe column
df['title_embeddings'] = df['processed_title'].apply(get_sentence_embedding)
df['description_embeddings'] = df['processed_description'].apply(get_sentence_embedding)
df2['text_embeddings'] = df2['processed_textOriginal'].apply(get_sentence_embedding)

# Suppose your dataframe is df2 and text_embeddings is already a column of lists
X = np.vstack(df2['text_embeddings'].values)

# Range of k to try
k_values = range(1, 11)  # from 1 to 10 clusters
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    print(f"k={k}, inertia={kmeans.inertia_:.2f}")

# Plot the elbow curve
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_values, 'o-', color='blue')
plt.xticks(k_values)
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# Try clustering with k=2...10
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"k={k}, silhouette_score={score:.4f}")

import matplotlib.cm as cm

k = 2  # choose based on above
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)

sample_silhouette_values = silhouette_samples(X, labels)

y_lower = 10
for i in range(k):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / k)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title(f"Silhouette plot for k={k}")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
plt.show()

# Choose number of clusters
num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df2["cluster"] = kmeans.fit_predict(X)

# Show cluster assignments
print(df2[["processed_textOriginal", "cluster"]].head(20))

# Preview comments by cluster
for c in range(num_clusters):
    print(f"\nCluster {c}:")
    sample = df2[df2["cluster"] == c]["processed_textOriginal"]
    for comment in sample:
        print("  -", comment)

#Plot for clustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Run t-SNE (make sure perplexity < 29 since you have 29 samples)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(X)   # X = stacked embeddings

# Create scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(reduced[:,0], reduced[:,1], c=df2["cluster"], cmap="tab10", alpha=0.7)

# Annotate each point with cluster number
for i, txt in enumerate(df2["processed_textOriginal"]):
    cluster_label = df2.iloc[i]["cluster"]   # <-- iloc fixes the KeyError
    plt.annotate(str(cluster_label),
                 (reduced[i,0], reduced[i,1]),
                 fontsize=8, alpha=0.6)

plt.title("t-SNE Visualization of Comment Clusters (k=2)")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.show()

#Keywords
from sklearn.feature_extraction.text import TfidfVectorizer

# Reset index to match tfidf row positions
df2 = df2.reset_index(drop=True)

# Build TF-IDF matrix on processed text
vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df2["processed_textOriginal"])
terms = vectorizer.get_feature_names_out()

def top_keywords_per_cluster(n_terms=7):
    cluster_keywords = {}
    for c in sorted(df2["cluster"].unique()):
        # get row indices for this cluster
        idxs = df2[df2["cluster"] == c].index.tolist()

        # average TF-IDF for the cluster
        cluster_tfidf = tfidf_matrix[idxs].mean(axis=0).A1

        # pick top terms
        top_idx = cluster_tfidf.argsort()[-n_terms:][::-1]
        cluster_keywords[c] = [terms[i] for i in top_idx]
    return cluster_keywords

keywords = top_keywords_per_cluster(n_terms=7)
for c, words in keywords.items():
    print(f"Cluster {c}: {', '.join(words)}")

#Cluster Distancing Score
import numpy as np

# After fitting kmeans
distances = kmeans.transform(X)  # shape: (n_samples, n_clusters), distance to each centroid

# Get the distance to the assigned cluster centroid
assigned_distances = distances[np.arange(len(X)), df2["cluster"].values]

# Option 1: raw distances (lower = closer, better fit)
df2["cluster_distancing_score"] = assigned_distances

# Option 2: similarity-style score (higher = closer, better fit)
# Normalize so scores are between 0 and 1
max_dist = assigned_distances.max()
min_dist = assigned_distances.min()
df2["cluster_distancing_score_norm"] = 1 - (assigned_distances - min_dist) / (max_dist - min_dist)

# Preview
print(df2[["processed_textOriginal", "cluster", "cluster_distancing_score", "cluster_distancing_score_norm"]].head(20))

#Content-level Relevance Score
#Combined video embedding
import numpy as np

def combine_embeddings(row):
    emb_list = []
    if row["title_embeddings"] is not None and isinstance(row["title_embeddings"], (list, np.ndarray)):
        emb_list.append(np.array(row["title_embeddings"]))
    if row["description_embeddings"] is not None and isinstance(row["description_embeddings"], (list, np.ndarray)):
        emb_list.append(np.array(row["description_embeddings"]))

    if len(emb_list) > 0:
        return np.mean(emb_list, axis=0)
    else:
        return None  # return None if both are missing

df["video_embeddings"] = df.apply(combine_embeddings, axis=1)

#Map videoId → embedding
videoid_to_emb = dict(zip(df["videoId"], df["video_embeddings"]))

#Compute comment–video relevance
from sklearn.metrics.pairwise import cosine_similarity

relevance_scores = []
for i, row in df2.iterrows():
    vid = row["videoId"]
    comment_emb = np.array(row["text_embeddings"])
    video_emb = videoid_to_emb.get(vid, None)

    if video_emb is not None:
        score = cosine_similarity([comment_emb], [video_emb])[0][0]
    else:
        score = np.nan
    relevance_scores.append(score)

df2["content-level_relevance"] = relevance_scores

#Normalize
min_score = df2["content-level_relevance"].min()
max_score = df2["content-level_relevance"].max()

df2["content-level_relevance_norm"] = (
    df2["content-level_relevance"] - min_score
) / (max_score - min_score)

#Recency Score
import pandas as pd
from datetime import datetime, timezone

# Ensure updatedAt is datetime (you already did this)
df2["updatedAt"] = pd.to_datetime(df2["updatedAt"], utc=True)

# Use current time as reference
now = datetime.now(timezone.utc)

# Compute days since comment was posted
df2["days_since_comment"] = (now - df2["updatedAt"]).dt.total_seconds() / (60*60*24)

# Normalize so that latest comment = 1, oldest = 0
min_days = df2["days_since_comment"].min()
max_days = df2["days_since_comment"].max()
df2["recency_norm"] = 1 - (df2["days_since_comment"] - min_days) / (max_days - min_days)

#Overall Relevance Score Without Lengthy Score
def compute_overall_relevance_without_lengthy(row):
    if row["days_since_comment"] > 30:
        # Older comments → recency matters less
        return (
            0.65 * row["content-level_relevance_norm"] +
            0.25 * row["cluster_distancing_score_norm"] +
            0.10 * row["recency_norm"]
        )
    else:
        # Fresh comments → recency not factored
        return (
            0.7 * row["content-level_relevance_norm"] +
            0.3 * row["cluster_distancing_score_norm"]
        )

df2["overall_relevance_score_without_lengthy_score"] = df2.apply(compute_overall_relevance_without_lengthy, axis=1)

# Length Score
#--- Compute comment length from original text ---
df2["comment_length"] = df2["textOriginal"].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

# --- Normalize length score among non-spam comments ---
non_spam_mask = df2["overall_relevance_score_without_lengthy_score"] >= 0.2
min_len = df2.loc[non_spam_mask, "comment_length"].min()
max_len = df2.loc[non_spam_mask, "comment_length"].max()

def normalize_length(length):
    if max_len == min_len:
        return 1.0
    return (length - min_len) / (max_len - min_len)

df2["length_score_norm"] = 0.0  # default 0 for spam
df2.loc[non_spam_mask, "length_score_norm"] = df2.loc[non_spam_mask, "comment_length"].apply(normalize_length)
df2

#Overall Quality Score
def compute_overall_quality (row):
    if row["days_since_comment"] > 30:
        # Older comments → recency matters less
        return (
            0.60 * row["content-level_relevance_norm"] +
            0.20 * row["cluster_distancing_score_norm"] +
            0.10 * row["recency_norm"] +
            0.10 * row["length_score_norm"]
        )
    else:
        # Fresh comments → recency not factored
        return (
            0.55 * row["content-level_relevance_norm"] +
            0.35 * row["cluster_distancing_score_norm"] +
            0.10 * row["length_score_norm"]
        )

df2["overall_quality_score"] = df2.apply(compute_overall_quality, axis=1)

# --- Assign quality categories based on thresholds ---
def assign_quality(row):
    if row["spam"]:   # spam always overrides
        return "Spam"
    score = row["overall_quality_score"]
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.4:
        return "Good"
    elif score >= 0.2:
        return "Poor"
    else:
        return "Spam"   # fallback if very low score

df2["overall_quality"] = df2.apply(assign_quality, axis=1)

# Optional: preview distribution
print(df2[["textOriginal", "overall_quality_score", "overall_quality"]])

# --- SOE metric grade function (separate output) ---
def soe_grade(view_count, engagement_count):
    """Compute SOE grade based on ratio (engagement / views * 100)"""
    if engagement_count == 0 or view_count == 0:
        return "poor"
    ratio = (engagement_count / view_count) * 100
    if ratio < 2.5:
        return "poor"
    elif 2.5 <= ratio < 4.0:
        return "average"
    elif 4.0 <= ratio <= 6.0:
        return "good"
    else:
        return "outperforming"

def compute_soe_grades(df):
    grade_df = df[['videoId', 'viewCount', 'likeCount', 'favouriteCount', 'commentCount']].copy()
    for metric in ['likeCount', 'favouriteCount', 'commentCount']:
        if metric in grade_df.columns:
            grade_col = f"{metric}_grade"
            grade_df[grade_col] = grade_df.apply(
                lambda row: soe_grade(row['viewCount'], row[metric]), axis=1
            )
    return grade_df

# --- Classify video metadata (title, description, tags) ---
def classify_video_metadata(df):
    category_results = []
    for _, row in df.iterrows():
        text_parts = []
        for col in ["title", "description", "tags"]:
            if col in df.columns and pd.notnull(row.get(col, "")):
                text_parts.append(str(row[col]))
        combined_text = " ".join(text_parts)
        flags = classify_text_with_terms(combined_text)
        categories = []
        if flags.get("fragrance"):
            categories.append("fragrance")
        if flags.get("makeup"):
            categories.append("makeup")
        if flags.get("skincare"):
            categories.append("skincare")
        category_results.append(", ".join(categories) if categories else "other")
    df["video_categories"] = category_results
    return df

# Apply to videos DataFrame
df = classify_video_metadata(df)

# --- Generate SOE grades output ---
soe_grades_df = compute_soe_grades(df)

# --- Outputs ---
print("Video categories from metadata:")
print(df[["videoId", "video_categories"]].head(10))

print("\nSOE grades dataframe:")
print(soe_grades_df.head(10))

# Map sentiment to numeric score
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df2["sentiment_score"] = df2["sentiment"].map(sentiment_map)

# Group by category
category_sentiment_summary = df2.groupby("category").agg(
    total_comments=("sentiment", "count"),
    positive_count=("sentiment", lambda x: (x == "positive").sum()),
    neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
    negative_count=("sentiment", lambda x: (x == "negative").sum()),
    avg_sentiment_score=("sentiment_score", "mean")
).reset_index()

# Assign average sentiment label
def avg_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

category_sentiment_summary["avg_sentiment_label"] = category_sentiment_summary["avg_sentiment_score"].apply(avg_sentiment_label)

print("\n=== Category Sentiment Summary ===")
print(category_sentiment_summary)

# If commentId exists
if "commentId" in df2.columns:
    comment_quality = df2[["commentId", "overall_quality_score", "overall_quality"]].copy()
else:
    # fallback: use index as ID
    comment_quality = df2.reset_index()[["index", "overall_quality_score", "overall_quality"]].rename(columns={"index": "commentId"})

print("\n📊 Comment Quality Table:")
print(comment_quality)

# --- Save to Excel ---
# 1) Sentiment Breakdown sheet
# We'll build rows for Products, Brands, and a final Overall row with totals.

# Extract original entities
df2["original_entities"] = df2["textOriginal"].fillna("").apply(lambda x: extract_entities(str(x)))

def get_product_entities(ent_list):
    product_labels = {"PRODUCT", "WORK_OF_ART", "ORG"}
    found = []
    for ent_text, ent_label in ent_list:
        if ent_label in product_labels:
            t = ent_text.strip()
            if t:
                found.append(t)
    return list(dict.fromkeys(found))

df2["product_entities"] = df2["original_entities"].apply(get_product_entities)

# Ensure list consistency
df2["brands_detected"] = df2["brands_detected"].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
)
df2["product_entities"] = df2["product_entities"].apply(
    lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
)

# Flatten unique entities
all_brands = sorted(list(dict.fromkeys([b for lst in df2["brands_detected"] for b in lst if b and str(b).strip()])))
all_products = sorted(list(dict.fromkeys([p for lst in df2["product_entities"] for p in lst if p and str(p).strip()])))

breakdown_rows = []

# Products
for prod in all_products:
    mask = df2["textOriginal"].fillna("").str.contains(re.escape(prod), case=False, na=False) | \
           df2["product_entities"].apply(lambda lst: prod in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({
        "Topic": "Product",
        "Detected Terms": prod,
        "Positive Sentiment": pos,
        "Negative Sentiment": neg,
        "Neutral Sentiment": neu
    })

# Brands
for brand in all_brands:
    mask = df2["brands_detected"].apply(lambda lst: brand in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({
        "Topic": "Brand",
        "Detected Terms": brand,
        "Positive Sentiment": pos,
        "Negative Sentiment": neg,
        "Neutral Sentiment": neu
    })

# Overall totals
total_pos = int((df2["sentiment"] == "positive").sum())
total_neg = int((df2["sentiment"] == "negative").sum())
total_neu = int((df2["sentiment"] == "neutral").sum())

breakdown_rows.append({
    "Topic": "Overall",
    "Detected Terms": "total",
    "Positive Sentiment": total_pos,
    "Negative Sentiment": total_neg,
    "Neutral Sentiment": total_neu
})

sentiment_breakdown_df = pd.DataFrame(breakdown_rows, columns=[
    "Topic", "Detected Terms", "Positive Sentiment", "Negative Sentiment", "Neutral Sentiment"
])

print("\nSentiment breakdown preview:")
print(sentiment_breakdown_df.head(30))

# 2) Comment Categorisation
cat_rows = []
for idx, row in df2.iterrows():
    comment_category = row.get("category", "other")
    sentiment = row.get("sentiment", "neutral")
    brands_list = row.get("brands_detected", [])
    products_list = row.get("product_entities", [])

    if isinstance(brands_list, list):
        for b in brands_list:
            cat_rows.append({
                "Category": comment_category.capitalize(),
                "Topic": "Brand",
                "Detected Text": b,
                "Sentiment": sentiment.capitalize()
            })
    if isinstance(products_list, list):
        for p in products_list:
            cat_rows.append({
                "Category": comment_category.capitalize(),
                "Topic": "Product",
                "Detected Text": p,
                "Sentiment": sentiment.capitalize()
            })

comment_categorisation_df = pd.DataFrame(cat_rows, columns=["Category", "Topic", "Detected Text", "Sentiment"])

print("\nComment categorisation preview:")
print(comment_categorisation_df.head(30))

# --- Save to Excel ---
# 1) Sentiment Breakdown sheet
df2["original_entities"] = df2["textOriginal"].fillna("").apply(lambda x: extract_entities(str(x)))

def get_product_entities(ent_list):
    product_labels = {"PRODUCT", "WORK_OF_ART", "ORG"}
    return list(dict.fromkeys([ent_text.strip() for ent_text, ent_label in ent_list if ent_label in product_labels and ent_text.strip()]))

df2["product_entities"] = df2["original_entities"].apply(get_product_entities)

df2["brands_detected"] = df2["brands_detected"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
df2["product_entities"] = df2["product_entities"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

all_brands = sorted(list(dict.fromkeys([b for lst in df2["brands_detected"] for b in lst if b and str(b).strip()])))
all_products = sorted(list(dict.fromkeys([p for lst in df2["product_entities"] for p in lst if p and str(p).strip()])))

breakdown_rows = []

# Products
for prod in all_products:
    mask = df2["textOriginal"].fillna("").str.contains(re.escape(prod), case=False, na=False) | \
           df2["product_entities"].apply(lambda lst: prod in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({"Topic": "Product", "Detected Terms": prod, "Positive Sentiment": pos, "Negative Sentiment": neg, "Neutral Sentiment": neu})

# Brands
for brand in all_brands:
    mask = df2["brands_detected"].apply(lambda lst: brand in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({"Topic": "Brand", "Detected Terms": brand, "Positive Sentiment": pos, "Negative Sentiment": neg, "Neutral Sentiment": neu})

# Totals
total_pos = int((df2["sentiment"] == "positive").sum())
total_neg = int((df2["sentiment"] == "negative").sum())
total_neu = int((df2["sentiment"] == "neutral").sum())
breakdown_rows.append({"Topic": "Overall", "Detected Terms": "total", "Positive Sentiment": total_pos, "Negative Sentiment": total_neg, "Neutral Sentiment": total_neu})

sentiment_breakdown_df = pd.DataFrame(breakdown_rows, columns=["Topic", "Detected Terms", "Positive Sentiment", "Negative Sentiment", "Neutral Sentiment"])


# 2) Comment Categorisation
cat_rows = []
for idx, row in df2.iterrows():
    comment_category = row.get("category", "other")
    sentiment = row.get("sentiment", "neutral")
    brands_list = row.get("brands_detected", [])
    products_list = row.get("product_entities", [])

    if isinstance(brands_list, list):
        for b in brands_list:
            cat_rows.append({"Category": comment_category.capitalize(), "Topic": "Brand", "Detected Text": b, "Sentiment": sentiment.capitalize()})
    if isinstance(products_list, list):
        for p in products_list:
            cat_rows.append({"Category": comment_category.capitalize(), "Topic": "Product", "Detected Text": p, "Sentiment": sentiment.capitalize()})

comment_categorisation_df = pd.DataFrame(cat_rows, columns=["Category", "Topic", "Detected Text", "Sentiment"])

# --- Save to Excel ---
# 1) Sentiment Breakdown sheet
df2["original_entities"] = df2["textOriginal"].fillna("").apply(lambda x: extract_entities(str(x)))

def get_product_entities(ent_list):
    product_labels = {"PRODUCT", "WORK_OF_ART", "ORG"}
    return list(dict.fromkeys([ent_text.strip() for ent_text, ent_label in ent_list if ent_label in product_labels and ent_text.strip()]))

df2["product_entities"] = df2["original_entities"].apply(get_product_entities)

df2["brands_detected"] = df2["brands_detected"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
df2["product_entities"] = df2["product_entities"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

all_brands = sorted(list(dict.fromkeys([b for lst in df2["brands_detected"] for b in lst if b and str(b).strip()])))
all_products = sorted(list(dict.fromkeys([p for lst in df2["product_entities"] for p in lst if p and str(p).strip()])))

breakdown_rows = []

# Products
for prod in all_products:
    mask = df2["textOriginal"].fillna("").str.contains(re.escape(prod), case=False, na=False) | \
           df2["product_entities"].apply(lambda lst: prod in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({"Topic": "Product", "Detected Terms": prod, "Positive Sentiment": pos, "Negative Sentiment": neg, "Neutral Sentiment": neu})

# Brands
for brand in all_brands:
    mask = df2["brands_detected"].apply(lambda lst: brand in lst if isinstance(lst, list) else False)
    pos = int((df2.loc[mask, "sentiment"] == "positive").sum())
    neg = int((df2.loc[mask, "sentiment"] == "negative").sum())
    neu = int((df2.loc[mask, "sentiment"] == "neutral").sum())
    breakdown_rows.append({"Topic": "Brand", "Detected Terms": brand, "Positive Sentiment": pos, "Negative Sentiment": neg, "Neutral Sentiment": neu})

# Build dataframe first
sentiment_breakdown_df = pd.DataFrame(
    breakdown_rows,
    columns=["Topic", "Detected Terms", "Positive Sentiment", "Negative Sentiment", "Neutral Sentiment"]
)

# Now calculate totals
total_pos = sentiment_breakdown_df["Positive Sentiment"].sum()
total_neg = sentiment_breakdown_df["Negative Sentiment"].sum()
total_neu = sentiment_breakdown_df["Neutral Sentiment"].sum()

# Append totals row
sentiment_breakdown_df = pd.concat([
    sentiment_breakdown_df,
    pd.DataFrame([{
        "Topic": "Overall",
        "Detected Terms": "Total",
        "Positive Sentiment": total_pos,
        "Negative Sentiment": total_neg,
        "Neutral Sentiment": total_neu
    }])
], ignore_index=True)

# 2) Comment Categorisation
cat_rows = []
for idx, row in df2.iterrows():
    comment_category = row.get("category", "other")
    sentiment = row.get("sentiment", "neutral")
    brands_list = row.get("brands_detected", [])
    products_list = row.get("product_entities", [])

    if isinstance(brands_list, list):
        for b in brands_list:
            cat_rows.append({"Category": comment_category.capitalize(), "Topic": "Brand", "Detected Text": b, "Sentiment": sentiment.capitalize()})
    if isinstance(products_list, list):
        for p in products_list:
            cat_rows.append({"Category": comment_category.capitalize(), "Topic": "Product", "Detected Text": p, "Sentiment": sentiment.capitalize()})

comment_categorisation_df = pd.DataFrame(cat_rows, columns=["Category", "Topic", "Detected Text", "Sentiment"])

# ============================
# Video Meta Data Section
# ============================
video_meta_rows = []

for _, row in df.iterrows():
    vid = row["videoId"]
    video_meta_rows.extend([
        {"Video Details": "VideoID", "Description": vid},
        {"Video Details": "Video Category", "Description": row.get("video_categories", "other")},
        {"Video Details": "Detected Theme", "Description": row.get("title", "")},
        {"Video Details": "", "Description": ""}  # spacer row
    ])

video_meta_df = pd.DataFrame(video_meta_rows, columns=["Video Details", "Description"])


# ============================
# Content Effectiveness Section
# ============================
def ratio_class_from_ratio(ratio):
    if pd.isna(ratio) or ratio == 0: return "Poor"
    if ratio < 2.5: return "Poor"
    elif ratio < 4.0: return "Average"
    elif ratio < 6.0: return "Good"
    else: return "Outperforming"

content_eff_rows = []

for _, row in df.iterrows():
    views = int(row.get("viewCount", 0))
    likes = int(row.get("likeCount", 0))
    favs = int(row.get("favouriteCount", 0))
    comments = int(row.get("commentCount", 0))

    like_ratio = (likes / views * 100) if views > 0 else 0.0
    fav_ratio = (favs / views * 100) if views > 0 else 0.0
    comment_ratio = (comments / views * 100) if views > 0 else 0.0

    content_eff_rows.extend([
        {"Metric": "Like-to-View", "SOE Ratio": round(like_ratio, 4), "SOE Class": ratio_class_from_ratio(like_ratio)},
        {"Metric": "Favourite-to-View", "SOE Ratio": round(fav_ratio, 4), "SOE Class": ratio_class_from_ratio(fav_ratio)},
        {"Metric": "Comment-to-View", "SOE Ratio": round(comment_ratio, 4), "SOE Class": ratio_class_from_ratio(comment_ratio)},
        {"Metric": "", "SOE Ratio": "", "SOE Class": ""}  # spacer row
    ])

content_eff_df = pd.DataFrame(content_eff_rows, columns=["Metric", "SOE Ratio", "SOE Class"])

# Add constant target column
content_eff_df["target"] = 6.0

# ============================
# Save everything to Excel
# ============================
output_path = "comment_analysis_results.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    sentiment_breakdown_df.to_excel(writer, sheet_name="SentimentBreakdown", index=False)
    comment_categorisation_df.to_excel(writer, sheet_name="CommentCategorisation", index=False)
    comment_quality.to_excel(writer, sheet_name="CommentQuality", index=False)
    video_meta_df.to_excel(writer, sheet_name="VideoMetaData", index=False)
    content_eff_df.to_excel(writer, sheet_name="ContentEffectiveness", index=False)

print(f"✅ Excel file saved as {output_path}")