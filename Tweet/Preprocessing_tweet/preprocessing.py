
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import spacy
from spellchecker import SpellChecker
import emoji
import contractions

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
spell = SpellChecker()
nlp = spacy.load('en_core_web_sm')

CONTRACTION_MAP = {
    "ain't"   : "am not / are not / is not / has not / have not",   "aren't"   : "are not",
    "can't"   : "cannot",                                           "could've" : "could have",
    "couldn't": "could not",                                        "didn't"   : "did not",
    "doesn't" : "does not",                                         "don't"    : "do not",
    "hadn't"  : "had not",                                          "hasn't"   : "has not",
    "haven't" : "have not",                                         "he'd"     : "he had / he would",
    "he'll"   : "he will / he shall",                               "he's"     : "he is / he has",
    "how'd"   : "how did / how would",                              "how'll"   : "how will",
    "how's"   : "how is / how has",                                 "i'd"      : "i would / i had",
    "i'll"    : "i will / i shall",                                 "i'm"      : "i am",
    "i've"    : "i have",                                           "isn't"    : "is not",
    "it'd"    : "it would / it had",                                "it'll"    : "it will / it shall",
    "it's"    : "it is / it has",                                   "let's"    : "let us",
    "ma'am"   : "madam",                                            "might've" : "might have",
    "mightn't": "might not",                                        "must've"  : "must have",
    "mustn't" : "must not",                                         "needn't"  : "need not",
    "o'clock" : "of the clock",                                     "ol'"      : "old",
    "oughtn't": "ought not",                                        "shan't"   : "shall not",
    "she'd"   : "she had / she would",                              "she'll"   : "she will / she shall",
    "she's"   : "she is / she has",                                 "should've": "should have",
    "shouldn't": "should not",                                      "that'd"   : "that would / that had",
    "that's"  : "that is / that has",                               "there'd"  : "there had / there would",
    "there'll": "there will / there shall",                         "there's"  : "there is / there has",
    "they'd"  : "they had / they would",                            "they'll"  : "they will / they shall",
    "they're" : "they are",                                         "they've"  : "they have",
    "wasn't"  : "was not",                                          "we'd"     : "we had / we would",
    "we'll"   : "we will / we shall",                               "we're"    : "we are",
    "we've"   : "we have",                                          "weren't"  : "were not",
    "what'd"  : "what did",                                         "what'll"  : "what will / what shall",
    "what're" : "what are",                                         "what's"   : "what is / what has",
    "what've" : "what have",                                        "when's"   : "when is / when has",
    "when've" : "when have",                                        "where'd"  : "where did / where would",
    "where's" : "where is / where has",                             "where've" : "where have",
    "who'd"   : "who had / who would",                              "who'll"   : "who will / who shall",
    "who're"  : "who are",                                          "who's"    : "who is / who has",
    "who've"  : "who have",                                         "why'd"    : "why did",
    "why's"   : "why is / why has",                                 "won't"    : "will not",
    "would've": "would have",                                       "wouldn't" : "would not",
    "y'all"   : "you all",                                          "you'd"    : "you had / you would",
    "you'll"  : "you will / you shall",                             "you're"   : "you are",
    "you've"  : "you have",

    # Informal abbreviations
    "wanna"   : "want to",                                          "gonna"    : "going to",
    "gotta"   : "got to",                                           "hafta"    : "have to",
    "hasta"   : "has to",                                           "oughta"   : "ought to",
    "needa"   : "need to",                                          "tryna"    : "trying to",
    "kinda"   : "kind of",                                          "sorta"    : "sort of",
    "outta"   : "out of",                                           "lotsa"    : "lots of",
    "loadsa"  : "loads of",                                         "cuppa"    : "cup of",
    "alotta"  : "a lot of",                                         "lemme"    : "let me",
    "gimme"   : "give me",                                          "tell'em"  : "tell them",
    "dunno"   : "do not know",                                      "whatcha"  : "what are you / what have you",
    "betcha"  : "bet you",                                          "doncha"   : "do not you",
    "gotcha"  : "got you",                                          "init"     : "is not it",
    "innit"   : "is not it",                                        "howdy"    : "how do you do",
    "sup"     : "what is up"
}

POLITICAL_SYNONYMS = {
    # National Parties
    "bjp": [
        "bharatiya janata party", "modi party", "b.j.p", "bjparty", "bjp govt", "nda", "bjp wale",
        "bhakts", "sangh", "right wing", "hindutva gang", "it cell", "sanghis", "bhakton",
        "#bjp", "#modisarkar", "godse bhakts", "chaddis", "mandir wale", "fascists", "andhbhakts",
        "modia", "saffron brigade", "gaumutra gang", "vishwaguru", "bhakt log", "desh ke gaddar"
    ],
    "congress": [
        "indian national congress", "inc", "i.n.c", "cong", "congress party", "congress govt", "upa",
        "congressi", "italian party", "pappu party", "#congress", "#rahulgandhi", "liberandus",
        "dynasts", "naamdar", "tukde tukde gang", "anti-hindu", "lutyens gang", "sickulars",
        "khan market gang", "pseudo-liberals", "nehru parivaar", "award wapsi gang"
    ],
    "aap": [
        "aam aadmi party", "a.a.p", "kejriwal party", "jhadu party", "aap govt", "#aap", "#kejriwal",
        "anarchists", "dharna party", "freebie gang", "mufflerman", "bhagoda party", "NGO party",
        "urban naxals", "AAPtards", "jhadu wale", "tax evaders", "L-G vs AAP"
    ],



    # Regional Parties
    "shiv_sena": [
        "shiv sena", "shivsena", "uddhav sena", "balasaheb party", "thackeray gang", "mumbai wale",
        "hindutva sena", "saamna party", "shiv sena ubt", "eknath shinde faction", "tiger army",
        "#shivsena", "#uddhavthackeray", "maratha manoos"
    ],
    "ncp": [
        "nationalist congress party", "n.c.p", "pawar party", "sharad pawar gang", "maha aghadi",
        "pawar saheb", "uncle politics", "ncp sharadchandra", "#ncp", "#sharadpawar", "sugar lobby"
    ],
    "tmc": [
        "trinamool congress", "t.m.c", "tmc party", "didi party", "tali gang", "aunty party",
        "mamata brigade", "cut money gang", "bengal model", "tolabaji sarkar", "syndicate raj",
        "#tmc", "#mamatabanerjee", "ghotis vs bangals"
    ],
    "bsp": [
        "bahujan samaj party", "b.s.p", "mayawati party", "elephant party", "dalit leader",
        "kanshi ram legacy", "statue politics", "#bsp", "#mayawati", "social justice"
    ],
    "sp": [
        "samajwadi party", "s.p", "yadav party", "cycle party", "akhilesh gang", "mulayam parivaar",
        "my vs ay", "gunda raj", "#sp", "#akhileshyadav", "pariwartan wale"
    ],
    "rjd": [
        "rashtriya janata dal", "r.j.d", "lalu party", "lantern party", "jungle raj", "tejashwi sena",
        "mandal politics", "#rjd", "#laluprasad", "bihar ka bhrashtachar"
    ],
    "aiadmk": [
        "all india anna dravida munnetra kazhagam", "aiadmk", "a.i.a.d.m.k", "amma party",
        "eps vs ops", "two leaves", "jaya legacy", "dravidian model", "#aiadmk", "#edappadi",
        "sasikala gang"
    ],
    "dmk": [
        "dravida munnetra kazhagam", "d.m.k", "dmk party", "stalin sarkar", "karunanidhi legacy",
        "rising sun", "anti-hindi gang", "periyarists", "#dmk", "#mkstalin", "kalaignar"
    ],
    "bjd": [
        "biju janata dal", "b.j.d", "bjd party", "navin patnaik gang", "odia asmita",
        "conch symbol", "#bjd", "#navinpatnaik", "outsider debate"
    ],
    "trs_brs": [
        "telangana rashtra samithi", "t.r.s", "trs party", "kcr party", "bharat rashtra samithi",
        "brs", "pink party", "telangana pride", "#trs", "#brs", "#kcr", "farmers leader"
    ],
    "cpi": [
        "communist party of india", "c.p.i", "red flag", "leftists", "comrades", "#cpi",
        "kerala communists", "anti-capitalists"
    ],
    "cpim": [
        "communist party of india marxist", "c.p.i.m", "cpim party", "left front", "pinarayi gang",
        "kerala model", "#cpim", "#pinarayivijayan", "lal salaam"
    ],



    # Political Leaders
    "modi": [
        "narendra modi", "modiji", "pm modi", "shri modi", "naMo", "#modi", "@narendramodi",
        "56 inch", "chowkidar", "chaiwala", "supreme leader", "feku", "jumlajeevi", "gujju",
        "hindu hriday samrat", "vikas purush", "mota bhai", "mitron", "demonetization man"
    ],
    "rahul_gandhi": [
        "rahul gandhi", "rg", "rahul baba", "@rahulgandhi", "pappu", "yuvraj", "#rahulgandhi",
        "shehzada", "failed politician", "buddhu", "vacation lover", "temple run", "bharat jodo",
        "janeudhari", "laptop bhagoda"
    ],
    "kejriwal": [
        "arvind kejriwal", "kejri", "kejriwal ji", "jhaduwala", "#kejriwal", "@arvindkejriwal",
        "anarchist", "bhagoda", "mufflerman", "freebie king", "NGO activist", "L-G vs kejri",
        "tax fraud", "kattar beiman"
    ],
    "mamata": [
        "mamata banerjee", "didi", "cm mamata", "aunty", "#mamata", "#didi", "tolabaji queen",
        "cut money didi", "syndicate raj", "bengal tigress", "anti-central", "jumla didi"
    ],
    "yogi": [
        "yogi adityanath", "cm yogi", "yogi ji", "monk cm", "gogi", "#yogi", "bulldozer baba",
        "hindu yogi", "love jihad crusader", "thakur leader", "up model", "gau rakshak"
    ],
    "nitish": [
        "nitish kumar", "cm nitish", "sushasan babu", "#nitish", "palturam", "opportunist",
        "bihar model", "lalten", "flip-flop CM", "tejashwi ka chacha"
    ],
    "sharad_pawar": [
        "sharad pawar", "pawar sahab", "uncle of all", "#sharadpawar", "maratha strongman",
        "political chanakya", "sugar baron", "ncp neta", "maha politics"
    ],
    "kcr": [
        "k chandrasekhar rao", "kcr", "cm kcr", "#kcr", "telangana tiger", "kaleshwaram",
        "family politics", "autocratic leader", "pink revolution"
    ],
    "mk_stalin": [
        "m.k. stalin", "mk stalin", "stalin", "#stalin", "dravidian leader", "kalaignar son",
        "anti-hindi", "periyarist", "tamil pride"
    ],



    # Ideologies & Online Warfare
    "leftists": [
        "left wing", "liberals", "communists", "anti-national", "urban naxals", "presstitutes",
        "toolkit gang", "jnu gang", "tukde tukde", "award wapsi gang", "sickulars", "librandus"
    ],
    "bhakts": [
        "bjp supporters", "sanghis", "blind bhakts", "modi fans", "it cell", "#bhakts",
        "chaddis", "gaumutra drinkers", "mandir wahis", "fascists", "godse lovers"
    ],
    "libtards": [
        "congressis", "liberals", "toolkit", "leftists", "ecosystem", "#libtards", "khan market gang",
        "nehruvian", "anti-hindu", "lutyens elite", "pseudo-secular"
    ],
    "upa": [
        "united progressive alliance", "congress govt", "#upa", "sonia sarkar", "manmohan era",
        "policy paralysis", "scam india", "2G era"
    ],
    "nda": [
        "national democratic alliance", "bjp govt", "#nda", "modi sarkar", "achhe din",
        "double engine", "sabka saath", "new india"
    ],



    # Controversial Terms
    "toolkit": [
        "propaganda document", "social media war", "congress toolkit", "aap toolkit", "bjp it cell",
        "fake narrative", "disinformation"
    ],
    "anti_national": [
        "deshdrohi", "tukde tukde gang", "urban maoists", "khalistanis", "jnu gang",
        "anti-india", "breaking india forces"
    ],
    "godse": [
        "nathuram godse", "gandhi killer", "godse bhakts", "hindu terror", "right-wing extremist"
    ],



    # Media & Hashtag Wars
    "godi_media": [
        "modi media", "sold out media", "gujarati media", "bhakt news", "godimedia",
        "lapdog journalists", "presstitutes"
    ],
    "pappu": [
        "rahul gandhi", "pappu", "failed leader", "buddhu", "shehzada", "congress prince"
    ],
    "feku": [
        "modi", "jumla king", "false promises", "56 inch fraud", "hollow speeches"
    ],
    "bhagoda": [
        "kejriwal", "runaway cm", "delhi bhagoda", "escape artist", "non-confronter"
    ],



    # State-Specific Slang
    "delhi_model": [
        "freebies", "aap governance", "mohalla clinics", "free electricity", "school reforms"
    ],
    "gujarat_model": [
        "vikas", "business friendly", "adani ambani", "2002 riots", "hindu rashtra lab"
    ],
    "kerala_model": [
        "communist govt", "human development", "gulf money", "god's own country", "lal salaam"
    ],
    "bengal_model": [
        "tmc rule", "political violence", "syndicate raj", "cut money", "didi's law"
    ]

}

# Initialize ekphrasis processor
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone',
               'user','time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated",
              "repeated",'emphasis', 'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

# To get WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# To lemmatize tokens
def lemmatize_tokens(tokens):
    pos_tags = pos_tag(tokens)
    lemmatized = []
    for token, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, wn_tag)
        lemmatized.append(lemma)
    return lemmatized

# To expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# To process mentions
def process_mentions(text):
    return re.sub(r'@(\w+)', lambda match: f'[USER_{match.group(1).upper()}]', text)

# To translate emojis
def translate_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

# To remove URLs
def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def normalize_slang(text):
    slang_map = {
        "idk"        : "i do not know",              "imo"        : "in my opinion",              "imho"       : "in my humble opinion",
        "brb"        : "be right back",              "gtg"        : "got to go",                  "ttyl"       : "talk to you later",
        "tbh"        : "to be honest",               "rofl"       : "rolling on the floor laughing",
        "lmao"       : "laughing my ass off",        "lol"        : "laugh out loud",             "omg"        : "oh my god",
        "wtf"        : "what the fuck",              "wth"        : "what the hell",              "fml"        : "fuck my life",
        "afk"        : "away from keyboard",         "irl"        : "in real life",               "smh"        : "shaking my head",
        "ikr"        : "i know right",               "bff"        : "best friends forever",       "fyi"        : "for your information",
        "bday"       : "birthday",                   "bc"         : "because",                    "jk"         : "just kidding",
        "nvm"        : "never mind",                 "tmi"        : "too much information",       "asap"       : "as soon as possible",
        "bts"        : "behind the scenes",          "gg"         : "good game",                  "afaik"      : "as far as i know",
        "btw"        : "by the way",                 "dm"         : "direct message",             "gm"         : "good morning",
        "gn"         : "good night",                 "gr8"        : "great",                      "hbd"        : "happy birthday",
        "hbu"        : "how about you",              "hmu"        : "hit me up",                  "idc"        : "i do not care",
        "ily"        : "i love you",                 "lmk"        : "let me know",                "nah"        : "no",
        "np"         : "no problem",                 "nsfw"       : "not safe for work",          "omw"        : "on my way",
        "oop"        : "oops",                       "plz"        : "please",                     "rn"         : "right now",
        "srsly"      : "seriously",                  "sus"        : "suspicious",                 "thx"        : "thanks",
        "tho"        : "though",                     "u"          : "you",                        "ur"         : "your",
        "ya"         : "you",                        "yolo"       : "you only live once",         "wyd"        : "what are you doing",
        "wbu"        : "what about you",             "xoxo"       : "hugs and kisses",            "yaas"       : "yes",
        "yo"         : "hey",                        "vibe"       : "feeling or atmosphere",      "noob"       : "newbie",
        "meh"        : "indifferent",                "lit"        : "amazing",                    "cringe"     : "embarrassing",
        "stan"       : "obsessive fan",              "fam"        : "close friend",               "salty"      : "bitter",
        "tea"        : "gossip",                     "cap"        : "lie",                        "no cap"     : "no lie",
        "ghosted"    : "ignored",                    "flex"       : "show off",                   "clout"      : "influence",
        "fire"       : "awesome",                    "slay"       : "excel",                      "lowkey"     : "quietly",
        "highkey"    : "openly",                     "v"          : "very",                       "w"          : "win",
        "l"          : "loss",                       "based"      : "confident and unapologetic", "simp"       : "overly submissive",
        "bet"        : "okay",                       "goat"       : "greatest of all time",       "drip"       : "stylish appearance",
        "ratio"      : "disproportionate engagement","fr"         : "for real",                   "deadass"    : "seriously",
        "big yikes"  : "very embarrassing",          "periodt"    : "final point",                "say less"   : "understood",
        "bop"        : "good song",                  "ok boomer"  : "dismissive retort",          "pressed"    : "annoyed",
        "snatched"   : "on point",                   "receipts"   : "proof",                      "shade"      : "disrespect",
        "iconic"     : "legendary",                  "savage"     : "bold",                       "fomo"       : "fear of missing out",
        "jomo"       : "joy of missing out",         "banger"     : "very good",                  "cancelled"  : "boycotted",
        "boi"        : "boy",                        "legit"      : "genuine",                    "vibe check" : "test of mood",
        "shook"      : "shocked",                    "hella"      : "very",                       "yass"       : "enthusiastic yes",
        "boomer"     : "old person",                 "karen"      : "entitled woman",             "yeet"       : "throw forcefully",
        "rekt"       : "destroyed",                  "troll"      : "online provocateur",         "clown"      : "foolish person",
        "mid"        : "mediocre",                   "slaps"      : "very good (music)",          "zaddy"      : "attractive man",
        "ship"       : "support a relationship",     "fb"         : "facebook",                   "ig"         : "instagram",
        "tt"         : "tiktok",                     "yt"         : "youtube",                    "snap"       : "snapchat",
        "filter"     : "image modification effect",  "tagged"     : "mentioned",                  "unfriend"   : "remove from friend list",
        "unfollow"   : "stop following",             "hashtag"    : "metadata tag",               "bio"        : "profile description",
        "handle"     : "username",                   "trending"   : "popular right now",          "yup"        : "yes",
        "nope"       : "no",                         "yep"        : "yes",                        "hmm"        : "thinking sound",
        "uh-huh"     : "yes",                        "uh-uh"      : "no",                         "bleh"       : "disgust",
        "duh"        : "obvious",                    "pfft"       : "disdain",                    "psst"       : "attention",
        "shh"        : "be quiet",                   "mhm"        : "yes",                        "hm"         : "thinking sound",
        "grr"        : "anger",                      "haha"       : "laughter",                   "hehe"       : "laughter",
        "lmfao"      : "laughing my fucking ass off","stfu"       : "shut the fuck up",           "gtfo"       : "get the fuck out",
        "sfw"        : "safe for work",              "tl;dr"      : "too long; did not read",     "icymi"      : "in case you missed it",
        "ama"        : "ask me anything",            "cmv"        : "change my view",             "dae"        : "does anyone else",
        "eli5"       : "explain like i am 5",        "ftfy"       : "fixed that for you",         "iirc"       : "if i recall correctly",
        "mrw"        : "my reaction when",           "mfw"        : "my face when",               "oof"        : "expression of pain",
        "rip"        : "rest in peace",              "roflmao"    : "rolling on floor laughing my ass off",
        "smfh"       : "shaking my fucking head",    "til"        : "today i learned",            "zzz"        : "sleeping"
    }
    words = text.split()
    normalized = [slang_map.get(word.lower(), word) for word in words]
    return ' '.join(normalized)

# To normalize synonyms
def normalize_synonyms(text):
    words = text.split()
    normalized = []
    for word in words:
        found = False
        for key, synonyms in POLITICAL_SYNONYMS.items():
            if word.lower() in synonyms:
                normalized.append(key)
                found = True
                break
        if not found:
            normalized.append(word)
    return ' '.join(normalized)

# To correct spelling
def correct_spelling(text):
    words = text.split()
    corrected = []
    for word in words:
        # Skip mentions and special tokens
        if word.startswith('@') or word.startswith('[USER_'):
            corrected.append(word)
            continue
        corrected_word = spell.correction(word)
        if corrected_word is not None:
            corrected.append(corrected_word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

# To remove special characters
def remove_special_chars(text):
    # Keep basic punctuation
    return re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

# To collapse whitespace
def collapse_whitespace(text):
    return ' '.join(text.split())

# Main preprocessing function
def preprocess_text(text):
    # Handle null values
    if not isinstance(text, str):
        return ""
     # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = remove_urls(text)

    # Process mentions
    text = process_mentions(text)

    # Translate emojis
    text = translate_emojis(text)

    # Expand contractions
    text = expand_contractions(text)

    # Normalize slang
    text = normalize_slang(text)

    # Normalize synonyms
    text = normalize_synonyms(text)

    # Apply ekphrasis processing
    text = ' '.join(text_processor.pre_process_doc(text))

    # Correct spelling
    text = correct_spelling(text)

    # Remove special characters
    text = remove_special_chars(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize with POS tagging
    tokens = lemmatize_tokens(tokens)

    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Collapse whitespace
    text = ' '.join(tokens)
    text = collapse_whitespace(text)

    return text