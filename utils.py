# utils.py
import nltk, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download once (quiet)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove punctuation
    text = ''.join([c for c in text if c not in string.punctuation])
    # tokenize + remove stopwords + stem
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)
