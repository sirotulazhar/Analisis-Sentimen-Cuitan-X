
import re
import string
import pandas as pd

from num2words import num2words
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==========================================
# 1. Memuat kamus preprocessing
# ==========================================

emoji_df = pd.read_csv("Dataset/emoji_dict.csv")
emoji_dict = dict(zip(emoji_df["Emoji"], emoji_df["Arti"]))

alay_df = pd.read_csv(
    "Dataset/new_kamusalay.csv",
    header=None,
    encoding="latin-1"
)
alay_df = alay_df.rename(
    columns={0: "original", 1: "replacement"}
)
alay_dict = dict(zip(alay_df["original"], alay_df["replacement"]))

slang_df = pd.read_csv(
    "Dataset/slang_words.csv",
    encoding="latin-1"
)
slang_dict = dict(
    zip(slang_df["original"], slang_df["replacement"])
)

stopword_df = pd.read_csv(
    "Dataset/stopwordbahasa.csv",
    header=None
)
stopword_df = stopword_df.rename(columns={0: "stopword"})
stopword_set = set(stopword_df["stopword"].values)


# ==========================================
# 2. Inisialisasi stemmer
# ==========================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ==========================================
# 3. Fungsi preprocessing
# ==========================================

def ganti_emoji_dengan_arti(text):
    for emoji, arti in emoji_dict.items():
        text = text.replace(emoji, str(arti))
    return text


def remove_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002500-\U00002BEF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F926-\U0001F937"
        "\U00010000-\U0010FFFF"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub("", text)


def nums_to_words(text):
    new_text = []

    for word in text.split():
        if word.isdigit():
            word = num2words(int(word), lang="id")
        new_text.append(word)

    return " ".join(new_text)


def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    no_punct = [char for char in text if char not in string.punctuation]
    return "".join(no_punct)


def remove_unnecessary_char(text):
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = text.replace("\\", "")
    text = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+))", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"RT", " ", text)
    text = re.sub(r"user", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"@[\w_]+", "", text)
    text = re.sub(r"#[\w_]+", "", text)
    return text


def remove_nonaplhanumeric(text):
    return re.sub(r"[^0-9a-zA-Z]+", " ", text)


def normalize_alay(text):
    return " ".join(
        alay_dict.get(word, word)
        for word in text.split(" ")
    )


def normalize_slang(text):
    return " ".join(
        slang_dict.get(word, word)
        for word in text.split(" ")
    )


def remove_stopword(text):
    text = " ".join(
        " " if word in stopword_set else word
        for word in text.split(" ")
    )
    text = re.sub(r"  +", " ", text)
    return text.strip()


def stemming(text):
    return stemmer.stem(text)


# ==========================================
# 4. Fungsi utama preprocessing
# ==========================================

def preprocess(text):
    text = str(text)

    text = ganti_emoji_dengan_arti(text)
    text = remove_emoji(text)
    text = nums_to_words(text)
    text = remove_unnecessary_char(text)
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_nonaplhanumeric(text)
    text = normalize_alay(text)
    text = normalize_slang(text)
    text = remove_stopword(text)
    text = stemming(text)

    return text
