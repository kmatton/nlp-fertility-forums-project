import re

from gensim.utils import simple_preprocess
import spacy
import nltk
import numpy as np


LIWC_2015_PATH = './LIWC.2015.all'


def read_file_by_lines(filename):
    """
    Read a file into a list of lines
    """
    with open(filename, "r") as f:
        return f.read().splitlines()


# prepare stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
# remove words in LIWC from stopwords list
liwc_lines = read_file_by_lines(LIWC_2015_PATH)
liwc_words = set([line.split(' ')[0] for line in liwc_lines])
liwc_star_words = set([lw for lw in liwc_words if lw[-1] == '*'])
sw = [word for word in sw if word not in liwc_words]
sw = [word for word in sw if not np.any([word.startswith(lw[:-1]) for lw in liwc_star_words])]
STOP_WORDS = sw

# prepare sentence tokenizer
nltk.download('punkt')
SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

# prepare spacy NLP lemmatizer
NLP = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  # only keep tagger component for efficiency


def process_single_post_text(text, do_lemmatize=True):
    """
    Process text from single post.
    :param text: string of text from post or post title
    :param do_lemmatize: if True, apply lemmatization
    :return: sents: cleaned text broken up into sentences
    """
    text = remove_special_chars(text)
    sents = split_into_sentences(text)
    # remove punctuation, stopwords, and convert to lowercase
    sents = clean_and_tokenize(sents)
    if do_lemmatize:
        # lemmatize words in each sentence
        for idx, sent in enumerate(sents):
            sents[idx] = lemmatize(sent)
    return sents


def lemmatize(doc):
    """
    Lemmatize words in doc.
    :param: doc: list of words (strs).
    """
    doc = NLP(" ".join(doc))
    doc = [word.lemma_ for word in doc]
    return doc


def clean_and_tokenize(sentences):
    """
    Converts each sentence (str) in sentences (list of strs) into a list of words.
    Also cleans text by removing punctuation, removing stopwords, and converting to lowercase.
    """
    # deacc=True to remove punctuation
    sentences = [[word for word in simple_preprocess(sent, deacc=True) if word not in STOP_WORDS] for sent in sentences]
    return sentences


def split_into_sentences(text):
    """
    Split text (str) into list of sentences
    """
    return SENT_TOKENIZER.tokenize(text)


def remove_special_chars(text):
    """
    Remove special characters from text common in Reddit Posts as well as emails, URLS, and IP addresses.
    Adapted from https://github.com/LoLei/redditcleaner/blob/master/redditcleaner/__init__.py
    """
    # Newlines (replaced with space to preserve cases like word1\nword2)
    text = re.sub(r'\n+', ' ', text)
    # Remove resulting ' '
    text = text.strip()
    text = re.sub(r'\s\s+', ' ', text)

    # emails
    text = re.sub('\S*@\S*\s?', '', text)

    # > Quotes
    text = re.sub(r'\"?\\?&?gt;?', '', text)

    # Bullet points/asterisk (bold/italic)
    text = re.sub(r'\*', '', text)
    text = re.sub('&amp;#x200B;', '', text)

    # things in parantheses or brackets
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # remove URLS
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Strikethrough
    text = re.sub('~', '', text)

    # Spoiler, which is used with < less-than (Preserves the text)
    text = re.sub('&lt;', '', text)
    text = re.sub(r'!(.*?)!', r'\1', text)

    # Code, inline and block
    text = re.sub('`', '', text)

    # Superscript (Preserves the text)
    text = re.sub(r'\^\((.*?)\)', r'\1', text)

    # Table
    text = re.sub(r'\|', ' ', text)
    text = re.sub(':-', '', text)

    # Heading
    text = re.sub('#', '', text)

    return text
