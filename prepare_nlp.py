import unicodedata
import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('omw-1.4')


def basic_clean(filthy_data):
    filthy_data = filthy_data.lower()
    filthy_data = unicodedata.normalize('NFKD', filthy_data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    clean_data = re.sub(r"[^a-z0-9'\s]", "", filthy_data)
    return clean_data


def tokenize(data):
    tokenizer = ToktokTokenizer()
    data = tokenizer.tokenize(data, return_str=True)
    return data


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    """ This function takes in a string, optional extra_words and exclued_words parameters with default empty lists
     and returns a string """
    stopword_list = stopwords.words('english')
    # use set casting to remove any excluded stopwords
    stopword_set = set(stopword_list) - set(exclude_words)
    # add in extra words to stopwords set using a union
    stopword_set = stopword_set.union(set(extra_words))
    # split the document by spaces
    words = string.split()
    # every word in our document that is not a stopword
    filtered_words = [word for word in words if word not in stopword_set]
    # join it back together with spaces
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords


def lemmatize(data):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in data.split()]
    lemmatized_data = ' '.join(lemmas)
    return lemmatized_data


def stem(data):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in data.split()]
    stemmed_data = ' '.join(stems)
    return stemmed_data


def cleanse(dataframe, col='', stemm=True, lem=True, extra_words=[], exclude_words=[]):
    df = dataframe.copy()
    df['clean'] = df[col].apply(basic_clean)
    df['clean'] = df['clean'].apply(remove_stopwords, extra_words=extra_words, exclude_words=exclude_words)
    if stemm:
        df['stemmed'] = df.clean.apply(stem)
    if lem:
        df['lemmatized'] = df.clean.apply(lemmatize)
    return df
