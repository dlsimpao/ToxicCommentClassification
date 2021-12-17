import pandas as pd
from nltk.corpus import stopwords
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
train = pd.read_csv("train.csv")
# columns: id, comment_text
test = pd.read_csv("test.csv")

stop = stopwords.words('english')
# sub new line with space
train['clean'] = train['comment_text'].apply(lambda row: re.sub("\n", " ", row))
# sub non-character with ""
train['clean'] = train['clean'].apply(lambda comment: re.sub("[^A-Za-z\' ]+", "", comment))
# remove stopwords from comment
train['clean'] = train['clean'].apply(
    lambda comment: " ".join([word.lower() for word in comment.split() if word not in stop]))

train_clean = train


def clean(df):
    # sub new line with space
    df['clean'] = df['comment_text'].apply(lambda row: re.sub("\n", " ", row))
    # sub non-character with ""
    df['clean'] = df['clean'].apply(lambda comment: re.sub("[^A-Za-z\' ]+", "", comment))
    # remove stopwords from comment
    df['clean'] = df['clean'].apply(
        lambda comment: " ".join([word.lower() for word in comment.split() if word not in stop]))
    print("Cleaning...")
    return df


# test_clean = clean(test) # worksv
# print(test_clean)

# Feature Extraction
vector = TfidfVectorizer()
vector.fit_transform(train_clean['clean'].values)
train_tfidf = vector.transform(train_clean['clean'].values)




def feature_extraction_tfidf(text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(text)
    print("Fitting TFIDF Vectorizer...")
    return vectorizer

def feature_extraction_count(text):
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(text)
    print("Fitting Count Vectorizer...")
    return vectorizer