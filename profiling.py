from pymongo import MongoClient
from insertIntoDb import insert
from getTweets import lookupTweets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import random

client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

if db.collection.count() < lookupTweets():
    db.collection.remove()
    insert()
else:
    print("Hello world", lookupTweets())
    # Partition in to Training and Test Set.
    data = [db["StatusID"]+"-"+db["Sex"] for db in db.collection.find()]
    random.shuffle(list(data))
    training_data = data[:int((len(data)+1)*.80)]
    test_data = data[int(len(data)*.80+1):]

    # Tokenizing & Filtering the text
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(training_data)

    # Fit and Transform the data
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # Train the classifer using various methods


