from pymongo import MongoClient
from insertIntoDb import insert
from getTweets import lookupTweets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import random

client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

if db.collection.count() < lookupTweets():
    db.collection.remove()
    insert()
else:
    print("Hello world", lookupTweets())
    # Partition in to Training and Test Set for gender.
    data = [db["StatusID"]+"---"+db["Sex"] for db in db.collection.find()]
    random.shuffle(data)

    train_data = data[:int((len(data)+1)*.80)]
    test_data = data[int(len(data)*.80+1):]
    train_labels = [data.split("---")[1] for data in train_data]
    test_labels = [data.split("---")[1] for data in test_data]


    # Tokenizing & Filtering the text
    # min_df=5, discard words appearing in less than 5 documents
    # max_df=0.8, discard words appering in more than 80% of the documents
    # sublinear_tf=True, use sublinear weighting
    # use_idf=True, enable IDF

    vectorizer = TfidfVectorizer(min_df=3,
                             max_df = 0.8,
                             sublinear_tf=True,
                             use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data) # Create the vocabulary and the feature weights from the training data
    test_vectors = vectorizer.transform(test_data) # Create the feature weights for the test data
    print(len(train_labels))
    # Classification using SVM kernel=rbf
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, train_labels)
    prediction_rbf = classifier_rbf.predict(test_vectors)
    print(classification_report(test_labels, prediction_rbf))


    # # Fit and Transform the data
    # # tfidf_transformer = TfidfTransformer()
    # # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # # Train the classifer using various methods


