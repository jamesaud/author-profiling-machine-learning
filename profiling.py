from pymongo import MongoClient
from insertIntoDb import insert
from getTweets import lookupTweets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import random
import numpy as np

client = MongoClient('localhost:27017', serverSelectionTimeoutMS=1000)

db = client['user-details']
collection = db['status']

if db.collection.count() < lookupTweets():
    db.collection.remove()
    insert()
else:
    print("Hello world", lookupTweets())
    # Partition in to Training and Test Set for gender.

    data = [db["StatusID"] + "-GENDER-" + db["Sex"] for db in db.collection.find()]
    random.seed(1234)  # randomizing data
    random.shuffle(data)

    train_data = data[:int((len(data) + 1) * .80)]
    test_data = data[int(len(data) * .80 + 1):]

    train_data = train_data[:3000]
    test_data = test_data[:300]

    train_labels = [data.split("-GENDER-")[1] for data in train_data]
    test_labels = [data.split("-GENDER-")[1] for data in test_data]
    train_data = [data.split("-GENDER-")[0] for data in train_data]
    test_data = [data.split("-GENDER-")[0] for data in test_data]


    def convert_to_integer(train_labels):
        train_labels_int = [0 if x == "MALE" else 1 for x in train_labels]
        return train_labels_int


    # Tokenizing & Filtering the text
    # min_df=5, discard words appearing in less than 5 documents
    # max_df=0.8, discard words appering in more than 80% of the documents
    # sublinear_tf=True, use sublinear weighting
    # use_idf=True, enable IDF
    vectorizer = TfidfVectorizer(min_df=8,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    # Create the vocabulary and the feature weights from the training data
    train_vectors = vectorizer.fit_transform(train_data)
    # Create the feature weights for the test data
    test_vectors = vectorizer.transform(test_data)

    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'kernel': ['linear']},
        {'C': [0.01, 0.1, 1, 10, 50, 100, 500, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # Finding the Best Parameters
    classifier_best = svm.SVC()
    classifier_cv = GridSearchCV(classifier_best, param_grid, scoring='accuracy')
    classifier_cv.fit(train_vectors, train_labels)
    print("Best Score for given dataset for SVM Classifier - ", classifier_cv.best_score_)
    print("Best Parameter for given dataset for SVM Classifier - ", classifier_cv.best_params_)

    # Classification with SVM, kernel=linear is the best
    classifier_linear = svm.SVC(C=0.01, kernel='linear')
    classifier_linear.fit(train_vectors, train_labels)
    prediction_linear = classifier_linear.predict(test_vectors)
    print(classification_report(test_labels, prediction_linear))

    # Plotting the data for different values of C
    def rmse_cv(model):
        rmse = np.sqrt(cross_val_score(classifier_linear, train_vectors, train_labels, scoring='accuracy', cv=5))
        return (rmse)




    train_labels_int = convert_to_integer(train_labels)
    c_lst = [0.01, 0.1, 1, 10, 50, 100, 500, 1000]
    crossval_ridge = [np.mean(rmse_cv(classifier_linear)) for x in c_lst]
    import pandas as pd
    import matplotlib.pyplot as plt

    crossval_ridge = pd.Series(crossval_ridge, index=c_lst)
    crossval_ridge.plot(title="Validation")
    plt.xlabel("c_values")
    plt.ylabel("accuracy")
    plt.show()

    # Cross Validation Scores
    # scores = cross_val_score(classifier_linear, train_vectors, train_labels, cv=10)
    # from sklearn.model_selection import cross_val_score
    # print("Accuracy",scores.mean(),scores.std()* 2)

    # Predictions using cross validation
    # from sklearn.model_selection import cross_val_predict
    # import sklearn.metrics as metrics
    # predicted = cross_val_predict(classifier_linear, train_vectors, train_labels, cv=10)
    # print("Prediction Accuracy for SVM linear",metrics.accuracy_score(train_labels, predicted))



    # Classification with Naive Bayes

    classifier_nb = MultinomialNB()
    classifier_nb.fit(train_vectors, train_labels)
    prediction_nb = classifier_nb.predict(test_vectors)
    print(classification_report(test_labels, prediction_nb))

    # Plotting the data for different values of C
    # def rmse_cv(model):
    #     rmse = np.sqrt(cross_val_score(classifier_nb, train_vectors, train_labels, scoring='accuracy', cv=5))
    #     return (rmse)
    #
    # from sklearn.model_selection import cross_val_score
    # import numpy as np
    # train_labels_int = convert_to_integer(train_labels)
    # crossval_ridge = [np.mean(rmse_cv(classifier_linear)) for x in c_lst]
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # crossval_ridge = pd.Series(crossval_ridge, index=c_lst)
    # crossval_ridge.plot(title="Validation")
    # plt.xlabel("c_values")
    # plt.ylabel("accuracy")
    # plt.show()


    # Finding the best parameters
    k = np.arange(20) + 1
    parameters = {'n_neighbors': k}
    classifier_knearest = KNeighborsClassifier()
    classifier_cv_kn = GridSearchCV(classifier_knearest, parameters, scoring='accuracy', cv=10)
    classifier_cv_kn.fit(train_vectors, train_labels)
    print("Best Score for given dataset for KNearest Classifier - ", classifier_cv_kn.best_score_)
    print("Best Parameter for given dataset for KNearest Classifier - ", classifier_cv_kn.best_params_)

    # Classification with KNearest Neighbors with parameter 3
    classifier_knearest = KNeighborsClassifier(classifier_cv_kn.best_params_["n_neighbors"])
    classifier_knearest.fit(train_vectors, train_labels)
    prediction_knearest = classifier_knearest.predict(test_vectors)
    print(classification_report(test_labels, prediction_knearest))





    ############### AGE PREDICTION ###############
    # data_ag = [db["StatusID"] + "-AGE-" + db["Age"] for db in db.collection.find()]
    # random.shuffle(data)
    #
    # train_data_age = data[:int((len(data) + 1) * .80)]
    # test_data = data[int(len(data) * .80 + 1):]
    #
    # train_data = train_data[:350]
    # test_data = test_data[:350]
    #
    # train_labels = [data.split("-AGE-")[1] for data in train_data]
    # test_labels = [data.split("-AGE-")[1] for data in test_data]
    # train_data = [data.split("-AGE-")[0] for data in train_data]
    # test_data = [data.split("-AGE-")[0] for data in test_data]
    #
    # # Tokenizing & Filtering the text
    # # min_df=5, discard words appearing in less than 5 documents
    # # max_df=0.8, discard words appering in more than 80% of the documents
    # # sublinear_tf=True, use sublinear weighting
    # # use_idf=True, enable IDF
    #
    # vectorizer = TfidfVectorizer(min_df=8,
    #                              max_df=0.8,
    #                              sublinear_tf=True,
    #                              use_idf=True)
    # train_vectors = vectorizer.fit_transform(
    #     train_data)  # Create the vocabulary and the feature weights from the training data
    # test_vectors = vectorizer.transform(test_data)  # Create the feature weights for the test data
    # print(len(train_labels))
    #
    # # Classification with SVM, kernel=linear
    # classifier_linear = svm.SVC(kernel='linear')
    # classifier_linear.fit(train_vectors, train_labels)
    # prediction_linear = classifier_linear.predict(test_vectors)
    # print(classification_report(test_labels, prediction_linear))

    # # Classification with Naive Bayes
    # from sklearn.naive_bayes import MultinomialNB
    # classifier_nb = MultinomialNB()
    # classifier_nb.fit(train_vectors,train_labels)
    # prediction_nb = classifier_nb.predict(test_vectors)
    # print(classification_report(test_labels, prediction_nb))
    #
    # # Classification with KNearest Neighbors with parameter 3
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier_knearest = KNeighborsClassifier(n_neighbors=3)
    # classifier_knearest.fit(train_vectors,train_labels)
    # prediction_knearest = classifier_knearest.predict(test_vectors)
    # print(classification_report(test_labels, prediction_knearest))
    #

    # # Fit and Transform the data
    # # tfidf_transformer = TfidfTransformer()
    # # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # # Train the classifer using various methods
