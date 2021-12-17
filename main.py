import pandas as pd
from preprocess import clean, feature_extraction_tfidf, feature_extraction_count

from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
# add neural network later?

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
train = pd.read_csv("train.csv")
# columns: id, comment_text
test = pd.read_csv("test.csv")
test_Y = pd.read_csv("test_labels.csv")

# -1 labels are not used for accuracy
test_Y = test_Y[test_Y["toxic"] != -1]

test = pd.merge(test, test_Y, on="id", how="left")
test = test.dropna()

# Preprocessing the data
train_clean = clean(train)
test_clean = clean(test)

vectorizer_tfidf = feature_extraction_tfidf(train_clean['clean'].values)

train_tfidf = vectorizer_tfidf.transform(train_clean['clean'].values)
test_tfidf = vectorizer_tfidf.transform(test_clean['clean'].values)

vectorizer_count = feature_extraction_count(train_clean['clean'].values)

train_count = vectorizer_count.transform(train_clean['clean'].values)
test_count = vectorizer_count.transform(test_clean['clean'].values)

# Model Building

# Logistic Regression
# liblinear - limited to one-versus-rest schemes
# optimizer = liblinear, penalty = "l1"
lrcv = LogisticRegressionCV(cv=5, solver="liblinear", random_state=0)

# Naive Bayes
# usually integer features, but tf-idf is okay
mnb = MultinomialNB()

# KNN
knn = KNeighborsClassifier(n_neighbors=3)

# SVM
lscv = LinearSVC(random_state=0)

# Adaboost
adc = AdaBoostClassifier(n_estimators=10, random_state=0)

# ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# # Toxic first



# parameters: classification model obj, X, Y
# returns: model built
def eval_model(model_obj, x, y, test_x, test_y):
    train_x, valid_x, train_y, valid_y = train_test_split(x, y, stratify=y)
    model_obj.fit(train_x, train_y)

    # Evaluate validation scores
    # valid_preds = lrcv.predict(valid_X) #to be used later
    valid_score = model_obj.score(valid_x, valid_y)
    print("Validation Score - " + str(valid_score))

    # Evaluate testing scores
    test_preds = model_obj.predict(test_x)
    print("Accuracy Score - " + str(accuracy_score(test_y, test_preds)))
    print("AUC - " + str(roc_auc_score(test_y, test_preds)))

    return model_obj

offense_type = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
models = [lrcv, mnb, knn, lscv, adc]
for offense in offense_type:
    for model in models:
        print("Building model for " + offense)
        eval_model(model, train_tfidf, train_clean[offense], test_x=test_tfidf, test_y=test_Y[offense])
    # print("Building logistic regression model for " + offense)
    # eval_model(lrcv, train_tfidf, train_clean[offense], test_x=test_tfidf, test_y=test_Y[offense])
    # print("Building naive-bayes model for "+offense)
    # eval_model(mnb, train_count, train_clean[offense], test_x=test_count, test_y=test_Y[offense])
    # print("Building KNN model for "+offense)
    # eval_model(knn, train_tfidf, train_clean[offense], test_x=test_count, test_y=test_Y[offense])
    # print("Building SVM model for "+offense)
    # eval_model(lscv, train_tfidf, train_clean[offense], test_x=test_count, test_y=test_Y[offense])
    # print("Building Adaboost model for "+offense)
    # eval_model(adc, train_tfidf, train_clean[offense], test_x=test_count, test_y=test_Y[offense])
