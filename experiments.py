

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
import json
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
from sklearn.feature_extraction import _stop_words
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


# read in json data from pre-processed file
def load_data(file):
    with open(file) as data:
        return json.loads(data.read())


# calculate and return performance metrics given predictions and true labels
def calculate_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_pred))
    precision = precision_score(
        y_true, y_pred, labels=np.unique(y_pred), average="macro")
    recall = recall_score(y_true, y_pred, average="macro",
                          labels=np.unique(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return f1, accuracy, precision, recall, conf_matrix


# return card data, filtering out unwanted features
def filter_train_features(cards, train_features, test_feature):
    features = train_features
    features.append(test_feature)

    def filter_(card):
        for feature in features:
            if feature not in card.keys():
                return False
        return True
    return list(filter(filter_, cards))


# prepares train and test data by performing feature exraction and transformation
def get_train_test_data(test, train_features, test_feature):
    X = []
    for card in test:
        card_features = []
        for feature in train_features:
            card_features.append(card[feature])
        X.append(card_features)

    y = [card[test_feature] for card in test]

    # create 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2)

    # stop_words are ignored during tokenization
    english_stop_words = _stop_words.ENGLISH_STOP_WORDS
    custom_stop_words = ["creature", "card",
                         "target", "control", "turn", "creatures"]
    extra = ["simic", "rakdos", "gruul", "azorius", "dimir",
             "selesnya", "izzet", "golgari", "orzhov", "boros", "bant", "esper", "grixis", "jund", "naya", "abzan", "jeskai", "sultai", "mardu", "temur"]
    stop_words = english_stop_words.union(custom_stop_words).union(extra)

    # compute tf-idf matrix
    features_to_train = [" ".join(f[:-1]) for f in X_train]
    count_vect = CountVectorizer(ngram_range=(
        1, 2), stop_words=stop_words, lowercase=True)
    X_train_counts = count_vect.fit_transform(features_to_train)
    tfidf_transformer = TfidfTransformer(
        smooth_idf=True, sublinear_tf=True, norm=None)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # convert color labels to binary
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    new_y_train = []
    for i in range(len(y_train)):
        new_y_train.append("".join(list(map(str, y_train[i]))))

    # process test data
    features_to_test = [" ".join(f[:-1]) for f in X_test]
    X_test_counts = count_vect.transform(features_to_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    y_test = mlb.fit_transform(y_test)
    new_y_test = []
    for i in range(len(y_test)):
        new_y_test.append("".join(list(map(str, y_test[i]))))

    return X_train_tfidf, new_y_train, X_test_tfidf, new_y_test, count_vect


# converts character color_identity representation into binary
def convert_binary_to_colors(binary_color):
    colors = ['B', 'C', 'G', 'R', 'U', 'W']

    color_identity = ""
    for i in range(6):
        if binary_color[i] == "1":
            color_identity += colors[i]

    return color_identity


# transforms binary string into binary digit
def numberify(string):
    num = []
    for c in string:
        num.append(int(c))
    return num


# calcualtes and returns the hamming score given predictions and true labels
def hamming(y_pred, y_true):
    temp = 0
    for i in range(len(y_true)):
        true = numberify(y_true[i])
        pred = numberify(y_pred[i])
        a = np.logical_and(true, pred)
        b = np.logical_or(true, pred)
        temp += sum(a) / sum(b)
    return temp / len(y_true)


def main(train_features, test_feature):
    processed = load_data("cleaned_new.json")

    # filter cards that don't have specified train features
    filtered_cards = filter_train_features(
        processed, train_features, test_feature)

    # filter out multi-colored cards
    # filtered_monocolored = []
    # for x in filtered_cards:
    #     if "color_identity" in x.keys():
    #         if len(x["color_identity"]) <= 1:
    #             filtered_monocolored.append(x)

    # cards = filtered_monocolored
    cards = filtered_cards

    X_train, y_train, X_test, y_test, count_vect = get_train_test_data(
        cards, train_features, test_feature)

    # train the model
    clf = MultinomialNB()
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = DummyClassifier(strategy="most_frequent")
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf = svm.LinearSVC()
    # clf = MLPClassifier()
    clf.fit(X_train, y_train)

    # classify test data
    predictions = clf.predict(X_test)

    print("\n----------Dataset Class Counts----------")
    class_counts = {convert_binary_to_colors(
        key): value for key, value in dict(Counter(y_train) + Counter(y_test)).items()}
    print(class_counts)

    print(predictions)

    print("\n----------Model Metrics----------")
    f1, accuracy, precision, recall, conf_matrix = calculate_metrics(
        predictions, y_test)
    print(f"f1: {f1}")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print("hamming:", hamming(predictions, y_test))
    # print(conf_matrix)
    # print(conf_matrix.diagonal()/conf_matrix.sum(axis=1))

    if type(clf).__name__ == "MultinomialNB":
        print("\n----------Most Relevant Features Per Class----------")
        for i in range(len(clf.classes_)):
            zipped = list(zip(count_vect.get_feature_names_out(),
                              clf.feature_log_prob_[i]))
            print(sorted(zipped, key=lambda x: x[1], reverse=True)[
                :3], convert_binary_to_colors(clf.classes_[i]))


if __name__ == "__main__":
    main(["flavor_text", "oracle_text", "name", "type_line"], "color_identity")
