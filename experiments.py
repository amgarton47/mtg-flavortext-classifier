from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
import json
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse


def load_data(file):
    with open(file) as data:
        return json.loads(data.read())


def calculate_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    return f1, accuracy, precision, recall


def filter_train_features(cards, train_features, test_feature):
    features = train_features
    features.append(test_feature)

    def filter_(card):
        for feature in features:
            if feature not in card.keys():
                return False
        return True
    return list(filter(filter_, cards))


def main(train_features, test_feature):
    # train_features = [f for f in [args.oracle_text,
    #                               args.name, args.flavor_text] if f != None]
    processed = load_data("cleaned.json")
    print(len(processed))

    # filter cards that don't have specified train features
    filtered_cards = filter_train_features(
        processed, train_features, test_feature)

    # filter out multi-colored cards
    filtered_monocolored = []
    for x in filtered_cards:
        if "color_identity" in x.keys():
            if len(x["color_identity"]) <= 1:
                filtered_monocolored.append(x)

    test = filtered_monocolored
    # test = filtered_cards

    X = []
    for card in test:
        card_features = []
        for feature in train_features:
            card_features.append(card[feature])
        X.append(card_features)

    print(len(X))

    y = [card[test_feature] for card in test]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # get tf-idf counts
    features_to_train = [" ".join(f[:-1]) for f in X_train]
    print(features_to_train[47])
    print(X_train[47])
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(features_to_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # convert color labels to binary
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    new_y_train = []
    for i in range(len(y_train)):
        new_y_train.append("".join(list(map(str, y_train[i]))))

    # train the model
    clf = MultinomialNB()
    # clf = DummyClassifier(strategy="most_frequent")
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf = svm.LinearSVC()
    # clf = MLPClassifier()
    clf.fit(X_train_tfidf, new_y_train)
    # clf.fit(X_train_counts, new_y_train)

    # process test data
    # flavor_test = [f[0] for f in X_test]
    features_to_test = [" ".join(f[:-1]) for f in X_test]
    X_test_counts = count_vect.transform(features_to_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    y_test = mlb.fit_transform(y_test)
    new_y_test = []
    for i in range(len(y_test)):
        new_y_test.append("".join(list(map(str, y_test[i]))))

    # classify test data
    predictions = clf.predict(X_test_tfidf)

    f1, accuracy, precision, recall = calculate_metrics(
        predictions, new_y_test)

    print(f"f1: {f1}")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")

    # def evaluate(tokens, reference_file, hypothesis_file, verbose=0):
    # reference = set([int(x.rstrip()) for x in reference_file])
    # hypothesis = set([int(x.rstrip()) for x in hypothesis_file])
    # all_tokens = set(range(len(tokens)))

    # # Compute confusion matrix
    # true_positives = hypothesis & reference
    # true_negatives = all_tokens - (hypothesis | reference)
    # false_positives = hypothesis - reference
    # false_negatives = reference - hypothesis

    # # Compute precision, recall, F1
    # precision = len(true_positives) / len(hypothesis)
    # recall = len(true_positives) / len(reference)
    # f = 2*precision*recall/(precision+recall)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--flavor_text",
    #                     help="Use flavor text as a training feature")
    # parser.add_argument("-n", "--name",
    #                     help="Use card name as a training feature")
    # parser.add_argument("-o", "--oracle_text",
    #                     help="Use oracle text as a training feature")
    # parser.add_argument("-c", "--test_feature",
    #                     choices=["colors", "color_identity", "type_line"], default="color_identity")
    # args = parser.parse_args()
    main(["flavor_text", "oracle_text", "name", "type_line"], "color_identity")
    # main(args)
