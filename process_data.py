from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
import json
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
import argparse


def process_data(file):
    data = None
    with open(file) as json_data:
        data = json.loads(json_data.read())

    wanted_keys = ["name", "mana_cost", "cmc", "type_line", "oracle_text", "colors",
                   "color_identity", "legalities", "reserved", "reprint", "set_name", "keywords", "flavor_text"]
    processed = []
    for card in data:
        processed_card = {your_key: card[your_key]
                          for your_key in wanted_keys if your_key in card.keys()}
        processed.append(processed_card)
    for card in processed:
        if "color_identity" in card.keys() and card["color_identity"] == []:
            card["color_identity"] = ['C']
        if "colors" in card.keys() and card["colors"] == []:
            card["colors"] = ['C']
    return processed


def main(train_features, test_feature):
    # train_features = [f for f in [args.oracle_text,
    #                               args.name, args.flavor_text] if f != None]
    processed = process_data("oracle-cards-20221130100205.json")

    # filter cards that don't have specified train features
    def filter_train_features(card):
        for feature in train_features:
            if feature not in card.keys():
                return False
        return True
    cards_with_flavor = list(
        filter(filter_train_features, processed))

    # filter out multi-colored cards
    flavor_monocolored = []
    for x in cards_with_flavor:
        if "color_identity" in x.keys():
            if len(x["color_identity"]) <= 1:
                flavor_monocolored.append(x)

    test = flavor_monocolored
    # test = cards_with_flavor

    X = []
    for card in test:
        card_features = []
        for feature in train_features:
            card_features.append(card[feature])
        X.append(card_features)

    y = [card[test_feature] for card in test]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # get tf-idf counts
    flavor = [f[0] for f in X_train]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(flavor)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # convert color labels to binary
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    new_y_train = []
    for i in range(len(y_train)):
        new_y_train.append("".join(list(map(str, y_train[i]))))

    # train the model
    clf = MultinomialNB().fit(X_train_tfidf, new_y_train)

    # process test data
    flavor_test = [f[0] for f in X_test]
    X_test_counts = count_vect.transform(flavor_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    y_test = mlb.fit_transform(y_test)
    new_y_test = []
    for i in range(len(y_test)):
        new_y_test.append("".join(list(map(str, y_test[i]))))

    # classify test data
    predictions = clf.predict(X_test_tfidf)

    # calculate accuracy
    total = 0
    correct = 0
    for i in range(len(predictions)):
        print(X_test[i], new_y_test[i], predictions[i], mlb.classes_)
        if predictions[i] == new_y_test[i]:
            correct += 1
        total += 1

    print(correct/total)

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

    # if verbose >= 1:
    #     words = tokens
    #     if verbose == 2:
    #         for i in true_positives:
    #             concordance(words, i, 'TP')
    #         for i in true_negatives:
    #             concordance(words, i, 'TN')
    #     else:
    #         for i in false_positives:
    #             concordance(words, i, 'FP')
    #         for i in false_negatives:
    #             concordance(words, i, 'FN')

    # print("TP: {:7d}".format(len(true_positives)), end="")
    # print("\tFN: {:7d}".format(len(false_negatives)))

    # print("FP: {:7d}".format(len(false_positives)), end="")
    # print("\tTN: {:7d}".format(len(true_negatives)))

    # print()

    # print("PRECISION: {:5.2%}".format(precision), end="")
    # print("\tRECALL: {:5.2%}".format(recall), end="")
    # print("\tF: {:5.2%}".format(f))


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
    main(["flavor_text", "name", "oracle_text"], "color_identity")
    # main(args)
