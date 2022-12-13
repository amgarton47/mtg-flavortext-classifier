import json


def process_data(file):
    data = None
    with open(file) as json_data:
        data = json.loads(json_data.read())

    wanted_keys = ["name", "mana_cost", "cmc", "type_line", "oracle_text", "colors",
                   "color_identity", "legalities", "reserved", "reprint", "set_name", "keywords", "flavor_text"]

    # def contains_all_keys(card):
    #     for key in wanted_keys:
    #         if key not in card.keys():
    #             return False
    #     return True

    # wanted_data = []
    # for item in data:
    #     if contains_all_keys(item):
    #         wanted_data.append(item)

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

    # cards = [json.dumps(card) for card in processed]
    # print(cards)
    with open("cleaned.json", "w") as outfile:
        # outfile.write(cards)
        json.dump(processed, outfile, indent=4)
    # return processed


# json_object = json.dumps(dictionary, indent=4)
process_data("oracle-cards-20221130100205.json")


def load_data(file):
    with open(file) as data:
        return json.loads(data.read())


# print(load_data("cleaned.json")[47])
# for card in load_data("cleaned.json"):
#     tally = 0
#     if "flavor_text" not in card.keys():
#         tally += 1

# print(tally)
