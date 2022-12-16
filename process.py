import json


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

    with open("cleaned_new.json", "w") as outfile:
        json.dump(processed, outfile, indent=4)


# process_data("oracle-cards-20221130100205.json")
process_data("oracle-cards-20221215220238.json")  # newer data (12/15/2022)


def load_data(file):
    with open(file) as data:
        return json.loads(data.read())
