import os

import regex as re
import requests

filename = 'addresses.txt'
api_url = "https://data.gov.lv/dati/lv"
api_start_endpoint = "/api/3/action/datastore_search?resource_id=54ced227-e043-486c-a4c9-d6b2dc241c4b"
pattern = "([1-9][0-9]?\.\s)?(\p{L}+\s?[-]?){1,}([1-9][0-9]{0,2})?(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]{0,2}\p{L}?)?(\s\u004B[-][1-9])?(\u002F[1-9][0-9]?)?(\s?[-]?\u004B?\u006B?[-]?[1-9][0-9]*)?"
#"([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]*\p{L}?)?(\u002F[1-9][0-9]?)?((\s|[-])?(\u006B|\u006B)[-]?[1-9][0-9]*)?"
#"([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\s)?(\u002F[1-9][0-9])?\p{L}*\s?(\u006B[-])?[1-9]?[0-9]*\p{L}?"

# RegexGenerator + +
# 1. "(?<= )[^_][a-p]*+[^_][r-v][^_][^_][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )|(?<= )[^ ][^ ][^ ]\w\w\w[ \d\w][^>]\w*+ [^ ]*+ \w\d++\w[^ ]*+"
# 2. "(?<= )[A-Za-z][^,][^ ][^iela ][^,][^,][^/][^ ][a-p]*+[^ ] iela [^<]\d++(?= )"
# 3. "(?<= )[^_][a-p]*+[^_][r-v][^_][^/][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )"

address_list = []
file_exists = os.path.isfile(filename)
address_file = open(filename, 'r' if file_exists else 'x', encoding='utf-8')
if file_exists:
    for line in address_file.readlines():
        address = line.strip()
        if address:
            address_list.append(address)
else:
    print("Loading address dataset from data.gov.lv API...")
    response = requests.get(api_url + api_start_endpoint).json()
    loaded_count = 0
    total_count = response["result"]["total"]
    while len(response["result"]["records"]) > 0:
        for address_obj in response["result"]["records"]:
            address = address_obj["adrese"].strip()
            if address:
                address_list.append(address)
                address_file.write(f"{address}\n")
            loaded_count += 1
            if loaded_count % 1000 == 0 or loaded_count == total_count:
                print(f"Address dataset progress: {loaded_count}/{total_count}")
        new_url = api_url + response["result"]["_links"]["next"]
        response = requests.get(new_url).json()
    print("Address dataset loaded!")

address_file.close()

valid_counter = 0
for adrese in address_list:
    isValid = bool(re.fullmatch(pattern, adrese))
    if not isValid:
        print(adrese)
    else:
        valid_counter += 1

print(f"Valid {valid_counter}")
print(f"Invalid {len(address_list) - valid_counter}")

# with open(filename, mode="r", encoding="utf-8-sig") as csvFile:
#     valid_counter = 0
#     dataReader = csv.reader(csvFile)
#     for row in dataReader:
#         address = row[0].replace(';', ' ')
#         isValid = bool(re.fullmatch(pattern, address))
#         if not isValid:
#             print(address)
#         else:
#             valid_counter += 1
#
#     print(f"Valid row count: {valid_counter}")