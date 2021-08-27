import csv
import regex as re
import requests

filename = 'latvia.csv'
api_url = "https://data.gov.lv/dati/lv"
api_start_endpoint = "/api/3/action/datastore_search?resource_id=54ced227-e043-486c-a4c9-d6b2dc241c4b"
pattern = "([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]{0,2}\p{L}?)?(\s\u004B[-][1-9])?(\u002F[1-9][0-9]?)?(\s?[-]?\u004B?\u006B?[-]?[1-9][0-9]*)?"
#"([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]*\p{L}?)?(\u002F[1-9][0-9]?)?((\s|[-])?(\u006B|\u006B)[-]?[1-9][0-9]*)?"
#"([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\s)?(\u002F[1-9][0-9])?\p{L}*\s?(\u006B[-])?[1-9]?[0-9]*\p{L}?"

# RegexGenerator + +
# 1. "(?<= )[^_][a-p]*+[^_][r-v][^_][^_][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )|(?<= )[^ ][^ ][^ ]\w\w\w[ \d\w][^>]\w*+ [^ ]*+ \w\d++\w[^ ]*+"
# 2. "(?<= )[A-Za-z][^,][^ ][^iela ][^,][^,][^/][^ ][a-p]*+[^ ] iela [^<]\d++(?= )"
# 3. "(?<= )[^_][a-p]*+[^_][r-v][^_][^/][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )"

address_list = []
req = requests.get(api_url + api_start_endpoint).json()
while len(req["result"]["records"]) > 0:
    for adreses_obj in req["result"]["records"]:
        adrese = adreses_obj["adrese"]
        if adrese.strip():
            address_list.append(adrese)
    new_url = api_url + req["result"]["_links"]["next"]
    req = requests.get(new_url).json()

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