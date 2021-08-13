import csv
import re

filename = 'latvia.csv'
pattern = "(?<= )[^_][a-p]*+[^_][r-v][^_][^_][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )|(?<= )[^ ][^ ][^ ]\w\w\w[ \d\w][^>]\w*+ [^ ]*+ \w\d++\w[^ ]*+"
# RegexGenerator + +
# 1. "(?<= )[^_][a-p]*+[^_][r-v][^_][^_][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )|(?<= )[^ ][^ ][^ ]\w\w\w[ \d\w][^>]\w*+ [^ ]*+ \w\d++\w[^ ]*+"
# 2. "(?<= )[A-Za-z][^,][^ ][^iela ][^,][^,][^/][^ ][a-p]*+[^ ] iela [^<]\d++(?= )"
# 3. "(?<= )[^_][a-p]*+[^_][r-v][^_][^/][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )"

with open(filename, mode="r", encoding="utf-8-sig") as csvFile:
    dataReader = csv.reader(csvFile)
    for row in dataReader:
        isValid = bool(re.match(pattern, row[0]))
        if not isValid:
            print(row[0])

print("Done.")