import csv
import re

filename = 'latvia.csv'
pattern = "(((\d+)?\.?\s?[a-zA-ZĀāČčĒēĢģĻļŅņŠšŪūŽžĪīĶķ])\s?)+"

with open(filename, mode="r", encoding="utf-8-sig") as csvFile:
    dataReader = csv.reader(csvFile)
    for row in dataReader:
        isValid = bool(re.match(pattern, row[0]))
        if not isValid:
            print(row[0])

print("Done.")