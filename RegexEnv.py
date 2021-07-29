import csv
import re
import gym

class AdresesEnv(gym.Env):
    def __init__(self):
        self.symbols = ['A', 'Ā', 'B', 'C', 'Č', 'D', 'E', 'Ē', 'F', 'G', 'Ģ',  'H', 'I', 'Ī', 'J', 'K', 'Ķ', 'L', 'Ļ', 'M',
                        'N', 'Ņ', 'O', 'P', 'R', 'S', 'Š', 'T', 'U', 'Ū', 'V', 'Z', 'Ž', 'a', 'ā', 'b', 'c', 'č', 'd', 'e', 'ē',
                        'f', 'g', 'ģ', 'h', 'i', 'ī', 'j', 'k', 'ķ', 'l', 'ļ', 'm', 'n', 'ņ', 'o', 'p', 'r', 's', 'š', 't', 'u',
                        'ū', 'v', 'z', 'ž', '-', '_', ',', '.', ';', ':', '!', '?', '*', '@', '0', '1', '2', '3', '4', '5', '6',
                        '7', '8', '9']
        self.address_text = ""
        self.previous_address = self.address_text
        self.action_space = gym.spaces.Discrete(86)
        self.observation_space = gym.spaces.Discrete(2)

    def validateAddress(self, address_text):
        filename = 'latvia.csv'
        pattern = "(((\d+)?\.?\s?[a-zA-ZĀāČčĒēĢģĻļŅņŠšŪūŽžĪīĶķ])\s?)+"

        if address_text == self.previous_address:
            return 0

        if not bool(re.match(pattern, address_text)):
            return 1

        with open(filename, mode="r", encoding="utf-8-sig") as csvFile:
            dataReader = csv.reader(csvFile)
            for row in dataReader:
                isValid = bool(re.match(pattern, row[0]))
                if not isValid:
                    return 1

        return 0

    def step(self, action):
        self.address_text += self.symbols[action]

        state = 1

        if action == 2:
            reward = 1
        else:
            reward = -1

        done = True
        info = {}
        return state, reward, done, info


    def reset(self):
        self.previous_address = self.address_text
        self.address_text = ""
        state = 0
        return state