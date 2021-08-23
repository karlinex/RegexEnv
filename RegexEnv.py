import csv
import re
import gym
import numpy as np
import enchant

class RegexEnv(gym.Env):

    filename = 'latvia.csv'
    pattern = "(((\d+)?\.?\s?[a-zA-ZĀāČčĒēĢģĻļŅņŠšŪūŽžĪīĶķ])\s?)+"

    # RegexGenerator + +
    # "(?<= )[^_][a-p]*+[^_][r-v][^_][^_][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )|(?<= )[^ ][^ ][^ ]\w\w\w[ \d\w][^>]\w*+ [^ ]*+ \w\d++\w[^ ]*+"
    # "(?<= )[A-Za-z][^,][^ ][^iela ][^,][^,][^/][^ ][a-p]*+[^ ] iela [^<]\d++(?= )"
    # "(?<= )[^_][a-p]*+[^_][r-v][^_][^/][^ ][a-p]*+\w[^_][a-p]*+ \d++(?= )"

    def __init__(self, max_text_length):
        self.lv_address_list = self.loadAddressesFromCsv()
        self.max_text_length = max_text_length
        self.resetText()
        self.action_space = gym.spaces.Discrete(383)
        self.observation_space = gym.spaces.Box(low=-1, high=382, shape=(max_text_length,), dtype=np.int64)

    def resetText(self):
        self.address_text = ""
        self.action_state = np.full((self.max_text_length,), -1)

    def loadAddressesFromCsv(self):
        address_list = []
        with open(self.filename, mode="r", encoding="utf-8-sig") as csvFile:
            data_reader = csv.reader(csvFile)
            for row in data_reader:
                address_list.append(row[0].replace(';', ' '))

        return address_list

    # Reward method: validates the generated address and returns a reward.
    def validateAddress(self, address_text):
        if not bool(re.match(self.pattern, address_text)): # if address does not match the Regex pattern
            return 0

        min_diff = None
        for address in self.lv_address_list:
            diff = enchant.utils.levenshtein(address_text, address)
            if min_diff == None or min_diff > diff:
                min_diff = diff

        return min_diff

    def step(self, action):
        self.address_text += chr(action) # Append the symbol to text (corresponding ASCII table character to the action int)
        self.action_state[len(self.address_text) - 1] = action
        reward = self.validateAddress(self.address_text)
        done = (reward == 0 or len(self.address_text) == self.max_text_length)

        return self.action_state, reward, done, {}

    def reset(self):
        self.resetText()
        return self.action_state