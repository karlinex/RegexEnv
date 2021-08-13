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

    def __init__(self, text_length):
        self.text_length = text_length
        self.address_text = ""
        self.action_state = np.full(text_length, -1)
        self.action_space = gym.spaces.Discrete(123)
        self.observation_space = gym.spaces.Box(low=-1, high=122, shape=(text_length,), dtype=np.int64)

    # Reward method: validates the generated address and returns a reward.
    def validateAddress(self, address_text):
        if not bool(re.match(self.pattern, address_text)): # if address does not match the Regex pattern
            return 1

        # Compare the generated address with addresses from CSV file
        with open(self.filename, mode="r", encoding="utf-8-sig") as csvFile:
            dataReader = csv.reader(csvFile)
            for row in dataReader:
                levenshtein_distance = enchant.utils.levenshtein(row[0], address_text)
                # TODO: Need to implement point system for the levenshtein distance
                # Average value from all distances ?

        return 0

    def step(self, action):
        self.address_text += chr(action) # Append the symbol to text (corresponding ASCII table character to the action int)
        self.action_state[len(self.address_text) - 1] = action

        reward = self.validateAddress(self.address_text)

        done = (reward == 1)

        return self.action_state, reward, done, {}


    def reset(self):
        self.address_text = ""
        self.action_state = np.full((1, self.text_length), -1)
        return self.action_state