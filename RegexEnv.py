import regex as re
import gym
import numpy as np
import requests
import os

class RegexEnv(gym.Env):

    filename = 'addresses.txt'
    pattern =  "([1-9][0-9]?\.\s)?(\p{L}+\s?[-]?){1,}([1-9][0-9]{0,2})?(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]{0,2}\p{L}?)?(\s\u004B[-][1-9])?(\u002F[1-9][0-9]?)?(\s?[-]?\u004B?\u006B?[-]?[1-9][0-9]*)?"

    address_api_url = "https://data.gov.lv/dati/lv"
    address_api_endpoint = "/api/3/action/datastore_search?resource_id=54ced227-e043-486c-a4c9-d6b2dc241c4b"

    def __init__(self, max_text_length):
        self.char_map = self.getAllowedCharMap()
        self.lv_address_list = self.loadAddresses()
        self.max_text_length = max_text_length
        self.resetText()
        self.previous_text = ""
        self.action_space = gym.spaces.Discrete(len(self.char_map))
        self.observation_space = gym.spaces.Box(low=-1, high=len(self.char_map), shape=(max_text_length,), dtype=np.int64)

    def resetText(self):
        self.address_text = ""
        self.action_state = np.full((self.max_text_length,), -1)

    def loadAddresses(self):
        file_exists = os.path.isfile(self.filename)
        address_file = open(self.filename, 'r' if file_exists else 'x', encoding='utf-8')
        address_list = []

        if file_exists:
            for line in address_file.readlines():
                address = line.strip()
                if address:
                    address_list.append(address)
        else:
            print("Loading address dataset from data.gov.lv API...")
            response = requests.get(self.address_api_url + self.address_api_endpoint).json()
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
                new_url = self.address_api_url + response["result"]["_links"]["next"]
                response = requests.get(new_url).json()
            print("Address dataset loaded!")

        address_file.close()
        return address_list

    def getAllowedCharMap(self):
        ranges = [(32, 86), (90, 118), (122, 126), (256, 257), (268, 269), (274, 275), (290, 291), (298, 299), (310, 311), (315, 316), (325, 326), (352, 353), (362, 363), (381, 381)]
        allowed_chars = []
        for r in ranges:
            for n in range(r[0], r[1] + 1):
                allowed_chars.append(n)
        return allowed_chars

    # Reward method: validates the generated address and returns a reward.
    def validateAddress(self, address_text):
        pattern_match = bool(re.fullmatch(self.pattern, address_text))
        if not pattern_match:
            if address_text == self.previous_text:
                return -100
            else:
                return 0

        if len(address_text) == self.max_text_length and address_text == self.previous_text:
            return -100

        min_diff = None
        for address in self.lv_address_list:
            diff = self.levenshtein(address_text, address) #enchant.utils.levenshtein(address_text, address)
            if min_diff == None or min_diff > diff:
                min_diff = diff

        return min_diff

    def step(self, action):
        self.address_text += chr(self.char_map[action]) # Append the symbol to text (corresponding ASCII table character to the action int)
        address_length = len(self.address_text)
        self.action_state[address_length - 1] = action
        reward = self.validateAddress(self.address_text)
        done = (reward == 0 or len(self.address_text) == self.max_text_length)
        if address_length >= 2 and action == self.action_state[address_length - 2]:
            reward = -10

        return self.action_state, reward, done, {}

    def reset(self):
        self.previous_text = self.address_text
        self.resetText()
        return self.action_state

    def levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)
        if not s1:
            return len(s2)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]