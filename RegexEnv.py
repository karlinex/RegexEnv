import regex as re
import gym
import numpy as np
import enchant
import requests
import os

class RegexEnv(gym.Env):

    filename = 'addresses.txt'
    pattern = "([1-9][0-9]\.\s)?(\p{L}+\s?[-]?){1,}[1-9][0-9]{0,2}(\.\p{L}+)?\p{Lu}?(\s[1-9][0-9]{0,2}\p{L}?)?(\s\u004B[-][1-9])?(\u002F[1-9][0-9]?)?(\s?[-]?\u004B?\u006B?[-]?[1-9][0-9]*)?"

    address_api_url = "https://data.gov.lv/dati/lv"
    address_api_endpoint = "/api/3/action/datastore_search?resource_id=54ced227-e043-486c-a4c9-d6b2dc241c4b"

    def __init__(self, max_text_length):
        self.lv_address_list = self.loadAddresses() #self.loadAddressesFromCsv()
        self.max_text_length = max_text_length
        self.resetText()
        self.action_space = gym.spaces.Discrete(383)
        self.observation_space = gym.spaces.Box(low=-1, high=382, shape=(max_text_length,), dtype=np.int64)

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

    # Reward method: validates the generated address and returns a reward.
    def validateAddress(self, address_text):
        if not bool(re.fullmatch(self.pattern, address_text)): # if address does not match the Regex pattern
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