# This script will scrape for active fighters and output a csv of active fighters

import os
import pandas as pd
from scrapy import cmdline


cmdline.execute("scrapy runspider active_fighters.py -o result.json -t json".split())

active = pd.read_json('result.json')

active_fighters = []

for i in range(len(active)):
    for element in active['Fighter'][i]:
        active_fighters.append(element)


active_fighters = pd.DataFrame(active_fighters, columns=['Fighter'])

active_fighters.to_csv('active_fighters.csv')
