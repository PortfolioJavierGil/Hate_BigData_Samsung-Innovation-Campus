from datetime import datetime
import requests
import json
import time
import pandas as pd
from YouTubeLiveAPI import LiveYouTubeComments

comments_updater = LiveYouTubeComments()
live_urls = [
    'https://www.youtube.com/watch?v=36YnV9STBqc',
    # 'https://www.youtube.com/watch?v=Sj57a1mP_n0', 

]

url_delete = "url"
headers_delete = {
    "Authorization": "XXXX"
}

deleteResponse = requests.delete(url_delete, headers=headers_delete)
print(f"DELETE : {deleteResponse}")

""" if deleteResponse.status_code == 200:
    df = pd.read_csv("../Data/LiveYT/liveComments.csv", sep=';')
    df.drop(df.index, inplace=True)
    df.to_csv('../Data/LiveYT/liveComments.csv', sep=';', index=False) """



# copy "Push URL" from "API Info" in Power BI
url = "https://api.powerbi.com/beta/6afea85d-c323-4270-b69d-a4fb3927c254/datasets/ebb3f0fc-5fe3-443f-9f8c-f23b056d176f/rows?clientSideAuth=0&experience=power-bi&key=%2F1RtRVbkWJhH7IcqkSUhQsPscqtz5HZ6T%2BGICkDuI8jhbMp5%2FXZ3bogcw9VK3bT1z9U8MZy4ThnMwLE%2BVYrzrg%3D%3D"

import random

while True:
    random_number = random.randint(1, 4)
    data = comments_updater.update_comments(live_urls, limit=random_number)

    print(data)
    # post/push data to the streaming API
    headers = {
        "Content-Type": "application/json"
        }
    response = requests.request(
        method="POST",
        url=url,
        headers=headers,
        data=json.dumps(data)
    )
    print(f"POST : {response}")



