import shutil
import requests
import pandas as pd

df = pd.read_csv('brennar5_filenames.csv')

for index, row in df.iterrows():
    # img = 'fffc96c79b6b2b3e1863a91d08c5f90335bd9cd5.png'
    img = row[1]

    url = 'http://cs7ns1.scss.tcd.ie/2122/brennar5/pi-project2/' + img

    response = requests.get(url, stream=True)
    with open(img, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    print(img + ' : ' + str(response.status_code))
    del response
