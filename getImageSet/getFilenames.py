import csv
import requests

payload = {'shortname': 'brennar5', 'myfilename': 'brennar5-challenge-filenames.csv'}
CSV_URL = 'http://cs7ns1.scss.tcd.ie/2122/brennar5/pi-project2/'


with requests.Session() as s:
    download = s.get(CSV_URL, params=payload)

    decoded_content = download.content.decode('utf-8')

    Html_file = open("brennar5_filenames.html", "w")
    Html_file.write(decoded_content)
    Html_file.close()