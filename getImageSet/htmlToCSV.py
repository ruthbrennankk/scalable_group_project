import codecs
from bs4 import BeautifulSoup as BeautifulSoup
import pandas as pd

f = codecs.open("brennar5_filenames.html", 'r')

contents = f.read()
soup = BeautifulSoup(contents, 'html.parser')

names = []
count = 0

for a in soup.find_all('a', href=True):
    names.append(a['href'])
    count = count + 1

print('count =' + str(count))

# Calling DataFrame constructor on list
df = pd.DataFrame(names)
df.to_csv('brennar5_filenames.csv')