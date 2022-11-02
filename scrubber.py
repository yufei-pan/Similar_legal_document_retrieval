# %%
from bs4 import BeautifulSoup
import sys
import os
import requests
import re
from anyascii import anyascii
import pandas as pd
year = sys.argv[1] if len(sys.argv) > 1 else '1760'
print(year)


# https://caselaw.findlaw.com/court/us-supreme-court/years/1760



# %%
#year = '2008'

# %%
soup = BeautifulSoup(requests.get('https://caselaw.findlaw.com/court/us-supreme-court/years/'+year).text, 'html.parser')

# %%
description = soup.find_all(attrs={"data-label": "Description"})

# %%
links = []
for item in description:
    links.append(item.a.get('href'))

# %%
links

# %%
outList = []

# %%
for link in links:
    print('Fetching '+link)
    soup = BeautifulSoup(requests.get(link).text, 'html.parser')
    dates = soup.find_all('h3')[2].get_text().replace('Argued:','').partition('Decided:')
    line = [soup.find_all('h3')[0].get_text(),soup.find_all('h3')[1].get_text(),dates[0].strip(),dates[2].strip()]
    line.append(anyascii('\n'.join([someTxt.get_text().strip().replace('\xa0',' ').replace('\n',' ') for someTxt in soup.find_all(class_='caselawcontent searchable-content')])))
    outList.append(line)

# %%

df = pd.DataFrame(outList,columns=['Title','Docket','Argued','Decided','Text'])

# %%
df

# %%


# %%
# with open('data/'+year+'.csv','w') as f:
#     for line in outList:
#         f.write(','.join(line)+'\n')
if len(outList)>0:
    df.to_pickle('data/'+year+'.pkl.zst',compression='zstd')
    print('File written to data/'+year+'.pkl.zst')
else:
    print('Skipped '+year)


