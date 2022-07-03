



from bs4 import BeautifulSoup
import urllib.request
from string import ascii_letters

OUT_FILE = './data/dictionary.txt' 
BASE_URL = 'https://corpora.fi.muni.cz/habit/run.cgi/wordlist?corpname=amwac16&refs=&wlmaxitems=1000&wlsort=f&subcnorm=freq&corpname=amwac16&reload=&wlattr=word&usengrams=0&ngrams_n=2&ngrams_max_n=2&nest_ngrams=0&wlpat=&wlminfreq=1&wlmaxfreq=0&wlfile=&wlblacklist=&wlnums=frq&wltype=simple&wlpage'
NUM_PAGES = 949 ## Crawl 949 pages


english_alphabet = list(ascii_letters)



if __name__ == '__main__':
    with open(OUT_FILE, 'a', encoding='utf-8') as f:
        for i in range(NUM_PAGES):
            print('Crawling page - {}'.format(i + 1))
            url = '{}={}'.format(BASE_URL, i + 1)
            print(url)
            response = urllib.request.urlopen(url)
            response = response.read()
            soap = BeautifulSoup(response, 'html.parser')
            rows = soap.table.find_all('tr')[1:]
            
            # print(len(rows))
            if not len(rows):
                break
            
            for row in rows:
                word, frequency = row.find_all('td')
                """if len(list(filter(lambda x: x in word.text, english_alphabet))):
                    continue"""

                #f.write(word, frequency)
                #f.write('\n')
                x= int(frequency.text)
                f.write('{} {}'.format(word.text.strip(), x))  #, int(frequency.text)))
                f.write('\n')
            
    f.close