from logging import root
import numpy as np
import pickle
import json
import nltk
import hm
import emoji
import string
import re
from pysymspell import SymSpell, EditDistance

spellcheck = SymSpell()
#print('Loading Dictionary File...')
spellcheck.load_dictionary('./data/dictionary.txt')


words = []
classes = []
documents = []
ignore_words = ['?', '።', '፣','፡', '.', '፤', ',', '፥', ',', '፩','፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲']
norm_1 = ['አ', 'ዐ', 'ኣ', 'ዓ', 'ሃ', 'ሓ', 'ኃ', 'ሀ', 'ሐ', 'ኀ', 'ሰ', 'ሠ', 'ጸ', 'ፀ']
norm_2 = ['አ', 'አ','አ','አ', 'ሀ', 'ሀ', 'ሀ', 'ሀ', 'ሀ', 'ሀ', 'ሰ', 'ሰ', 'ጸ', 'ጸ']
lemma_amh = []


stopwords = []
text_file = open("./data/stopwords.txt", "r", encoding = 'utf8')
for line in text_file:
    s_words = line.split(', ')
    stopwords.extend(s_words)
text_file.close()



with open("intents_1.json", encoding = 'utf8') as file:
    data = json.load(file)

for intent in data['intents']:
    for pattern in intent['patterns']:
#### add stopwords to nltk and remove them from patterns!!!

        #take each word and tokenize it
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        documents.append((word, intent['tag']))
#

        #adding classes to our list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(words)

#print(classes)

#print(documents)

### tokenize every word from words using hornmorpho




def remove_punc(words):
    new_words = []
    for word in words:
        clean_list = [char for char in word if char not in ignore_words]
        clean_str = ''.join(clean_list)
        new_words.append(clean_str)
    return new_words




def remove_nums(words):
    new_words = []
    for word in words:
        clean_list = [char for char in word if not re.search('\d', char)]
        clean_str = ''.join(clean_list)
        new_words.append(clean_str)
    return new_words
#print(remove_nums(words))


def remove_emoji(words):
    new_words = []
    for word in words:
        clean_list = [char for char in word if char not in emoji.EMOJI_DATA]
        clean_str = ''.join(clean_list)
        new_words.append(clean_str)
    return new_words
#print(remove_emoji(words))
    

def remove_stopwords(words):
    words = [w for w in words if w not in stopwords ]
    return words
#print(remove_stopwords(words))


def word_normalization(words):
    return words



def spell_corrector(words):
    for index, word in enumerate(words):
        sug_word = spellcheck.lookup_compound(word, 2)
        new_word = sug_word[0].term
        if new_word != word:
            words[index] = new_word 

    return words

print(spell_corrector(['አናቆት', 'ላይብረረ', 'መጽሃፍ', 'ነው', 'አለ', 'በየት']))




def root_word(words):
    lemm = [hm.anal_word('amh', w, root=False, citation=True, gram=False,
              roman=False, segment=False, guess=False, gloss=False,
              dont_guess=True, cache='', init_weight=None,
              lemma_only=True, ortho_only=False,
              normalize=True,
              rank=False, freq=False, nbest=1, um=False,
              phonetic=False, raw=True,
              pos=[], verbosity=0) for w in words if w not in ignore_words]

    new_words = []

    for lem in lemm:
        for le in lem:
            new_word = le['lemma'].split('|')[0]
            new_words.append(new_word)
            words = new_words

    return words

print(words)







#print(lemma_amh)
words = sorted(list(set(words)))

#print(words)
classes = sorted(list(set(classes)))

#{'POS': 'n', 'root': 'ሰላም|selam'}

#store them

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


