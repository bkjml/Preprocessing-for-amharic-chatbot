import nltk
import re
import hm
import numpy as np
#import fasttext.util
#from nltk.tokenize.treebank import TreebankWordDetokenizer
import gensim

model_we = gensim.models.doc2vec.Doc2Vec



with open("./data/document.txt", "r", encoding = 'utf8') as f:
    document = f.readlines()

stopwords = []
text_file = open("./data/stopwords.txt", "r", encoding = 'utf8')
for line in text_file:
    s_words = line.split(', ')
    stopwords.extend(s_words)
text_file.close()



#print(document)

#special_characters = ['“','”', '፤', '"''"',':','!','#','$','%', '&','@','[',']',' ',']','_', '(', ')', '=', '|', '*', '-', '"', '>', '<', '`', '።', '፣', '፡']
#preprocessing
special_characters = ['\u200d','?', '....','..','...','','@','#', ',', '.', '"', ':', ')', '(', '-', '!', '|', ';', "'", '$', '&', '[', ']', '>', '%', '=', '*', '+', '\\', 
    '•', '~', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '`', '።', '፣',
    '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', 
    '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', '፤',
    'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '፡',
    '¹', '≤', '‡', '√', '!','🅰','🅱']

# a dictionary of common contractions and colloquial language
contraction_colloq_dict = {"እ/ር": "እግዚአብሔር", "ፕ/ር": "ፕሮፌሰር", "መ/ር": "መምህር","ዶ/ር": "ዶክተር", "ት/ቤት" : "ትምህርት ቤት"}


words = []
for line in document:
    word = nltk.word_tokenize(line)
    words.extend(word)
#print(words)

def remove_links(words):
    return [re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in words]

#print(remove_links(['https://towardsdatascience.com/complete-guide-to-building-a-chatbot-with-spacy-and-deep-learning-d18811465876', 'አበበ', 'ከበደ']))

words =  remove_links(words)


#data = words.progress_apply(replace_from_dict, dic = contraction_colloq_dict)
def replace_from_dict(words,dic):
    replaced_counter = 0
    for item in dic.items():
        for i, e in enumerate(words):
            if e == item[0]:
                replaced_counter+=1
                    # Inserting the expanded tokens in a way that preserves the order
                del words[i]
                for ix, token in enumerate(item[1].split()):
                    words.insert(i+ix,token)
#         print(f"Amount of words replaced: {replaced_counter}")
    return words

words = replace_from_dict(words, contraction_colloq_dict)



def remove_charac(words):
    new_words = []
    for word in words:
        clean_list = [char for char in word if char not in special_characters]
        clean_str = ''.join(clean_list)
        new_words.append(clean_str)
    return new_words
words = remove_charac(words)


def remove_nums(words):
    new_words = []
    for word in words:
        clean_list = [char for char in word if not re.search('\d', char)]
        clean_str = ''.join(clean_list)
        new_words.append(clean_str)
    for index,word in enumerate(new_words):
        if word == '/':
            new_words[index] = ''
    new_words = ' '.join(new_words).split()
    return new_words

words = remove_nums(words)


def remove_stopwords(words):
    words = [w for w in words if w not in stopwords ]
    return words

words = remove_stopwords(words)

#print(words)

def root_word(words):
    lemm = [hm.anal_word('amh', w, root=False, citation=True, gram=False,
              roman=False, segment=False, guess=False, gloss=False,
              dont_guess=True, cache='', init_weight=None,
              lemma_only=True, ortho_only=False,
              normalize=True,
              rank=False, freq=False, nbest=1, um=False,
              phonetic=False, raw=True,
              pos=[], verbosity=0) for w in words]

    new_words = []

    for lem in lemm:
        for le in lem:
            new_word = le['lemma'].split('|')[0]
            new_words.append(new_word)
            words = new_words

    return words

words = root_word(words)        
print(words)



#loading Amharic fasttext model 
"""ft = fasttext.load_model('cc.am.300.bin')
fasttext.util.reduce_model(ft, 300)

vec_list = []
for i in range(len(words)):
    for j in range(len(words[i])):
        vec = ft.get_sentence_vector(TreebankWordDetokenizer().detokenize(words[i][j]))
        vec_list.append(vec)

np_vec_list = np.array(vec_list, dtype=object)
np_vec_list  = np_vec_list.reshape(np_vec_list.shape[0], 1, np_vec_list.shape[1])
"""

#training
def train_doc2vec(words, max_epochs, vec_size, alpha):
    # Tagging each of the data with an ID, and I use the most memory efficient one of just using it's ID
    
    # Instantiating my model
    model = model_we(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1)

    model.build_vocab(words)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(words, total_examples = model.corpus_count, epochs=model.iter)
        # Decrease the learning rate
        model.alpha -= 0.0002
        # Fix the learning rate, no decay
        model.min_alpha = model.alpha

    # Saving model
    model.save("models/d2v.model")
    print("Model Saved")
    
# Training
print(train_doc2vec(words, max_epochs = 100, vec_size = 20, alpha = 0.025))


#clustering


