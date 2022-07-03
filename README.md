# amharic_chatbot_for_aau
web_scrapper folder consit one class crawler.py 
  it is used to scrape data from internet using beautiful soup
  
data folder consists 3 files
  1. intents.json => sample training data for neural network model
  2. dictionary.txt => dictionary of words for spelling check
  3. stopwords => list of stopwords
  4. documents.txt => sample documents for training Information retrieval

pysymspell folder have symspell.py class which uses damerau levenshtien algorithm to check words distance. It is language independent.
 
 preprocess.py
  preprocessing of training data
    
 retrieval.py
  preprocessing of documents for Information retrieval
