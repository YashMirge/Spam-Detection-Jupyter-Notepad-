'''
Description: This program detects if an email is spam (1) or not (0)
Resources: (1) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
           (2) https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
           (3) https://www.kaggle.com/astandrik/simple-spam-filter-using-naive-bayes
           (4) https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier
           (5) https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/
           (6) https://towardsdatascience.com/spam-detection-with-logistic-regression-23e3709e522
           (7) https://github.com/SharmaNatasha/Machine-Learning-using-Python/blob/master/Classification%20project/Spam_Detection.ipynb
Data Source: https://www.kaggle.com/balakishan77/spam-or-ham-email-classification/data
'''

#Import libraries
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string

#Load the data
from google.colab import files # Use to load data on Google Colab
uploaded = files.upload() # Use to load data on Google Colab

df = pd.read_csv('emails.csv') #read the CSV file
df.head(5)

#Print the shape (Get the number of rows and cols)
df.shape

#Get the column names
df.columns

#Checking for duplicates and removing them
df.drop_duplicates(inplace = True)

#Show the new shape (number of rows & columns)
df.shape

#Show the number of missing (NAN, NaN, na) data for each column
df.isnull().sum()

#Need to download stopwords
nltk.download('stopwords')

#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words

#Show the Tokenization 
df['text'].head().apply(process_text)

'''
EXAMPLE OF THE PROCESS TO PREPARE THE DATA FOR TRAINING ON THE CLASSIFIER, 
THIS CELL/BLOCK ISNT NECESSARY TO RUN THE PROGRAM
'''
#Print the text (aka the email message)
message4 = 'hello world hello hello world play' #df['text'][3]
message5 = 'test test test test one hello'
print(message4)
print()

#Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
bow4 =  CountVectorizer(analyzer=process_text).fit_transform([[message4], [message5]])
print(bow4)
print()

#Transform a count matrix to a normalized tf or tf-idf representation
#Show the weight of TF-IDF
#Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. 
#from sklearn.feature_extraction.text import TfidfTransformer
#tfidf4 = TfidfTransformer().fit_transform(bow4)
#print(tfidf4)
#print()

#Print the shape (number of rows & columns) of bow4 
#print(bow4.shape)
#print()

#Show bow4 row at index 1
#bow4[1]

# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer

## Converts strings to integer counts
#vectorizer = CountVectorizer(analyzer=process_text)

## Learn a vocabulary dictionary of all tokens in the raw documents.
#bow_transformer = vectorizer.fit(df['text'])    

## Transform documents to document-term matrix.
#messages_bow = bow_transformer.transform(df['text'])

#Convert string to integer counts, learn the vocabulary dictionary and return term-document matrix
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

# Transform a count matrix to a normalized tf or tf-idf representation
# Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency. 
#from sklearn.feature_extraction.text import TfidfTransformer

## Learn the idf vector (global term weights)
#tfidf_transformer=TfidfTransformer().fit(messages_bow)

## Transform a count matrix to a tf or tf-idf representation
#messages_tfidf = tfidf_transformer.transform(messages_bow)

#Learn the idf vector to fit to data, then transform it.
#messages_tfidf = TfidfTransformer().fit_transform(messages_bow)

#Split the data into 80% training (X_train & y_train) and 20% testing (X_test & y_test) data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)

#Get the shape of messages_bow
messages_bow.shape

#Get the shape of messages_tfidf
#messages_tfidf.shape

#Create and train the Naive Bayes classifier
#The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))

#Print the actual values
print(y_train.values)

#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

#Print the predictions
print('Predicted value: ',classifier.predict(X_test))

#Print Actual Label
print('Actual value: ',y_test.values)

#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
