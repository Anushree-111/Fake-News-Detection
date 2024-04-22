#import modules and load the csv files given 

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('/Users/ananyasharma/Downloads/train.csv')

#drop empty values
df=df.dropna(subset=['text'])

#covert all text to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())

#remove punctuation
import string
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
df['text'] = df['text'].apply(punctuation_removal)

# Get the labels, 1-unreliable, 0 -reliable
labels=df.label
labels.head()

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix
confusion_matrix(y_test,y_pred, labels=[1,0])

from sklearn.metrics import classification_report, accuracy_score
print(f"Classification Report : \n\n{classification_report(y_test, y_pred)}")


#Wordcloud for false news
import matplotlib.pyplot as plt
from wordcloud import WordCloud
fake_data = df[df["label"] == 1]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Wordcloud for true news
from wordcloud import WordCloud
fake_data = df[df["label"] == 0]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Confusion Matrix
from sklearn import metrics
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color='white' if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=[1, 0])

#Now, to label new data
dd = pd.read_csv('/Users/ananyasharma/Downloads/test.csv')
dd=dd.dropna(subset=['text'])
x_test1=dd['text']
tfidf_test1=tfidf_vectorizer.transform(x_test1)
y_pred1=pac.predict(tfidf_test1)
y_pred1

#Plot histogram
plt.hist(y_pred1,bins=30)
plt.ylabel('Count')
plt.xlabel('Label')
plt.show()


#To show the frequency of each value 
(unique, counts) = np.unique(y_pred1, return_counts=True)
frequencies = np.asarray((unique, counts)).T
frequencies

