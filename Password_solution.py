import pandas as pd
import numpy as np

data = pd.read_csv('D:\\Avinash\DataScience_AI\Projects\PWD_Strength_Project\\data.csv',error_bad_lines=False)

data[data['password'].isnull()]

data['strength'].value_counts()

data.dropna(inplace=True)
passwords_tuple=np.array(data)
passwords_tuple

import random
random.shuffle(passwords_tuple) #shuffling randomly for robustness

y=[labels[1] for labels in passwords_tuple]

X=[labels[0] for labels in passwords_tuple]

import seaborn as sns
sns.set_style('whitegrid')
sns.countplot(x='strength',data=data,palette='RdBu_r')

data.shape

X

def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X=vectorizer.fit_transform(X)

X.shape

vectorizer.vocabulary_
X

data.iloc[0,0]

feature_names = vectorizer.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=X[0]
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False)

## Logistics Regression

from sklearn.linear_model import LogisticRegression
help(LogisticRegression())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

log_class=LogisticRegression(penalty='l2',multi_class='ovr')
log_class.fit(X_train,y_train)

print(log_class.score(X_test,y_test))

## Multinomial

clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(X_train, y_train) #training
print(clf.score(X_test, y_test))

X_predict=np.array(["%@123abcd"])
X_predict=vectorizer.transform(X_predict)
y_pred=log_class.predict(X_predict)
print(y_pred)
