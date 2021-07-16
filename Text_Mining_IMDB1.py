# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:37:18 2021

@author: Avinash
"""

# WordCloud

# pip install WordCloud

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

with open("C:\\Users\\Avinash\\Documents\\imdb.txt","r") as im:
 imdb = im.read()

other_stopwords_to_remove = ["much","can","just","going","actually"]

STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))

stopwords = set(STOPWORDS)

text = str(imdb)

wordcloud = WordCloud(width = 4000, height = 4000, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()