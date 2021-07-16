# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:30:49 2021

@author: Avinash
"""
# WordCloud

# pip install WordCloud

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

with open("C:\\Users\\Avinash\\Documents\\iphone_11.txt","r") as ip:
 iphone = ip.read()

other_stopwords_to_remove = ["iphone","amazon","phone","will","apple"]

STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))

stopwords = set(STOPWORDS)

text = str(iphone)

wordcloud = WordCloud(width = 4000, height = 4000, 
                background_color ='white', 
                max_words=200,
                stopwords = stopwords, 
                min_font_size = 10).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
