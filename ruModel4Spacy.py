# -*- coding: utf-8 -*-

'''
Get the files and more information on Slovnet: https://pypi.org/project/slovnet/
'''
import spacy
import pandas as pd
from razdel import sentenize, tokenize
from navec import Navec
from slovnet import Syntax
from slovnet import Morph
from pymystem3 import Mystem
class Tokens:
    def __init__(self, text, lemma_, pos_):
        self.text = text
        self.lemma_ = lemma_
        self.pos_=pos_

m = Mystem()
navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
morph.navec(navec)

text="Европейский союз добавил в санкционный список девять политических деятелей из самопровозглашенных республик Донбасса — Донецкой народной республики (ДНР) и Луганской народной республики (ЛНР) — в связи с прошедшими там выборами. Об этом говорится в документе, опубликованном в официальном журнале Евросоюза."
def nlp(text):
    chunks=[]
    lemmaSent=[]
    Doc=[]
    for sent in sentenize(text):
        tokens = [_.text for _ in tokenize(sent.text)]
        chunks.append(tokens)

    for chunk in chunks:
        filteredChunk=list(filter(lambda a: a != ' ', chunk))
        markup = next(morph.map([filteredChunk]))

        for token in markup.tokens:
            tokentext=token.text
            Doc.append(Tokens(tokentext,m.lemmatize(tokentext)[0],token.pos))
    return Doc

#for t in Doc:
#    print(t.text, t.lemma_, t.pos_)

doc = nlp(text)
features = [[t.text, t.lemma_, t.pos_] for t in doc]
doc_df = pd.DataFrame(features, columns=['text', 'lemma', 'pos'])

doc_df['text'] = doc_df['text'].str.lower()
doc_df['lemma'] = doc_df['lemma'].str.lower()
print(doc_df['lemma'])