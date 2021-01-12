# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:22:06 2020

@author: Hauke Jansen
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:34:39 2020
@author: gsper
"""

import spacy
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import random
import string
import numpy as np
from lv_distance import levenshteinDistanceDP

import itertools

nlp = spacy.load('de_core_news_sm')

gate = pd.read_csv("top100_topics.csv", encoding='latin-1')
gate_original = pd.read_csv("top100_topics_original.csv", encoding='utf8')
nichts_list=['nichts','nix']
#doc_df = pd.DataFrame(np.array([['Blumenmuster', 'farbe', 'NOUN'], ['dvlksmvlsk', 'mama', 'NOUN'], ['cnakc', 'farbe', 'NOUN']]),
                #   columns=['text', 'lemma', 'pos'])


# query= "Recycling"
# query2= "Der Grundton ist hässlich"
# query3= "Fabe hässlich"
# query4= "Blumenmuster hässlich"
# query5= "machen machen machen"
# query6= "dvnuvkwb"
# query7= "..."


# pos_tag(query)


def pos_tag(query):
    query = ''.join(i for i in query if not i.isdigit())    
        
    if query.lower() in nichts_list:
        return('###skip_second_question###')
        
    else:
    
        doc = nlp(query)
        features = [[t.text,t.lemma_,t.pos_] for t in doc]
        doc_df = pd.DataFrame(features, columns= ['text', 'lemma', 'pos'])
        
        doc_df['text']=doc_df['text'].str.lower()
        doc_df['lemma']=doc_df['lemma'].str.lower()
    
        # GGF Optional: lowercase of topic
        gate['topic']=gate['topic'].str.lower()

        
        gate_original['topic']=gate_original['topic'].str.lower()
        gate_original['topic_lemma']=gate_original['topic_lemma'].str.lower()
        
        doc_nouns=doc_df[doc_df['pos']=='NOUN']
        
        intersection = pd.merge(doc_df,
                                gate,
                                left_on='text',
                                right_on='topic',
                                how="inner")
        
        #GGF Optional: Propn->NOUN
        intersection['pos']=intersection['pos'].replace({'PROPN':'NOUN'})
    
        
        if len(intersection[intersection.pos=="NOUN"])>0:
            response_set = ['Können Sie bitte Ihre Gedanken zu dem Aspekt'+' '+intersection.text[intersection.pos=="NOUN"].sample().item().capitalize()+' '+'näher erläutern?',
                        'Können Sie bitte noch genauer beschreiben, was Ihnen an dem Aspekt' +' '+intersection.text[intersection.pos=="NOUN"].sample().item().capitalize()+' '+'nicht gefällt?']
        
            return(random.choice(response_set))
            # print(random.choice(response_set))
        else: 
            doc_df_nouns=doc_df[doc_df['pos']=="NOUN"]
            
            if len(doc_df_nouns)>0:
            
                text_search=pd.DataFrame(list(itertools.product(doc_df_nouns['text'],gate['topic'].str.lower())),columns=['comment_text','gate_topic'])
                text_search["distance"]= np.vectorize(levenshteinDistanceDP)(text_search['comment_text'],text_search['gate_topic'])
        
                
                if len(text_search['distance'].values)>0:
                    relevant_topics=text_search[text_search['distance']==min(text_search['distance'])]  
                    relevant_topics=relevant_topics[relevant_topics['distance']<=1]     
            
                    if len(relevant_topics)>0:
                            response_set = ['Können Sie bitte Ihre Gedanken zu dem Aspekt'+' '+relevant_topics.gate_topic.sample().item().capitalize()+' '+'näher erläutern?',
                                'Können Sie bitte noch genauer beschreiben, was Ihnen an dem Aspekt' +' '+relevant_topics.gate_topic.sample().item().capitalize()+' '+'nicht gefällt?']
                            return(random.choice(response_set))
                            # print(random.choice(response_set))
                    else:                
                        text_search_original_temp=pd.DataFrame(list(itertools.product(doc_df_nouns['text'],gate_original['topic_lemma'])),columns=['comment_text','gate_subtopic'])
                        text_search_original_temp["distance"]= np.vectorize(levenshteinDistanceDP)(text_search_original_temp['comment_text'],text_search_original_temp['gate_subtopic'])
                        text_search_original=pd.merge(text_search_original_temp,
                                                      gate_original[['topic','topic_lemma']],
                                                      left_on='gate_subtopic',
                                                      right_on='topic_lemma',
                                                      how='left')
                        text_search_original=text_search_original[['comment_text','topic','gate_subtopic','distance']]
                        text_search_original=text_search_original.rename(columns={"topic":"gate_topic"})
                        
                        if len(text_search_original['distance'].values)>0:
                            
                            relevant_topics_original=text_search_original[text_search_original['distance']==min(text_search_original['distance'])]  
                            relevant_topics_original=relevant_topics_original[relevant_topics_original['distance']<=1]  
                                        
                            if len(relevant_topics_original)>0:
                                response_set = ['Können Sie bitte Ihre Gedanken zu dem Aspekt'+' "'+relevant_topics_original.gate_topic.sample().item().capitalize()+'" '+'näher erläutern?',
                                                'Können Sie bitte noch genauer beschreiben, was Ihnen an dem Aspekt' +' '+relevant_topics_original.gate_topic.sample().item().capitalize()+' '+'nicht gefällt?']
                                return(random.choice(response_set))
                                # print(random.choice(response_set))
                                
                            else:
                                response_set_fallback = 'Können Sie bitte noch etwas genauer beschreiben, was Ihnen an diesem Produkt nicht gefällt?'
                                return(response_set_fallback)
                                # print(response_set_fallback)
                                
                        else:
                            response_set_fallback = 'Können Sie bitte noch etwas genauer beschreiben, was Ihnen an diesem Produkt nicht gefällt?'
                            return(response_set_fallback)
                            # print(response_set_fallback)
                else:
                    response_set_fallback = 'Können Sie bitte noch etwas genauer beschreiben, was Ihnen an diesem Produkt nicht gefällt?'
                    return(response_set_fallback)
                    # print(response_set_fallback)
            else:
                response_set_fallback = 'Können Sie bitte noch etwas genauer beschreiben, was Ihnen an diesem Produkt nicht gefällt?'
                return(response_set_fallback)
                # print(response_set_fallback)


from time import process_time 

