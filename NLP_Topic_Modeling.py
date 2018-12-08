#!/usr/bin/env python

from __future__ import print_function

import platform
import sys
import os

import warnings
warnings.simplefilter(action='ignore')

import logging
#print all logging.INFO details
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#Supress default INFO logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

from time import time
from itertools import groupby
from more_itertools import unique_everseen
from operator import itemgetter

import bs4
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen

import re
import regex
import string
import textwrap
from textwrap import fill

#import pycontractions
#from pycontractions import Contractions
import autocorrect
from autocorrect import spell

import numpy as np
import pandas as pd

import textacy
from textacy.preprocess import remove_punct

import spacy
from spacy.tokenizer import Tokenizer

import gensim
from gensim.matutils import hellinger
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec, CoherenceModel, LdaModel, HdpModel, LsiModel
from gensim.models.wrappers import LdaMallet #,LdaVowpalWabbit

import lda

import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, NMF, TruncatedSVD #,LatentDirichletAllocation
from sklearn.manifold import TSNE

#VISUALIZATION LIBRARIES (used for interactive development)

#import pyLDAvis.gensim
#pyLDAvis.enable_notebook()

#import IPython
#from IPython.display import display

#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#%matplotlib inline

#import plotly
#from plotly import tools
#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly.figure_factory as ff
#from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
#connects JS to notebook so plots work inline
#init_notebook_mode(connected=True)

#import cufflinks as cf
#allow offline use of cufflinks
#cf.go_offline()

#import bokeh
#from bokeh.io import push_notebook, show, output_notebook
#import bokeh.plotting as bp
#from bokeh.plotting import figure, save
#from bokeh.models import ColumnDataSource, LabelSet, HoverTool
#output_notebook()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_movie_reviews(soup_broth, n_reviews_per_movie=1):
    
    print('Retrieving all baseline URLs... \n')

    base_urls = [ ("https://www.imdb.com" + tag.attrs['href'], tag.text.replace('\n',' ') ) 
                            for tag in soup_broth.findAll('a', attrs={'href': re.compile("^/title/.*_tt")}) ]
    
    print('Retrieved all second-level URLs... \n')

    level_2_urls = []
    for url, title in base_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + tag.attrs['href']) 
                            for tag in soup.findAll('a', attrs={'href': re.compile("^/title/.*tt_urv")})]
        level_2_urls.append((update_url[0], title))
        
    print('Retrieved all third-level URLs... \n')

    level_3_urls = []
    for url, title in level_2_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + tag.attrs['href'])
                          for tag in soup.findAll('a', attrs={'href': re.compile("^/review/.*tt_urv")})[:(n_reviews_per_movie*2)]]
        update_url = list(unique_everseen(update_url))
        for i in update_url:
            level_3_urls.append((i, title))
        
    print('Retrieved all fourth-level URLs... \n')

    level_4_urls = []
    for url, title in level_3_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + soup.find('a', href=re.compile("^/review/.*rw_urv"))['href'])]
        level_4_urls.append((update_url[0], title))
        
    print('Retrieved all fifth-level URLs... \n')

    level_5_text = []
    for url, title in level_4_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        review_text = [(soup.find('div', {'class' : re.compile("^text")}).text)]
        level_5_text.append((review_text[0], title))
        
    print('All reviews retrieved! \n')

    return level_5_text
	
def clean_component(review, stop_words, tokenizer, puncts):
    """Text Cleaner: Tokenize, Remove Stopwords, Punctuation, Lemmatize, Spell Correct, Lowercase"""
    
    #rev_contract_exp = list(contract_model.expand_texts([review], precise=True))
    
    doc_tok = tokenizer(review)

    doc_lems = [tok.lemma_ for tok in doc_tok 
                    if (tok.text not in stop_words
                        and tok.text not in puncts
                        and tok.pos_ != "PUNCT" and tok.pos_ != "SYM")]
    
    lem_list = [re.search(r'\(?([0-9A-Za-z-]+)\)?', tok).group(1) 
                    if '-' in tok 
                    else spell(remove_punct(tok)) 
                        for tok in doc_lems]

    doc2vec_input = [t.lower() for tok in lem_list 
                         for t in tok.split() 
                             if t.lower() not in stop_words]
    
    return doc2vec_input
	
def get_tagged_documents(input_review_texts, stop_words, tokenizer, puncts, sentence_labeler):
    print('Creating Tagged Documents... \n')
    
    all_content = []
    j=0
    for rev, ttl in input_review_texts:
        print('Cleaning review #{}'.format(j+1))
        clean_rev = clean_component(rev, stop_words, tokenizer, puncts)
        all_content.append(sentence_labeler(clean_rev, [ttl]))
        j += 1

    print('Total Number of Movie Review Document Vectors: ', j)
    
    return all_content
	
def compute_coherence_values(model_type, dictionary, texts, max_topics, 
                                 corpus=False, topics=False, transformed_vectorizer=False, tfidf_norm=False,
                                 min_topics=2, stride=3, n_top_words=False, measure='u_mass',
                                 lsi_flag=False, nmf_flag=False, mallet_flag=False, mallet_path=False):

    coherence_values = []
    model_list = []
    if nmf_flag:
        feat_names = transformed_vectorizer.get_feature_names()
    
    for num_topics in range(min_topics, max_topics, stride):
        if lsi_flag:
            model=model_type(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        elif nmf_flag:
            model=model_type(n_components=num_topics, init='nndsvd', random_state=222)
        elif mallet_flag:
            model=model_type(mallet_path=mallet_path, corpus=corpus, id2word=dictionary, num_topics=num_topics) 
        else:
            model=model_type(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=111)   
            
        model_list.append(model)
        
        if nmf_flag:
            words_list = []
            
            for i in range(num_topics):
                model.fit(tfidf_norm)
                
                #for each topic, obtain the largest values, and add the words they map to into the dictionary
                words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
                words = [feat_names[key] for key in words_ids]
                words_list.append(words)
                
            coherencemodel = CoherenceModel(topics=words_list, texts=texts, dictionary=dictionary, coherence=measure)
            
        else:
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=measure)
        
        coherence_values.append(coherencemodel.get_coherence())
        print('Coherence of model with {} topics has been computed'.format(num_topics))
        
    print('All coherence values have been computed for the {} using the {} measure'.format(model_type, measure.upper()))
    print('\n')

    return model_list, coherence_values
	
def show_best_num_topics(model_type, umass_coherence_vals, cv_coherence_vals, max_topics, min_topics=2, stride=3):
    
	#Visualizations disabled for command line execution!
	
    #min_topics=min_topics
    #max_topics=max_topics
    #stride=stride
    
    x = range(min_topics, max_topics, stride)
    
    max_y1 = max(umass_coherence_vals)
    max_x1 = x[umass_coherence_vals.index(max_y1)]
    
    max_y2 = max(cv_coherence_vals)
    max_x2 = x[cv_coherence_vals.index(max_y2)] 
    
    #fig = plt.figure(constrained_layout=True, figsize=(14,5))

    #gs = GridSpec(1, 2, figure=fig)
    #ax1 = fig.add_subplot(gs[0, 0])
    #ax2 = fig.add_subplot(gs[0, -1])

    #ax1.plot(x, umass_coherence_vals, label='Coherence Values')
    #ax1.set_xlabel('Num Topics')
    #ax1.set_ylabel('Coherence score (U_MASS)')
    #ax1.legend(loc='best')
    #ax1.text(max_x1, max_y1, str((max_x1, max_y1)))

    #ax2.plot(x, cv_coherence_vals, label='Coherence Values')
    #ax2.set_xlabel('Num Topics')
    #ax2.set_ylabel('Coherence score (C_V)')
    #ax2.legend(loc='best')
    #ax2.text(max_x2, max_y2, str((max_x2, max_y2)))

    #fig.suptitle('Coherence Scores ({})'.format(model_type))

    #plt.show()

    print('The most coherent number of {} topics using the U_MASS measure is: {}'.format(model_type, max_x1))
    print('The most coherent number of {} topics using the C_V measure is: {}'.format(model_type, max_x2))
    print('\n')
    
    tup_list = [(max_x1, max_y1), (max_x2, max_y2)]
    best_num_topics = max(tup_list, key=itemgetter(1))[0]
    
    return best_num_topics
	
def get_topics(model, num_topics):
    
    word_dict = {}
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20)
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]
        
    return pd.DataFrame(word_dict)
	
def get_nmf_topics(model, num_topics, vectorizer, n_top_words):
    
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names()
    
    word_dict = {}
    for i in range(num_topics):
        
        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words
    
    return pd.DataFrame(word_dict)
	
def topics_2_bow(topic, model, lsi_flag=False, mallet_flag=False):

    topic = topic.split('+')
    topic_bow = []
    
    lsi_mallet_array = np.array([])
    lsi_mallet_dict = {}
    lsi_mallet_dict_scaled = {}
    
    for word in topic:
        #split probability and word
        try:
            prob, word = word.split('*')
        except:
            continue

        #replace unwanted characters
        rep = {' ': '', '"': ''}
        replace = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(replace.keys()))
        word = pattern.sub(lambda m: replace[re.escape(m.group(0))], word)

        #convert to word_type
        try:
            word = model.id2word.doc2bow([word])[0][0]
        except:
            continue
            
        if lsi_flag or mallet_flag:
            lsi_mallet_array = np.append(lsi_mallet_array, float(prob))
            lsi_mallet_dict.update({word:prob})
              
        else:
            topic_bow.append((word, float(prob)))
                        
    if lsi_flag or mallet_flag:
        scaler = MinMaxScaler(feature_range=(0,1), copy=True)
        lsi_mallet_scaled = scaler.fit_transform(lsi_mallet_array.reshape(-1,1))
        
        lsi_mallet_scaled = (lsi_mallet_scaled - lsi_mallet_scaled.min()) / (lsi_mallet_scaled - lsi_mallet_scaled.min()).sum()
        
        for k,v in zip(lsi_mallet_dict.keys(), lsi_mallet_scaled):
            lsi_mallet_dict_scaled.update({k:v[0]})        

        for k,v in lsi_mallet_dict_scaled.items():
            topic_bow.append((k,v))
        
    return topic_bow
	
def get_most_similar_topics(model, topics_df=False, num_topics=False, mallet_flag=False, hdp_flag=False, lsi_flag=False, nmf_flag=False, columns=False):
    
    if not nmf_flag:
    
        mod_topics = tuple(topics_df.columns)
        mod_top_dict = {}

        if hdp_flag:
            for k,v in zip(mod_topics, model.show_topics(num_words=len(model.id2word))):
                mod_top_dict.update({k:v})
        else:    
            for k,v in zip(mod_topics, model.show_topics(num_words=len(model.id2word), num_topics=num_topics)):
                mod_top_dict.update({k:v})

        for k,v in mod_top_dict.items():
            mod_top_dict[k] = topics_2_bow(v[1], model, lsi_flag, mallet_flag)
    
    else:
        mod_topics = tuple(columns)
        mod_top_dict = {}
        
        nmf_top_list = []
        for k,v in zip(mod_topics, model.components_):
            nmf_top_list.append(tuple((k, v)))
        
        scaler = MinMaxScaler(feature_range=(0,1), copy=True)
        
        for k,v in nmf_top_list:
            v_scaled = scaler.fit_transform(v.reshape(-1,1))
            v = (v_scaled - v_scaled.min()) / (v_scaled - v_scaled.min()).sum()
            mod_top_dict.update({k:v})

    hellinger_dists = [(hellinger(mod_top_dict[x], mod_top_dict[y]), x, y)
                          for i,x in enumerate(mod_top_dict.keys())
                          for j,y in enumerate(mod_top_dict.keys())
                          if i != j]       

    unique_hellinger = [tuple(x) for x in set(map(frozenset, hellinger_dists)) if len(tuple(x)) == 3]
    
    resorted_hellinger = []
    for i in range(len(unique_hellinger)):
        resorted_hellinger.append(sorted(tuple(str(e) for e in unique_hellinger[i])))
        resorted_hellinger[i][0] = float(resorted_hellinger[i][0])
        resorted_hellinger[i] = tuple(resorted_hellinger[i])

    resorted_hellinger = sorted(resorted_hellinger, reverse=True)

    return resorted_hellinger
	
def get_dominant_topics(model, corpus, texts, lsi_flag=False):

    dom_topics_df = pd.DataFrame()

    #for all topic/topic-probability pairings
    for i, row in enumerate(model[corpus]):
        #return the pairings, sorting first the one with the highest topic-probability in the mixture
        if lsi_flag:
            row = sorted(row, key=lambda x: abs(x[1]), reverse=True)
        else:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)     
        #for every topic/topic-probability pairing in the sorted tuple list
        for j, (topic_num, topic_prob) in enumerate(row):
            #take the pairing with the highest topic-probability
            if j == 0:
                #return the tuple list of top 'n' word-probabilities associated with that topic
                wp = model.show_topic(topic_num, topn=20)
                #create a list of those top 'n' words
                topic_keywords = ", ".join([word for word, prob in wp])
                #append to the empty dataframe a series containing:
                #    the dominant topic allocation for that document,
                #    the topic-probability it contributes to that document,
                #    and the top 'n' words associated with that topic
                dom_topics_df = dom_topics_df.append(pd.Series([int(topic_num), round(topic_prob,4), topic_keywords]), ignore_index=True)
            else:
                #ignore other topics in the mixture
                break
                
    #name the columns of the constructed dataframe
    dom_topics_df.columns = ['Dominant_Topic', 'Probability_Contribution', 'Topic_Keywords']

    #append to the original text as another column in the dataframe
    contents = pd.Series(texts)
    dom_topics_df = pd.concat([dom_topics_df, contents], axis=1)
    
    dom_topics_final = dom_topics_df.reset_index()
    dom_topics_final.columns = ['Document_Number', 'Dominant_Topic', 'Probability_Contribution', 'Topic_Keywords', 'Original_Text']
    dom_topics_final.set_index('Document_Number', inplace=True)
    
    return dom_topics_df, dom_topics_final
	
def get_most_representative_docs(dom_topics_final_df, n_topics=20, lsi_flag=False):
    
    representative_docs = pd.DataFrame()
    
    dom_topics_grpd = dom_topics_final_df.groupby('Dominant_Topic')

    #for every dominant topic, find the document to which it contributed the greatest (or largest magnitude [LSI]) probability contribution
    if not lsi_flag:    
        for i, grp in dom_topics_grpd:
            representative_docs = pd.concat([representative_docs, 
                                                grp.sort_values(['Probability_Contribution'], ascending=True).head(1)], 
                                                axis=0)
    else:
        for i, grp in dom_topics_grpd:
            representative_docs = pd.concat([representative_docs,
                                                #sort by descending absolute value
                                                grp.reindex(grp.Probability_Contribution.abs().sort_values(inplace=False, ascending=False).index).head(1)],
                                                axis=0)

    representative_docs.reset_index(drop=True, inplace=True)

    representative_docs.columns = ['Topic_Number', 'Probability_Contribution', 'Topic_Keywords', 'Most_Representative_Document']
    representative_docs['Topic_Number'] = representative_docs['Topic_Number'].astype(int)

    representative_docs.set_index('Topic_Number', inplace=True)

    rep_docs_first_n = representative_docs.head(n_topics)
    
    return rep_docs_first_n
	
def get_topic_distribution(dom_topics_final_df, rep_doc_df=False, n_topics=20, hdp_flag=False):
    
    if hdp_flag:
        
        rep_doc_df = get_most_representative_docs(dom_topics_final_df, n_topics=dom_topics_final_df['Dominant_Topic'].nunique())
    
    rep_doc_df.reset_index(inplace=True)
    
    #topic number and keywords
    topic_num_keywords = rep_doc_df[['Topic_Number', 'Topic_Keywords']]
    
    #number of documents allocated to each topic
    topic_counts = dom_topics_final_df['Dominant_Topic'].value_counts()
    
    #topic allocation percentage of total corpus
    topic_percent = round(topic_counts/topic_counts.sum(), 4)
    
    #concat number of allocated docs and percentage
    topic_dist = pd.concat([topic_num_keywords, topic_counts, topic_percent], axis=1)
    
    topic_dist.columns = ['Topic_Number', 'Topic_Keywords', 'Documents_per_Topic', 'Percent_of_Total_Corpus']
    topic_dist.dropna(axis=0, how='any', inplace=True)
    
    topic_dist['Topic_Number'] = topic_dist['Topic_Number'].astype(int)
    topic_dist.set_index('Topic_Number', inplace=True)
    
    topic_dist = topic_dist[['Documents_per_Topic', 'Percent_of_Total_Corpus', 'Topic_Keywords']]

    topic_dist_first_n = topic_dist.head(n_topics)
    
    return topic_dist_first_n

def svd_2D_scatter(svd_2D_transformed, vectorizer, color_scale_dimension=0):
    
    trace = go.Scattergl(
    x = svd_2D_transformed[:,0],
    y = svd_2D_transformed[:,1],
    mode = 'markers',
    marker = dict(
        color = svd_2D_transformed[:,color_scale_dimension],
        colorscale='Viridis',
        colorbar=dict(title='TF-IDF Variation: SVD Component #{}'.format(color_scale_dimension+1)),
        line = dict(width = 1)
    ),
    text = vectorizer.get_feature_names(),
    )
    
    data = [trace]
    iplot(data, filename='scatter-mode')
    
    return None
	
def svd_3D_scatter(svd_3D_transformed, vectorizer, color_scale_dimension=0):
    
    svd_3D_dim_range = np.stack((np.amax(svd_3D_transformed, axis=0), np.amin(svd_3D_transformed, axis=0)))
    svd_3D_max_dim = max(np.max(svd_3D_dim_range, axis=0))
    svd_3D_max_dim += svd_3D_max_dim*(0.1)
    
    fig = tools.make_subplots(rows=1, cols=1,
                          print_grid=False,
                          specs=[[{'is_3d': True}]])
    scene = dict(
        camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.5, y=0.1, z=0.1)
        ),

        xaxis=dict(
            range=[-svd_3D_max_dim, svd_3D_max_dim],
            title='SVD_C1',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        ),
        yaxis=dict(
            range=[-svd_3D_max_dim, svd_3D_max_dim],
            title='SVD_C2',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        ),
        zaxis=dict(
            range=[-svd_3D_max_dim, svd_3D_max_dim],
            title='SVD_C3',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        )
    )

    centers = [[1, 1], [-1, -1], [1, -1]]
    X = svd_3D_transformed

    trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                             showlegend=False,
                             mode='markers',

                             marker=dict(
                                color=X[:,color_scale_dimension],
                                colorscale='Viridis',
                                colorbar=dict(title='TF-IDF Variation: SVD Component #{}'.format(color_scale_dimension+1)),
                                line=dict(color='black', width=1)),

                            text = vectorizer.get_feature_names())

    fig.append_trace(trace, 1, 1)

    fig['layout'].update(height=1000, width=1200,
                         margin=dict(l=10,r=10))

    fig['layout']['scene'].update(scene)

    iplot(fig)
    
    return None
	
def plot_tsne(doc_list, fitted_lda, fitted_count_vectorizer, transformed_lda, transformed_tsne, color_map, n_top_words=10):
    
    n_top_words = n_top_words
    color_map = color_map
    
    #retrieve component key words
    _lda_keys = []
    for i in range(transformed_lda.shape[0]):
        _lda_keys +=  transformed_lda[i].argmax(),
        
    topic_summaries = []
    #matrix of shape n_topics x len(vocabulary)
    topic_words = fitted_lda.topic_word_
    #all vocab words (strings)
    vocab = fitted_count_vectorizer.get_feature_names()
    for i, topic_dist in enumerate(topic_words):
        #np.argsort returns indices that would sort an array
        #iterates over topic component vectors, sorts array in asc order, appends key words from end of array
        topic_word = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topic_summaries.append(' '.join(topic_word))
    
    num_example = len(transformed_lda)

    plot_dict = {
            'x': transformed_tsne[:, 0],
            'y': transformed_tsne[:, 1],
            'colors': color_map[_lda_keys][:num_example],
            'content': doc_list[:num_example],
            'topic_key': _lda_keys[:num_example]
            }

    #create dataframe from dictionary
    plot_df = pd.DataFrame.from_dict(plot_dict)

    source = bp.ColumnDataSource(data=plot_df)
    title = 'LDA T-SNE Visualization'

    plot_lda = bp.figure(plot_width=1400, plot_height=1100,
                         title=title,
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    plot_lda.scatter('x','y', color='colors', source=source)

    topic_coord = np.empty((transformed_lda.shape[1], 2)) * np.nan
    for topic_num in _lda_keys:
        if not np.isnan(topic_coord).any():
            break
        topic_coord[topic_num] = transformed_tsne[_lda_keys.index(topic_num)]

    #plot key words
    for i in range(transformed_lda.shape[1]):
        plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

    #hover tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"content": "@content - topic: @topic_key"}

    # save the plot
    #save(plot_lda, '{}.html'.format(title))

    #Cf. JSON Serialization issue: 
    #    https://github.com/bokeh/bokeh/issues/5439
    #    https://github.com/bokeh/bokeh/issues/6222
    #   https://github.com/bokeh/bokeh/issues/7523
    try:
        show(plot_lda, notebook_handle=True)
    except Exception as e:
        print('Note!: {}'.format(e.__doc__))
        print(e)

    return None
	
def color_green(val):
    color = 'green' if isinstance(val, float) else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if isinstance(val, float) else 400
    return 'font-weight: {wt}'.format(wt=weight)
	
def pretty_print(input_text):

	pieces = [str(word) for word in input_text]
	output = ' '.join(pieces)
	#write_up = fill(output)
	print('Hellinger Distances between topics are: \n')
	print('_'*70)
	print('\n')
	print(output.translate({ord(')'):')\n'}))
	print('_'*70)
	#print(write_up)

	return None
	
def main():
	
	###########################################Loading Language, Punctuation, and Tagged Document Models######################################
	print('Loading Language, Punctuation, and Tagged Document Models... \n')
	
	nlp = spacy.load('en')

	#cont = Contractions('../GoogleNews-vectors-negative300.bin.gz')
	#cont.load_models()

	punctuations = string.punctuation

	LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
	
	##################################################Loading Tokenizer and Updating Stop Words#################################################
	
	print('Loading Tokenizer and Updating Stop Words... \n')
	
	tokenizer = Tokenizer(nlp.vocab)

	#stopwords = spacy.lang.en.STOP_WORDS
	#spacy.lang.en.STOP_WORDS.add("e.g.")
	#nlp.vocab['the'].is_stop
	nlp.Defaults.stop_words |= {"(a)", "(b)", "(c)", "etc", "etc.", "etc.)", "w/e", "(e.g.", "no?", "s", 
							   "film", "movie","0","1","2","3","4","5","6","7","8","9","10","e","f","k","n","q",
								"de","oh","ones","miike","http","imdb", "horror", "like", "good", "great", "little", 
								"come", "way", "know", "michael", "lot", "thing", "films", "later", "actually", "find", 
								"big", "long", "away", "filmthe", "www", "com", "x", "aja", "agritos", "lon", "therebravo", 
								"gou", "b", "particularly", "probably", "sure", "greenskeeper", "try", 
								"half", "intothe", "especially", "exactly", "20", "ukr", "thatll", "darn", "certainly", "simply", }

	stopwords = list(nlp.Defaults.stop_words)
	
	##############################################Retrieving and Cleaning Reviews of the Best Horror Movies#######################################
	
	print('Retrieving and Cleaning Reviews of the Best Horror Movies... \n')

	#open the base URL webpage
	html_page = urlopen("https://www.imdb.com/list/ls059633855/")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')

	review_text_first_5 = get_movie_reviews(soup, n_reviews_per_movie=5)
	
	all_good_movie_reviews_500 = get_tagged_documents(review_text_first_5, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	##############################################Retrieving and Cleaning Reviews of the Worst Horror Movies######################################
	
	print('\n')
	print('Retrieving and Cleaning Reviews of the Worst Horror Movies... \n')

	#open the base URL webpage
	html_page_bad = urlopen("https://www.imdb.com/list/ls061324742/")

	#instantiate beautiful soup object of the html page
	soup_bad = BeautifulSoup(html_page_bad, 'lxml')

	review_text_first_5_bad = get_movie_reviews(soup_bad, n_reviews_per_movie=5)
	
	all_bad_movie_reviews_500 = get_tagged_documents(review_text_first_5_bad, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	#######################################Creating Corpus of Combined Reviews of Best and Worst Horror Movies#####################################
	
	print('\n')
	print('Creating Corpus of Combined Reviews of Best and Worst Horror Movies... \n')

	all_reviews = all_good_movie_reviews_500 + all_bad_movie_reviews_500
	
	all_reviews_text = []
	all_reviews_ttl = []
	for i in range(len(all_reviews)):
		all_reviews_text.append(all_reviews[i][0])
		all_reviews_ttl.append(all_reviews[i][1][0])
		
	reg_ex = regex.compile('[^a-zA-Z]')
	
	dictionary = corpora.Dictionary(all_reviews_text)
	
	#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
	#filtered_dict = dictionary.filter_extremes(no_below=1, no_above=0.8)

	#convert the dictionary to a bag of words corpus for reference
	corpus = [dictionary.doc2bow(text) for text in all_reviews_text]
	
	all_reviews_joined = [' '.join(w) for w in all_reviews_text]
	
	##################################################LDA (Latent Dirichlet Allocation)#########################################################
	
	print('_'*70)
	print('Performing LDA (Latent Dirichlet Allocation)... \n')

	## Cf. http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
	## Cf. https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf  (LDAvis)
	## Cf. http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf (Topic Coherence Measures)
	
	mod_list_lda_umass, coh_list_lda_umass = compute_coherence_values(LdaModel, dictionary, all_reviews_text, max_topics=20, 
                                                                     corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
                                                                     min_topics=2, stride=3, n_top_words=False, measure='u_mass')

	mod_list_lda_cv, coh_list_lda_cv = compute_coherence_values(LdaModel, dictionary, all_reviews_text, max_topics=20, 
																		 corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
																		 min_topics=2, stride=3, n_top_words=False, measure='c_v')
	
	best_num_lda_topics = show_best_num_topics('LDA', coh_list_lda_umass, coh_list_lda_cv, max_topics=20, min_topics=2, stride=3)
	print('The most coherent number of LDA topics to use is: {}'.format(best_num_lda_topics))
	
	print('\n')
	print('_'*70)
	print('Performing LDA with the most coherent number of topics... \n')
	
	lda_mod = models.LdaModel(corpus, num_topics=best_num_lda_topics,  
                                id2word=dictionary, 
                                update_every=0, #batch training
                                chunksize=1000, 
                                passes=500,
                                random_state=222)
	
	#Perplexity measure (lower is better)
	print('\n')
	print('The Perplexity measure of the final LDA model is: {}'.format(lda_mod.log_perplexity(corpus)))
	print('_'*70)
	print('\n')
	
	lda_df = get_topics(lda_mod, best_num_lda_topics)
	try:
		print('The top 20 words of each topic are: \n')
		print(lda_df)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	lda_hellinger_topic_distances = get_most_similar_topics(lda_mod, lda_df, num_topics=best_num_lda_topics)
	print('\n')
	pretty_print(lda_hellinger_topic_distances)
	
	lda_mod_dom_topics_df, lda_mod_dom_topics_final = get_dominant_topics(lda_mod, corpus, all_reviews_joined)
	
	#lda_mod_dom_topics_final.style.applymap(color_green).applymap(make_bold)
	lda_mod_dom_topics_final_first30 = lda_mod_dom_topics_final.head(30)
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(lda_mod_dom_topics_final_first30)
	#lda_mod_dom_topics_final_first30.style.applymap(color_green).applymap(make_bold)
	
	lda_most_representative_docs = get_most_representative_docs(lda_mod_dom_topics_final, n_topics=20)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(lda_most_representative_docs)
	#lda_most_representative_docs.style.applymap(color_green).applymap(make_bold)
	
	lda_topic_dist = get_topic_distribution(lda_mod_dom_topics_final, lda_most_representative_docs, n_topics=20)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(lda_topic_dist)
	#lda_topic_dist.style.applymap(color_green).applymap(make_bold)
	
	#prepared = pyLDAvis.gensim.prepare(lda_mod, corpus, dictionary)
	#display(pyLDAvis.display(prepared))
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LDA: Second Pass of LDA (Latent Dirichlet Allocation)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	print('~'*70)
	print('\n')
	print('Performing Second Pass of LDA (Latent Dirichlet Allocation) using 14 Topics... \n')
	
	lda_mod_2 = models.LdaModel(corpus, num_topics=14,  
                                id2word=dictionary, 
                                update_every=0, #batch training
                                chunksize=1000, 
                                passes=500,
                                random_state=222)
								
	#Perplexity measure (lower is better)
	print('\n')
	print('The Perplexity measure of the final LDA model is: {}'.format(lda_mod_2.log_perplexity(corpus)))
	print('_'*70)
	print('\n')
								
	lda_mod_df_2 = get_topics(lda_mod_2, 14)
	try:
		print('The top 20 words of each topic are: \n')
		print(lda_mod_df_2)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	lda_mod_hellinger_topic_distances_2 = get_most_similar_topics(lda_mod_2, lda_mod_df_2, num_topics=14, mallet_flag=True)
	print('\n')
	pretty_print(lda_mod_hellinger_topic_distances_2)
	
	lda_mod_dom_topics_df_2, lda_mod_dom_topics_final_2 = get_dominant_topics(lda_mod_2, corpus, all_reviews_joined)
	
	#lda_mod_dom_topics_final_2.style.applymap(color_green).applymap(make_bold)
	lda_mod_dom_topics_final_first30_2 = lda_mod_dom_topics_final_2.head(30)
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(lda_mod_dom_topics_final_first30_2)
	#lda_mod_dom_topics_final_first30_2.style.applymap(color_green).applymap(make_bold)
	
	lda_mod_most_representative_docs_2 = get_most_representative_docs(lda_mod_dom_topics_final_2, n_topics=20)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(lda_mod_most_representative_docs_2)
	#lda_mod_most_representative_docs_2.style.applymap(color_green).applymap(make_bold)
	
	lda_mod_topic_dist_2 = get_topic_distribution(lda_mod_dom_topics_final_2, lda_mod_most_representative_docs_2, n_topics=20)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(lda_mod_topic_dist_2)
	#lda_mod_topic_dist_2.style.applymap(color_green).applymap(make_bold)
	
	##################################################################LDA Mallet###################################################################
	
	print('_'*70)
	print('\n')
	print('Performing Mallet LDA (Latent Dirichlet Allocation)... \n')

	## Cf. http://mallet.cs.umass.edu/
	#File dl: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
	
	os.environ['MALLET_HOME'] = '.\\mallet-2.0.8\\'
	mallet_path = '.\\mallet-2.0.8\\bin\\mallet'
	
	mod_list_ldamallet_umass, coh_list_ldamallet_umass = compute_coherence_values(LdaMallet, dictionary, all_reviews_text, max_topics=20, 
                                                                     corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
                                                                     min_topics=2, stride=3, n_top_words=False, measure='u_mass', 
                                                                     mallet_flag=True, mallet_path=mallet_path)

	mod_list_ldamallet_cv, coh_list_ldamallet_cv = compute_coherence_values(LdaMallet, dictionary, all_reviews_text, max_topics=20, 
																		 corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
																		 min_topics=2, stride=3, n_top_words=False, measure='c_v', 
																		 mallet_flag=True, mallet_path=mallet_path)
																		 
	best_num_ldamallet_topics = show_best_num_topics('LDA_Mallet', coh_list_ldamallet_umass, coh_list_ldamallet_cv, max_topics=20, min_topics=2, stride=3)
	print('The most coherent number of LDA_Mallet topics to use is: {}'.format(best_num_ldamallet_topics))
		
	print('\n')
	print('_'*70)
	print('Performing Mallet LDA with the most coherent number of topics... \n')
	
	lda_mallet = LdaMallet(mallet_path, corpus=corpus, num_topics=best_num_ldamallet_topics, id2word=dictionary)
	
	lda_mallet_df = get_topics(lda_mallet, best_num_ldamallet_topics)
	try:
		print('\n')
		print('_'*70)
		print('The top 20 words of each topic are: \n')
		print(lda_mallet_df)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	ldamallet_hellinger_topic_distances = get_most_similar_topics(lda_mallet, lda_mallet_df, num_topics=best_num_ldamallet_topics, mallet_flag=True)
	print('\n')
	pretty_print(ldamallet_hellinger_topic_distances)
	
	lda_mallet_dom_topics_df, lda_mallet_dom_topics_final = get_dominant_topics(lda_mallet, corpus, all_reviews_joined)
	
	#lda_mallet_dom_topics_final.style.applymap(color_green).applymap(make_bold)
	lda_mallet_dom_topics_final_first30 = lda_mallet_dom_topics_final.head(30)
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(lda_mallet_dom_topics_final_first30)
	#lda_mallet_dom_topics_final_first30.style.applymap(color_green).applymap(make_bold)
	
	lda_mallet_most_representative_docs = get_most_representative_docs(lda_mallet_dom_topics_final, n_topics=20)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(lda_mallet_most_representative_docs)
	#lda_mallet_most_representative_docs.style.applymap(color_green).applymap(make_bold)
	
	lda_mallet_topic_dist = get_topic_distribution(lda_mallet_dom_topics_final, lda_mallet_most_representative_docs, n_topics=20)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(lda_mallet_topic_dist)
	#lda_mallet_topic_dist.style.applymap(color_green).applymap(make_bold)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LDA Mallet: Second Pass with second highest C_V Coherence Optimized number of topics~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	print('~'*70)
	print('\n')
	print('Performing Second Pass of Mallet LDA (Latent Dirichlet Allocation) using 11 Topics... \n')
	
	lda_mallet_2 = LdaMallet(mallet_path, corpus=corpus, num_topics=11, id2word=dictionary)
	
	lda_mallet_df_2 = get_topics(lda_mallet_2, 11)
	try:
		print('\n')
		print('_'*70)
		print('The top 20 words of each topic are: \n')
		print(lda_mallet_df_2)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	ldamallet_hellinger_topic_distances_2 = get_most_similar_topics(lda_mallet_2, lda_mallet_df_2, num_topics=11, mallet_flag=True)
	print('\n')
	pretty_print(ldamallet_hellinger_topic_distances_2)
	
	lda_mallet_dom_topics_df_2, lda_mallet_dom_topics_final_2 = get_dominant_topics(lda_mallet_2, corpus, all_reviews_joined)
	
	#lda_mallet_dom_topics_final_2.style.applymap(color_green).applymap(make_bold)
	lda_mallet_dom_topics_final_first30_2 = lda_mallet_dom_topics_final_2.head(30)
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(lda_mallet_dom_topics_final_first30_2)
	#lda_mallet_dom_topics_final_first30_2.style.applymap(color_green).applymap(make_bold)
	
	lda_mallet_most_representative_docs_2 = get_most_representative_docs(lda_mallet_dom_topics_final_2, n_topics=20)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(lda_mallet_most_representative_docs_2)
	#lda_mallet_most_representative_docs_2.style.applymap(color_green).applymap(make_bold)
	
	lda_mallet_topic_dist_2 = get_topic_distribution(lda_mallet_dom_topics_final_2, lda_mallet_most_representative_docs_2, n_topics=20)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(lda_mallet_topic_dist_2)
	#lda_mallet_topic_dist_2.style.applymap(color_green).applymap(make_bold)
	
	###################################################HDP (Hierarchical Dirichlet Process)######################################################
	
	print('_'*70)
	print('\n')
	print('Performing HDP (Hierarchical Dirichlet Process)... \n')

	## Cf. http://proceedings.mlr.press/v15/wang11a/wang11a.pdf
	
	hdp = models.HdpModel(corpus, id2word=dictionary, chunksize=1000, random_state=222)
	
	hdp_df = get_topics(hdp, 20)
	try:
		print('The top 20 words of each topic are: \n')
		print(hdp_df)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	hdp_hellinger_topic_distances = get_most_similar_topics(hdp, hdp_df, hdp_flag=True)
	print('\n')
	pretty_print(hdp_hellinger_topic_distances)
	
	hdp_dom_topics_df, hdp_dom_topics_final = get_dominant_topics(hdp, corpus, all_reviews_joined)
	
	#hdp_dom_topics_final.style.applymap(color_green).applymap(make_bold)
	hdp_dom_topics_final_first30 = hdp_dom_topics_final.head(30)
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(hdp_dom_topics_final_first30)
	#hdp_dom_topics_final_first30.style.applymap(color_green).applymap(make_bold)
	
	hdp_most_representative_docs = get_most_representative_docs(hdp_dom_topics_final, n_topics=20)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(hdp_most_representative_docs)
	#hdp_most_representative_docs.style.applymap(color_green).applymap(make_bold)
	
	hdp_topic_dist = get_topic_distribution(hdp_dom_topics_final, n_topics=20, hdp_flag=True)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(hdp_topic_dist)
	#hdp_topic_dist.style.applymap(color_green).applymap(make_bold)
	
	hdp_topics = []
	for topic_id, topic in hdp.show_topics(num_topics=20, formatted=False):
		topic = [word for word, _ in topic]
		hdp_topics.append(topic)
		
	hdp_cm_umass = CoherenceModel(topics=hdp_topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
	print('\n')
	print('_'*70)
	print('The U_MASS coherence measure of the HDP Model is: {}'.format(hdp_cm_umass.get_coherence()))
	
	hdp_cm_cv = CoherenceModel(topics=hdp_topics, texts=all_reviews_text, dictionary=dictionary, coherence='c_v')
	print('The C_V coherence measure of the HDP model is: {}'.format(hdp_cm_cv.get_coherence()))
	print('\n')
	
	#prepared_hdp = pyLDAvis.gensim.prepare(hdp, corpus, dictionary)
	#display(pyLDAvis.display(prepared_hdp))
	
	#####################################################LSI (Latent Semantic Indexing)############################################################
	
	print('_'*70)
	print('\n')
	print('Performing LSI (Latent Semantic Indexing)... \n')

	## Cf. http://lsa.colorado.edu/papers/JASIS.lsi.90.pdf
	
	mod_list_lsi_umass, coh_list_lsi_umass = compute_coherence_values(LsiModel, dictionary, all_reviews_text, max_topics=20, 
                                                                     corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
                                                                     min_topics=2, stride=3, n_top_words=False, measure='u_mass', 
                                                                     lsi_flag=True)

	mod_list_lsi_cv, coh_list_lsi_cv = compute_coherence_values(LsiModel, dictionary, all_reviews_text, max_topics=20, 
																	 corpus=corpus, topics=False, transformed_vectorizer=False, tfidf_norm=False,
																	 min_topics=2, stride=3, n_top_words=False, measure='c_v', 
																	 lsi_flag=True)
																	 
	best_num_lsi_topics = show_best_num_topics('LSI', coh_list_lsi_umass, coh_list_lsi_cv, max_topics=20, min_topics=2, stride=3)
	print('The most coherent number of LSI topics to use is: {}'.format(best_num_lsi_topics))

	print('\n')
	print('_'*70)
	print('Performing LSI with the most coherent number of topics... \n')
	
	lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=best_num_lsi_topics, chunksize=1000)
	
	lsi_df = get_topics(lsi, best_num_lsi_topics)
	try:
		print('The top 20 words of each topic are: \n')
		print(lsi_df)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	lsi_hellinger_topic_distances = get_most_similar_topics(lsi, lsi_df, lsi_flag=True, num_topics=best_num_lsi_topics)
	print('\n')
	pretty_print(lsi_hellinger_topic_distances)
	
	lsi_dom_topics_df, lsi_dom_topics_final = get_dominant_topics(lsi, corpus, all_reviews_joined, lsi_flag=True)
	
	#lsi_dom_topics_final.style.applymap(color_green).applymap(make_bold)
	lsi_dom_topics_final_first30 = lsi_dom_topics_final.head(30)
	print('Note: For the LSI model, the (un-normalized) topic-probability with the largest magnitude is shown! \n')
	print('The probability contributions of the dominant topics of the first thirty documents are: \n')
	print(lsi_dom_topics_final_first30)
	#lsi_dom_topics_final_first30.style.applymap(color_green).applymap(make_bold)
	
	lsi_most_representative_docs = get_most_representative_docs(lsi_dom_topics_final, n_topics=20, lsi_flag=True)
	print('\n')
	print('_'*70)
	print('The most representative documents for the identified topics are: \n')
	print(lsi_most_representative_docs)
	#lsi_most_representative_docs.style.applymap(color_green).applymap(make_bold)
	
	lsi_topic_dist = get_topic_distribution(lsi_dom_topics_final, lsi_most_representative_docs, n_topics=20)
	print('\n')
	print('_'*70)
	print('The distribution of topics across the corpus documents is: \n')
	print(lsi_topic_dist)
	#lsi_topic_dist.style.applymap(color_green).applymap(make_bold)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVD 2D and 3D Visualizations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#token pattern prevents default splitting of hyphenated words
	#vectorizer = CountVectorizer(token_pattern='(?u)\\b[\\w-]+\\b', analyzer='word', max_features=5000)
	#x_counts = vectorizer.fit_transform(all_reviews_joined)

	#transformer = TfidfTransformer(smooth_idf=False)
	#x_tfidf = transformer.fit_transform(x_counts)
	
	#lsi_svd_2D = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=500, random_state=222)
	#lsi_2D = lsi_svd_2D.fit_transform(x_tfidf.T)
	
	#svd_2D_scatter(lsi_2D, vectorizer, color_scale_dimension=1)
	
	#lsi_svd_3D = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=500, random_state=222)
	#lsi_3D = lsi_svd_3D.fit_transform(x_tfidf.T)
	
	#svd_3D_scatter(lsi_3D, vectorizer, color_scale_dimension=0)
	
	####################################################NMF (Non-Negative Matrix Factorization)##################################################

	print('_'*70)
	print('\n')
	print('Performing NMF (Non-Negative Matrix Factorization)... \n')

	## Cf. https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
	
	#token pattern prevents default splitting of hyphenated words
	vectorizer = CountVectorizer(token_pattern='(?u)\\b[\\w-]+\\b', analyzer='word', max_features=5000)
	x_counts = vectorizer.fit_transform(all_reviews_joined)

	transformer = TfidfTransformer(smooth_idf=False)
	x_tfidf = transformer.fit_transform(x_counts)

	xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
	
	mod_list_nmf_umass, coh_list_nmf_umass = compute_coherence_values(NMF, dictionary, all_reviews_text, max_topics=20, 
                                                                  transformed_vectorizer=vectorizer, tfidf_norm=xtfidf_norm,
                                                                  min_topics=2, stride=3, n_top_words=20, measure='u_mass', 
                                                                  nmf_flag=True)

	mod_list_nmf_cv, coh_list_nmf_cv = compute_coherence_values(NMF, dictionary, all_reviews_text, max_topics=20, 
																	  transformed_vectorizer=vectorizer, tfidf_norm=xtfidf_norm,
																	  min_topics=2, stride=3, n_top_words=20, measure='c_v', 
																	  nmf_flag=True)
																	  
	best_num_nmf_topics = show_best_num_topics('NMF', coh_list_nmf_umass, coh_list_nmf_cv, max_topics=20, min_topics=2, stride=3)
	print('The most coherent number of NMF topics to use is: {}'.format(best_num_nmf_topics))
	
	print('\n')
	print('_'*70)
	print('Performing NMF with the most coherent number of topics... \n')
	
	nmf = NMF(n_components=best_num_nmf_topics, init='nndsvd', random_state=222)

	nmf.fit(xtfidf_norm)
	
	nmf_df = get_nmf_topics(nmf, best_num_nmf_topics, vectorizer, 20)
	try:
		print('The top 20 words of each topic are: \n')
		print(nmf_df)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	nmf_hellinger_topic_distances = get_most_similar_topics(nmf, nmf_flag=True, columns=nmf_df.columns)
	print('\n')
	pretty_print(nmf_hellinger_topic_distances)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NMF: Second Pass with U_Mass Coherence optimized number of components~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	print('~'*70)
	print('\n')
	print('Performing Second Pass of NMF (Non-Negative Matrix Factorization) using 14 Topics... \n')
	
	nmf_2 = NMF(n_components=14, init='nndsvd', random_state=222)

	nmf_2.fit(xtfidf_norm)
	
	nmf_df_2 = get_nmf_topics(nmf_2, 14, vectorizer, 20)
	try:
		print('The top 20 words of each topic are: \n')
		print(nmf_df_2)
	except Exception as e:
		print('Text encoding failure when printing to terminal! Carry on... \n')
		print(e.__doc__)
		print('\n')
	
	nmf_hellinger_topic_distances_2 = get_most_similar_topics(nmf_2, nmf_flag=True, columns=nmf_df_2.columns)
	print('\n')
	pretty_print(nmf_hellinger_topic_distances_2)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVD 2D Visualizations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	#svd = TruncatedSVD(n_components=2)
	#documents_2d = svd.fit_transform(x_tfidf)
	 
	#df = pd.DataFrame(columns=['x', 'y', 'document'])
	#df['x'], df['y'], df['document'] = documents_2d[:,0], documents_2d[:,1], range(len(all_reviews_text))
	 
	#source = ColumnDataSource(ColumnDataSource.from_df(df))

	#alternative syntax
	#source = ColumnDataSource({'x': np.array([documents_2d[:,0]]), 'y': np.array([documents_2d[:,1]]), 'document': np.array([range(len(all_reviews_text))])})

	#labels = LabelSet(x="x", y="y", text="document", y_offset=8,
	#				  text_font_size="8pt", text_color="#555555",
	#				  source=source, text_align='center')
	 
	#plot = figure(plot_width=1200, plot_height=1000)
	#plot.circle("x", "y", size=12, source=source, line_color="black", fill_alpha=0.8)
	#plot.add_layout(labels)

	#Cf. JSON Serialization issue: 
	#    https://github.com/bokeh/bokeh/issues/5439
	#    https://github.com/bokeh/bokeh/issues/6222
	#   https://github.com/bokeh/bokeh/issues/7523
	#try:
	#	show(plot, notebook_handle=True)
	#except Exception as e:
	#	print('Note!: {}'.format(e.__doc__))
	#	print(e)
	
	#svd = TruncatedSVD(n_components=2)
	#words_2d = svd.fit_transform(x_tfidf.T)
	 
	#df = pd.DataFrame(columns=['x', 'y', 'word'])
	#df['x'], df['y'], df['word'] = words_2d[:,0], words_2d[:,1], vectorizer.get_feature_names()
	 
	#source = ColumnDataSource(ColumnDataSource.from_df(df))
	#labels = LabelSet(x="x", y="y", text="word", y_offset=8,
	#				  text_font_size="8pt", text_color="#555555",
	#				  source=source, text_align='center')
	 
	#plot = figure(plot_width=1200, plot_height=1000)
	#plot.circle("x", "y", size=8, source=source, line_color="black", fill_alpha=0.8)
	#plot.add_layout(labels)

	#Cf. JSON Serialization issue: 
	#    https://github.com/bokeh/bokeh/issues/5439
	#    https://github.com/bokeh/bokeh/issues/6222
	#   https://github.com/bokeh/bokeh/issues/7523
	#try:
	#	show(plot, notebook_handle=True)
	#except Exception as e:
	#	print('Note!: {}'.format(e.__doc__))
	#	print(e)
	
	################################LDA Visualization: T-distributed Stochastic Neighbor Embedding with PCA initialization########################

	## Cf. http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
	
	#lda_base_model = lda.LDA(n_topics=10, n_iter=1000, random_state=111)
	
	#doc_x_tops = lda_base_model.fit_transform(x_counts)
	
	# angle close to 1 prioritizes speed over accuracy
	#tsne_model = TSNE(n_components=2, verbose=1, random_state=222, angle=.1, init='pca')

	# 10D -> 2D
	#tsne_lda = tsne_model.fit_transform(doc_x_tops)
	
	#20 colors
	#colormap = np.array([
	#	"#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
	#	"#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
	#	"#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
	#	"#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
	#])

	#number of displayed keywords
	#n_top_words = 10
	
	#Cf. JSON Serialization issue: 
	#    https://github.com/bokeh/bokeh/issues/5439
	#    https://github.com/bokeh/bokeh/issues/6222
	#   https://github.com/bokeh/bokeh/issues/7523
	#plot_tsne(all_reviews_joined, lda_base_model, vectorizer, doc_x_tops, tsne_lda, colormap, n_top_words=10)
	
	return None
	

if __name__ == '__main__':

	#environment and package versions
	print('\n')
	print("_"*70)
	print('The environment and package versions used in this script are:')
	print('\n')

	print(platform.platform())
	print('Python', sys.version)
	print('OS', os.name)
	print('Beautiful Soup', bs4.__version__)
	print('Urllib', urllib.request.__version__) 
	print('Regex', re.__version__)
	print('Numpy', np.__version__)
	print('Pandas', pd.__version__)
	print('Textacy', textacy.__version__)
	print('SpaCy', spacy.__version__)
	print('Gensim', gensim.__version__)
	print('LDA', lda.__version__)
	print('Sklearn', sklearn.__version__)
	#print('PyLDAvis', pyLDAvis.__version__)
	#print('IPython', IPython.__version__)
	#print('Matplotlib', matplotlib.__version__)
	#print('Plotly', plotly.__version__)
	#print('Cufflinks', cf.__version__)
	#print('Bokeh', bokeh.__version__)

	print('\n')
	print("~"*70)
	print('\n')

	main()

	print('\n')
	print('END OF PROGRAM')