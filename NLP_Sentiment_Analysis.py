#!/usr/bin/env python

from __future__ import print_function

import platform
import sys
import os
import locale
import glob
from time import time

import multiprocessing

import warnings
warnings.simplefilter(action='ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

import codecs
import tarfile
from smart_open import smart_open

import requests
import bs4
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen

from random import sample, shuffle

from itertools import groupby
from more_itertools import unique_everseen
from operator import itemgetter
from collections import namedtuple, defaultdict, OrderedDict, Counter

import numpy as np
import pandas as pd

import re
import regex
import string
import textwrap
from textwrap import fill

#import pycontractions
#from pycontractions import Contractions
import autocorrect
from autocorrect import spell

import textacy
from textacy.preprocess import remove_punct

import spacy
from spacy.tokenizer import Tokenizer

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')

import afinn
from afinn import Afinn

import pattern
from pattern.en import sentiment as p_sent #, mood, modality

import gensim
from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from gensim.models.doc2vec import TaggedDocument
#must pip install testfixtures
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

import statsmodels
import statsmodels.api as sm

import imblearn
from imblearn.over_sampling import SMOTE, ADASYN

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import keras
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense#, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
#%matplotlib inline

#import seaborn as sns

#import plotly
#from plotly import tools
#import plotly.plotly as py
#import plotly.graph_objs as go
#import plotly.figure_factory as ff
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#connects JS to notebook so plots work inline
#init_notebook_mode(connected=True)

#import cufflinks as cf
#allow offline use of cufflinks
#cf.go_offline()

#####################################################HELPER FUNCTIONS####################################################################

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    return norm_text
	
def logistic_predictor_from_data(train_targets, train_regressors):
    """Fit a statsmodel logistic predictor on supplied data"""
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor

def error_rate_for_model(test_model, train_set, test_set, 
                         reinfer_train=False, reinfer_test=False, 
                         infer_steps=None, infer_alpha=None, infer_subsample=0.2):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc.sentiment for doc in train_set]
    if reinfer_train:
        train_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in train_set]
    else:
        train_regressors = [test_model.docvecs[doc.tags[0]] for doc in train_set]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if reinfer_test:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_set]
    test_regressors = sm.add_constant(test_regressors)
    
    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)
	
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
        
        span = soup.find('span', attrs={'rating-other-user-rating'})
        if span != None:
            stars = str(span.find('span').text).strip()
        else:
            stars = 5
        
        level_5_text.append((review_text[0], title, stars))
        
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
    for rev, ttl, strz in input_review_texts:
        print('Cleaning review #{} \n'.format(j+1))
        clean_rev = clean_component(rev, stop_words, tokenizer, puncts)
        print('The number of stars for this review is: {}'.format(strz))
        all_content.append(sentence_labeler(clean_rev, [tuple([ttl, float(strz)])]))
        j += 1

    print('Total Number of Movie Review Document Vectors: ', j)
    
    return all_content
	
def pattern_sentiment(review, threshold=0.1, verbose=False):
    analysis = p_sent(review)
    sentiment_score = round(analysis[0], 3)
    sentiment_subjectivity = round(analysis[1], 3)

    final_sentiment = 'positive' if sentiment_score >= threshold else 'negative'
    sent_binary = 1 if sentiment_score >= threshold else 0
    
    if verbose:
        #detailed sentiment statistics
        sentiment_frame = pd.DataFrame([[final_sentiment, sentiment_score, sentiment_subjectivity]],
        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                      ['Predicted Sentiment','Polarity Score','Subjectivity Score']],
                                      labels=[[0,0,0],[0,1,2]]))
        print(sentiment_frame, '\n')
        
        assessment = analysis.assessments
        assessment_frame = pd.DataFrame(assessment, 
                                        columns=pd.MultiIndex(levels=[['DETAILED ASSESSMENT STATS:'],
                                                                ['Key Terms', 'Polarity Score', 'Subjectivity Score','Type']],
                                                                labels=[[0,0,0,0],[0,1,2,3]]))
        #print(assessment_frame)

    return final_sentiment, sentiment_frame, assessment_frame, sent_binary
	
def afinn_sentiment(review, threshold, verbose=False):
    
    afn = Afinn(emoticons=True)
    sent_binary = (np.array(afn.score(review))>threshold).astype(int)
        
    final_sentiment = 'positive' if sent_binary == 1 else 'negative'
    
    if verbose:
        sentiment_frame = pd.DataFrame([[final_sentiment, sent_binary]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                            ['Predicted Sentiment', 'Binary Score']],
                                            labels=[[0,0],[0,1]]))
    
        print(sentiment_frame, '\n')
    
    return sent_binary
	
def sentiwordnet_sentiment(review, verbose=False):

    text_tokens = nltk.word_tokenize(review)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0
    
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        
        if 'NN' in tag and swn.senti_synsets(word, 'n'):
            ss_set = list(swn.senti_synsets(word, 'n'))
        elif 'VB' in tag and swn.senti_synsets(word, 'v'):
            ss_set = list(swn.senti_synsets(word, 'v'))
        elif 'JJ' in tag and swn.senti_synsets(word, 'a'):
            ss_set = list(swn.senti_synsets(word, 'a'))
        elif 'RB' in tag and swn.senti_synsets(word, 'r'):
            ss_set = list(swn.senti_synsets(word, 'r'))
        
        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            for s in ss_set:
                pos_score += s.pos_score()
                neg_score += s.neg_score()
                obj_score += s.obj_score()
            token_count += 1
        
    # aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    
    sent_binary = 1 if final_sentiment == 'positive' else 0
    
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score, norm_pos_score, norm_neg_score, norm_final_score]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                                        ['Predicted Sentiment','Objectivity','Positive', 'Negative','Overall']],
                                                        labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        
        print(sentiment_frame, '\n')
    
    return sent_binary
	
def vader_sentiment(review, threshold=0.1, verbose=False):
    
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    
    #get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold else 'negative'
    
    sent_binary = 1 if final_sentiment == 'positive' else 0 
    
    if verbose:
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive, negative, neutral]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                            ['Predicted Sentiment', 'Polarity Score', 'Positive', 'Negative', 'Neutral']],
                                            labels=[[0,0,0,0,0],[0,1,2,3,4]]))
    
    print(sentiment_frame, '\n')
    
    return sent_binary

def pretty_print(input_text):

	for r in input_text:
		pieces = [str(ele) for ele in r]
		for p in pieces:
			write_up = fill(p)
			try:
				print(write_up, '\n')
			except Exception as e:
				print('Encoding issue detected! \n')
				print(e.__doc__)
				print('\n')

	return None

#########################################################################################################################################	

# For 'Learning Word Vectors for Sentiment Analysis' by:
### Maas, A., Daly, R., Pham, P., Huang, D., Ng, A., and Potts, C.,
##    Cf. http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf

#########################################################################################################################################

def main():

	# Download and Clean Large Movie Review Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	## Cf. https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb

	dirname = 'aclImdb'
	filename = 'aclImdb_v1.tar.gz'
	locale.setlocale(locale.LC_ALL, 'C')
	all_lines = []

	if sys.version > '3':
		control_chars = [chr(0x85)]
	else:
		control_chars = [unichr(0x85)]

	if not os.path.isfile('./aclImdb/alldata-id.txt'):
		if not os.path.isdir(dirname):
			if not os.path.isfile(filename):
				# Download IMDB archive
				print("Downloading IMDB archive...")
				url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
				r = requests.get(url)
				with smart_open(filename, 'wb') as f:
					f.write(r.content)
			# if error here, try `tar xfz aclImdb_v1.tar.gz` outside notebook, then re-run this cell
			tar = tarfile.open(filename, mode='r')
			tar.extractall()
			tar.close()
		else:
			print("IMDB archive directory already available without download.")
			
		# Collect & normalize test/train data
		print("Cleaning up dataset...")
		folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
		for fol in folders:
			temp = u''
			newline = "\n".encode("utf-8")
			output = fol.replace('/', '-') + '.txt'
			# Is there a better pattern to use?
			txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
			print(" %s: %i files" % (fol, len(txt_files)))
			with smart_open(os.path.join(dirname, output), "wb") as n:
				for i, txt in enumerate(txt_files):
					with smart_open(txt, "rb") as t:
						one_text = t.read().decode("utf-8")
						for c in control_chars:
							one_text = one_text.replace(c, ' ')
						one_text = normalize_text(one_text)
						all_lines.append(one_text)
						n.write(one_text.encode("utf-8"))
						n.write(newline)
			
		# Save to disk for instant re-use on any future runs
		with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
			for idx, line in enumerate(all_lines):
				num_line = u"_*{0} {1}\n".format(idx, line)
				f.write(num_line.encode("utf-8"))
				
	assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
	print("Success, alldata-id.txt is available for next steps.")
	
	# Train/Test Split Large Movie Review Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	##   Cf. http://ai.stanford.edu/~amaas/data/sentiment/
	
	# this data object class suffices as a `TaggedDocument` (with `words` and `tags`) 
	# plus adds other state helpful for our later evaluation/reporting
	SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

	alldocs = []
	with smart_open('./aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
		for line_no, line in enumerate(alldata):
			tokens = gensim.utils.to_unicode(line).split()
			words = tokens[1:]
			tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
			split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
			sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
			alldocs.append(SentimentDocument(words, tags, split, sentiment))

	train_docs = [doc for doc in alldocs if doc.split == 'train']
	test_docs = [doc for doc in alldocs if doc.split == 'test']

	print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))
	
	# Shuffle Reviews for Better Learning~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	doc_list = alldocs[:]  
	shuffle(doc_list)
	
	# Train Distributed Bag-of-Words (DBOW) Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	## Cf. https://arxiv.org/pdf/1607.05368.pdf
	
	cores = multiprocessing.cpu_count()
	assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

	simple_models = [
		# PV-DBOW plain
		Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, 
				epochs=20, workers=cores),
		# PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
		#Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0, 
		#        epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
		# PV-DM w/ concatenation - big, slow, experimental mode
		# window=5 (both sides) approximates paper's apparent 10-word total window size
		#Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0, 
		#        epochs=20, workers=cores),
	]

	for model in simple_models:
		model.build_vocab(alldocs)
		print("%s vocabulary scanned & state initialized" % model)

	models_by_name = OrderedDict((str(model), model) for model in simple_models)
	
	#models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
	#models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])
	
	#models_by_name.items()
	
	for model in simple_models: 
		print("Training %s" % model)
		model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs)
		
	# Compute Error Rate of DBOW Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	#to selectively print only best errors achieved
	error_rates = defaultdict(lambda: 1.0)
	
	for model in simple_models:
		print("\nEvaluating error rate of %s" % model)
		err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
		error_rates[str(model)] = err_rate
		print("\n%f %s\n" % (err_rate, model))
		
	#for model in [models_by_name['dbow+dmm'], models_by_name['dbow+dmc']]: 
	#    print("\nEvaluating %s" % model)
	#    %time err_rate, err_count, test_count, predictor = error_rate_for_model(model, train_docs, test_docs)
	#    error_rates[str(model)] = err_rate
	#    print("\n%f %s\n" % (err_rate, model))	
	
	# Compare error rates achieved, best-to-worst
	print("Err_rate Model")
	for rate, name in sorted((rate, name) for name, rate in error_rates.items()):
		print("%f %s" % (rate, name))
		
	# Retrieve Arrays of Regressors and Targets~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		
	#dbow = simple_models[0]
	#dmm = simple_models[1]
	#dmc = simple_models[2]

	dbow = simple_models[0]	
		
	train_targets = [doc.sentiment for doc in train_docs]
	test_targets = [doc.sentiment for doc in test_docs]
		
	dbow_train_regressors = [dbow.docvecs[doc.tags[0]] for doc in train_docs]
	dbow_test_regressors = [dbow.docvecs[doc.tags[0]] for doc in test_docs]
	
	# Train and Evaluate Logistic Regression Classifier~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	clf = LogisticRegression()
	clf.fit(dbow_train_regressors, train_targets)
	
	print('\n')
	print('Logistic Classifier performance on the training set is: {}'.format(clf.score(dbow_train_regressors, train_targets)))
	print('Logistic Classifier performance on the test set is: {}'.format(clf.score(dbow_test_regressors, test_targets)))
	print('\n')
		
	#predict method to generate predictions from Logistic model and test data
	logistic_pred = clf.predict(dbow_test_regressors)

	threshold=0.5	
	
	print(confusion_matrix(np.array(test_targets), (logistic_pred>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(test_targets), (logistic_pred>threshold).astype(int)))

	#classification accuracy score
	logisitic_accuracy = accuracy_score(np.array(test_targets), (logistic_pred>threshold).astype(int))
	print("Correct classification rate:", logisitic_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(test_targets), (logistic_pred>threshold).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (Logisitic Binary Classifier) \n', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	# Shape Regressor Arrays for Input to Dense Neural Network (DNN) Classifier~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	dbow_train_input = np.stack(dbow_train_regressors, axis=0)
	dbow_test_input = np.stack(dbow_test_regressors, axis=0)
	
	# Train and Evaluate DBOW Dense Neural Network Classifier~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	K.clear_session()

	np.random.seed(222)
	model_d2v_01 = Sequential()
	model_d2v_01.add(Dense(64, activation='relu', input_dim=100))
	model_d2v_01.add(Dense(32, activation='relu'))
	model_d2v_01.add(Dense(16, activation='relu'))
	model_d2v_01.add(Dense(8, activation='relu'))
	model_d2v_01.add(Dense(1, activation='sigmoid'))
	model_d2v_01.compile(optimizer='rmsprop',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])

	model_d2v_01.fit(dbow_train_input, train_targets, validation_split=0.20,
					 #validation_data=(dbow_test_regressors, test_targets),
					 epochs=10, batch_size=32, verbose=2)
	
	#NB: saving model can throw segmentation fault!
	#model_d2v_01.save('./imdb_dbow_25000revs_0.2_split.hdf5', include_optimizer=True)
	
	#NB: early stopping model checkpoint can throw segmentation fault!

	#filepath="./d2v_dbow_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5"
	#filepath="d2v_dbow_best_weights.hdf5"
	#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	#early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max') 
	#callbacks_list = [checkpoint, early_stop]

	#np.random.seed(222)
	#d2v_dbow_es = Sequential()
	#d2v_dbow_es.add(Dense(32, activation='relu', input_dim=100))
	#d2v_dbow_es.add(Dense(8, activation='relu'))
	#d2v_dbow_es.add(Dense(4, activation='relu'))
	#d2v_dbow_es.add(Dense(1, activation='sigmoid'))
	#d2v_dbow_es.compile(optimizer='rmsprop',
	#              loss='binary_crossentropy',
	#              metrics=['accuracy'])

	#d2v_dbow_es.fit(dbow_train_input, train_targets, validation_split=0.25, 
	#                    epochs=50, batch_size=32, verbose=2, callbacks=callbacks_list)
	
	#predict method to generate predictions from DNN model and test data
	pred = model_d2v_01.predict(dbow_test_input)

	threshold=0.5
	
	print('\n')
	print(confusion_matrix(np.array(test_targets), (pred>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(test_targets), (pred>threshold).astype(int)))

	#classification accuracy score
	large_movie_rev_dataset_accuracy = accuracy_score(np.array(test_targets), (pred>threshold).astype(int))
	print("Correct classification rate:", large_movie_rev_dataset_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(test_targets), (pred>threshold).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (Distributed Bag-of-Words Dense Neural Network-based Sentiment Analyzer) \n \
	#					Performance on the \'Large Movie Review Dataset\'', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	# Train on all Data to Apply Model to External Test Set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	traintest_docs = [doc for doc in alldocs if doc.split in ['train','test']]
	
	dbow_all_regressors = [dbow.docvecs[doc.tags[0]] for doc in traintest_docs]
	dbow_all_targets = [doc.sentiment for doc in traintest_docs]

	dbow_all_input = np.stack(dbow_all_regressors, axis=0)
	
	K.clear_session()

	np.random.seed(999)
	model_d2v_all = Sequential()
	model_d2v_all.add(Dense(64, activation='relu', input_dim=100))
	model_d2v_all.add(Dense(32, activation='relu'))
	model_d2v_all.add(Dense(16, activation='relu'))
	model_d2v_all.add(Dense(8, activation='relu'))
	model_d2v_all.add(Dense(1, activation='sigmoid'))
	model_d2v_all.compile(optimizer='rmsprop',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])

	model_d2v_all.fit(dbow_all_input, dbow_all_targets, validation_split=0.20,
					 #validation_data=(dbow_test_regressors, test_targets),
					 epochs=10, batch_size=32, verbose=2)
					 
	# Crawl IMDB for 1000 reviews of the 100 Best and Worst Horror Movies~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	nlp = spacy.load('en')

	#cont = Contractions('../GoogleNews-vectors-negative300.bin.gz')
	#cont.load_models()

	punctuations = string.punctuation

	LabeledSentence1 = gensim.models.doc2vec.TaggedDocument

	tokenizer = Tokenizer(nlp.vocab)

	#stopwords = spacy.lang.en.STOP_WORDS
	#spacy.lang.en.STOP_WORDS.add("e.g.")
	#nlp.vocab['the'].is_stop
	#nlp.Defaults.stop_words |= {"(a)", "(b)", "(c)", "etc", "etc.", "etc.)", "w/e", "(e.g.", "no?", "s", 
	#                           "film", "movie","0","1","2","3","4","5","6","7","8","9","10","e","f","k","n","q",
	#                            "de","oh","ones","miike","http","imdb", "horror", "little", 
	#                            "come", "way", "know", "michael", "lot", "thing", "films", "later", "actually", "find", 
	#                            "big", "long", "away", "filmthe", "www", "com", "x", "aja", "agritos", "lon", "therebravo", 
	#                            "gou", "b", "particularly", "probably", "sure", "greenskeeper", "try", 
	#                            "half", "intothe", "especially", "exactly", "20", "ukr", "thatll", "darn", "certainly", "simply", }

	#stopwords = list(nlp.Defaults.stop_words)

	stopwords = list(["(a)", "(b)", "(c)", "etc", "etc.", "etc.)", "w/e", "(e.g.", "no?", "s", 
					 "0","1","2","3","4","5","6","7","8","9","10","e","f","k","n","q",
					 "de","oh","miike","http","imdb","michael","filmthe","www","com","x", 
					 "aja","agritos","lon","therebravo","gou","b","intothe","20", "ukr","thatll"])
					 
	#open the base URL webpage
	html_page = urlopen("https://www.imdb.com/list/ls059633855/")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')

	print('\n')
	review_text_first_5 = get_movie_reviews(soup, n_reviews_per_movie=5)
	
	all_good_movie_reviews_500 = get_tagged_documents(review_text_first_5, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	#open the base URL webpage
	html_page_bad = urlopen("https://www.imdb.com/list/ls061324742/")

	#instantiate beautiful soup object of the html page
	soup_bad = BeautifulSoup(html_page_bad, 'lxml')

	print('\n')
	review_text_first_5_bad = get_movie_reviews(soup_bad, n_reviews_per_movie=5)
	
	all_bad_movie_reviews_500 = get_tagged_documents(review_text_first_5_bad, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	# Combine all Reviews~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	all_reviews = all_good_movie_reviews_500 + all_bad_movie_reviews_500
	
	all_reviews_text = []
	all_reviews_ttl = []
	all_reviews_strz = []
	for i,j in all_reviews:
		if j[0][1] == 5:
			continue
		else:
			all_reviews_text.append(i)
			all_reviews_ttl.append(j[0][0])
			all_reviews_strz.append(j[0][1])

	#reg_ex = regex.compile('[^a-zA-Z]')

	#dictionary = corpora.Dictionary(all_reviews_text)

	#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
	#filtered_dict = dictionary.filter_extremes(no_below=1, no_above=0.8)

	#convert the dictionary to a bag of words corpus for reference
	#corpus = [dictionary.doc2bow(text) for text in all_reviews_text]

	all_reviews_joined = [' '.join(w) for w in all_reviews_text]

	print('\n')
	print('Number of reviews left after dropping scores of 5/10: \n')
	print(len(all_reviews_text), len(all_reviews_ttl), len(all_reviews_strz), len(all_reviews_joined))
	print('\n')
	
	# Use the Number of Stars as an Approximation to 'Ground Truth' Labels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	all_strz_binary = []
	for strz in all_reviews_strz:
		strz = 1 if strz > 5.0 else 0
		all_strz_binary.append(strz)
		
	true_targets = all_strz_binary
	
	# Use DBOW Model to Infer Vector space of Web-Scraped Reviews~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	rev_regressors = [dbow.infer_vector(rev, epochs=100) for rev in all_reviews_text]
	
	# Shape Regressor Arrays for Input to trained DNN Classifier~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	nn_rev_input = np.stack(rev_regressors, axis=0)
	
	# Evaluate trained DNN Classifier on Web-Scraped Reviews~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	#predict method to generate predictions from DNN model and test data
	pred_rev = model_d2v_all.predict(nn_rev_input)
	
	threshold=0.5
	
	print(confusion_matrix(np.array(true_targets), (pred_rev>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(true_targets), (pred_rev>threshold).astype(int)))

	#classification accuracy score
	dbow_dnn_horror_rev_accuracy = accuracy_score(np.array(true_targets), (pred_rev>threshold).astype(int))
	print("Correct classification rate:", dbow_dnn_horror_rev_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(true_targets), (pred_rev>threshold).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (Distributed Bag-of-Words Dense Neural Network-based Sentiment Analyzer) \n \
	#				Performance of the Trained Model on Scraped Reviews of the 100 Best and Worst Horror Movies', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	# Investigate Misclassified Reviews~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	true_targets_arr = np.array(true_targets)
	
	pred_rev_arr = pred_rev.reshape(len(pred_rev))
	pred_rev_arr = (pred_rev_arr>threshold).astype(int)
	
	misclass_indices_pos = np.where((true_targets_arr != pred_rev_arr) & (pred_rev_arr == 0))
	misclass_indices_neg = np.where((true_targets_arr != pred_rev_arr) & (pred_rev_arr == 1))
	
	misclass_revs_neg = [(all_reviews_joined[i], all_reviews_ttl[i], all_reviews_strz[i]) for i in misclass_indices_neg[0]]
	misclass_revs_pos = [(all_reviews_joined[i], all_reviews_ttl[i], all_reviews_strz[i]) for i in misclass_indices_pos[0]]
	
	print('The false positives produced by the DBOW Sentiment Classifier were: \n\n')
	pretty_print(misclass_revs_neg)
	
	print('The false negatives produced by the DBOW Sentiment Classifier were: \n\n')
	pretty_print(misclass_revs_pos)
	
	# Visualize Distributions of Targets and Predictions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	#x0 = pred_rev.flatten()
	#x1 = true_targets

	#trace1 = go.Histogram(
	#	x=x0,
	#	opacity=0.4,
	#	name='Predicted Class <br> Probabilities',
	#	xbins=dict(
	#		start=-0.1,
	#		end=1.1,
	#		size=0.05
	#	),
	#)
	#trace2 = go.Histogram(
	#	x=x1,
	#	opacity=0.4,
	#	name='True Classes',
	#	xbins=dict(
	#		start=-0.1,
	#		end=1.1,
	#		size=0.05
	#	),
	#)

	#data = [trace1, trace2]
	#layout = go.Layout(barmode='overlay',
	#					title='Distributed Bag-of-Words Dense Neural Network Classifier Results',
	#					xaxis=dict(
	#						title='Class Probabilities'
	#					),
	#					yaxis=dict(
	#						title='Number of Reviews'
	#					),)
	#fig = go.Figure(data=data, layout=layout)

	#iplot(fig, filename='overlaid histogram')
	
	# Apply SMOTE and ADASYN Oversampling to Negative Reviews~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	## Cf. https://arxiv.org/pdf/1106.1813.pdf
	##       http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.309.942&rep=rep1&type=pdf
	
	# Evaluate DBOW DNN Classifier Performance after Oversampling~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	X_SMOTE, y_SMOTE = SMOTE().fit_sample(nn_rev_input, true_targets)

	print('SMOTE upsampled counts of classes: \n')
	print(sorted(Counter(y_SMOTE).items()))
	print('\n')

	X_ADASYN, y_ADASYN = ADASYN().fit_sample(nn_rev_input, true_targets)

	print('ADASYN upsampled counts of classes: \n')
	print(sorted(Counter(y_ADASYN).items()))
	print('\n')

	print('Evaluting SMOTE and ADASYN upsampled models...')
	model_d2v_all.evaluate(X_SMOTE, y_SMOTE)
	model_d2v_all.evaluate(X_ADASYN , y_ADASYN)

	#predict method to generate predictions from DNN model and test data
	y_SMOTE_pred = model_d2v_all.predict(X_SMOTE)

	#predict method to generate predictions from DNN model and test data
	y_ADASYN_pred = model_d2v_all.predict(X_ADASYN)

	threshold=0.5

	print('\n')
	print(confusion_matrix(np.array(y_SMOTE), (y_SMOTE_pred>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(y_SMOTE), (y_SMOTE_pred>threshold).astype(int)))

	dbow_dnn_smote_accuracy = accuracy_score(np.array(y_SMOTE), (np.array(y_SMOTE_pred)>threshold).astype(int))
	print("Correct classification rate: (SMOTE) ", dbow_dnn_smote_accuracy)
	print('\n')

	print(confusion_matrix(np.array(y_ADASYN), (y_ADASYN_pred>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(y_ADASYN), (y_ADASYN_pred>threshold).astype(int)))

	dbow_dnn_adasyn_accuracy = accuracy_score(np.array(y_ADASYN), (np.array(y_ADASYN_pred)>threshold).astype(int))
	print("Correct classification rate: (ADASYN) ", dbow_dnn_adasyn_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)

	#model_types = 'DBOW DNN-based Sentiment Analyzer: \n SMOTE and ADASYN Upsampling Negative Reviews'

	#conf_matrix_SMOTE = confusion_matrix(np.array(y_SMOTE), (np.array(y_SMOTE_pred)>threshold).astype(int))
	#conf_matrix_ADASYN = confusion_matrix(np.array(y_ADASYN), (np.array(y_ADASYN_pred)>threshold).astype(int))

	#fig = plt.figure(constrained_layout=True, figsize=(26,13))

	#gs = GridSpec(1, 2, figure=fig)
	#ax1 = fig.add_subplot(gs[0, 0])
	#ax2 = fig.add_subplot(gs[0, -1])

	#sns.heatmap(conf_matrix_SMOTE, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax1)
	#ax1.set_xlabel('Predicted Label: (SMOTE)')
	#ax1.set_ylabel('True Label')

	#sns.heatmap(conf_matrix_ADASYN, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax2)
	#ax2.set_xlabel('Predicted Label: (ADASYN) ')
	#ax2.set_ylabel('True Label')

	#fig.suptitle('Confusion Matrices: {} \n'.format(model_types))

	#plt.show()
	#plt.tight_layout()
	
	# Apply Lexicon-based Sentiment Analyzers to Scraped Reviews##########################################################################
	
	## Implement Pattern Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	### Cf. https://www.clips.uantwerpen.be/pages/pattern-en#sentiment
	
	print('Implementing Pattern Sentiment Analyzer... \n')
	
	Rev_SentimentDocument = namedtuple('SentimentDocument', 'words ttl tags pattern_sent_score')

	i=0
	pattern_output = []
	for rev, ttl in zip(all_reviews_joined, all_reviews_ttl):
		final_sent, sent_df, assess_df, sent_binary = pattern_sentiment(rev, threshold=0.1, verbose=True)
		words = rev
		tags = i
		pattern_sent_score = float(sent_binary)
		pattern_output.append(Rev_SentimentDocument(words, ttl, tags, pattern_sent_score))
		try:
			print(fill(words[:250]), '\n')
			print('Movie title: {} \n'.format(ttl), '\n')
		except Exception as e:
			print('Encoding issue detected! \n')
			print(e.__doc__)
			print('\n')
		i += 1
		
	pattern_preds = [doc.pattern_sent_score for doc in pattern_output]
	
	## Evaluate Pattern Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print(confusion_matrix(np.array(true_targets), (np.array(pattern_preds)>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(true_targets), (np.array(pattern_preds)>threshold).astype(int)))

	#classification accuracy score
	pattern_accuracy = accuracy_score(np.array(true_targets), (np.array(pattern_preds)>threshold).astype(int))
	print("Correct classification rate:", pattern_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(true_targets), (np.array(pattern_preds)>threshold).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (Pattern Vocabulary-Based Sentiment Analyzer) \n', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	## Implement AFINN Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	### Cf. https://github.com/fnielsen/afinn
	
	print('Implementing AFINN Sentiment Analyzer... \n')
	
	afn = Afinn(emoticons=True)
	threshold = 0.0
	
	Rev_SentimentDocument = namedtuple('SentimentDocument', 'words ttl tags afinn_sent_score')

	i=0
	afinn_output = []
	for rev, ttl in zip(all_reviews_joined, all_reviews_ttl):
		sent_binary = afinn_sentiment(rev, threshold, verbose=True)
		words = rev
		tags = i
		afinn_sent_score = float(sent_binary)
		afinn_output.append(Rev_SentimentDocument(words, ttl, tags, afinn_sent_score))
		try:
			print(fill(words[:250]), '\n')
			print('Movie title: {} \n'.format(ttl), '\n')
		except Exception as e:
			print('Encoding issue detected! \n')
			print(e.__doc__)
			print('\n')
		i += 1

	afinn_preds_raw = [afn.score(rev) for rev in all_reviews_joined]
	afinn_preds = (np.array(afinn_preds_raw)>threshold).astype(int)
	
	## Evaluate AFINN Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print(confusion_matrix(np.array(true_targets), (np.array(afinn_preds)>threshold).astype(int)))
	print('\n')
	print(classification_report(np.array(true_targets), (np.array(afinn_preds)>threshold).astype(int)))

	#classification accuracy score
	afinn_accuracy = accuracy_score(np.array(true_targets), (np.array(afinn_preds)>threshold).astype(int))
	print("Correct classification rate:", afinn_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(true_targets), (np.array(afinn_preds)>threshold).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (AFINN Vocabulary-Based Sentiment Analyzer) \n', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	## Implement SentiWordNet Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print('Implementing SentiWordNet Sentiment Analyzer... \n')
	
	Rev_SentimentDocument = namedtuple('SentimentDocument', 'words ttl tags wn_sent_score')

	i=0
	sentiwordnet_output = []
	for rev, ttl in zip(all_reviews_joined, all_reviews_ttl):
		sent_binary = sentiwordnet_sentiment(rev, verbose=True)
		words = rev
		tags = i
		wn_sent_score = float(sent_binary)
		sentiwordnet_output.append(Rev_SentimentDocument(words, ttl, tags, wn_sent_score))
		try:
			print(fill(words[:250]), '\n')
			print('Movie title: {} \n'.format(ttl), '\n')
		except Exception as e:
			print('Encoding issue detected! \n')
			print(e.__doc__)
			print('\n')
		i += 1
	
	sentiwordnet_preds = [doc.wn_sent_score for doc in sentiwordnet_output]
	
	## Evaluate SentiWordNet Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print(confusion_matrix(np.array(true_targets), np.array(sentiwordnet_preds).astype(int)))
	print('\n')
	print(classification_report(np.array(true_targets), np.array(sentiwordnet_preds).astype(int)))

	#classification accuracy score
	sentiwordnet_accuracy = accuracy_score(np.array(true_targets), np.array(sentiwordnet_preds).astype(int))
	print("Correct classification rate:", sentiwordnet_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(true_targets), np.array(sentiwordnet_preds).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (SentiWordNet Vocabulary-Based Sentiment Analyzer) \n', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	## Implement VADER Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print('Implementing VADER Sentiment Analyzer... \n')
	
	Rev_SentimentDocument = namedtuple('SentimentDocument', 'words ttl tags vader_sent_score')

	i=0
	vader_output = []
	for rev, ttl in zip(all_reviews_joined, all_reviews_ttl):
		sent_binary = vader_sentiment(rev, verbose=True)
		words = rev
		tags = i
		vader_sent_score = float(sent_binary)
		vader_output.append(Rev_SentimentDocument(words, ttl, tags, vader_sent_score))
		try:
			print(fill(words[:250]), '\n')
			print('Movie title: {} \n'.format(ttl), '\n')
		except Exception as e:
			print('Encoding issue detected! \n')
			print(e.__doc__)
			print('\n')
		i += 1
		
	vader_preds = [doc.vader_sent_score for doc in vader_output]
	
	## Evaluate VADER Sentiment Analyzer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	print(confusion_matrix(np.array(true_targets), np.array(vader_preds).astype(int)))
	print('\n')
	print(classification_report(np.array(true_targets), np.array(vader_preds).astype(int)))

	#classification accuracy score
	vader_accuracy = accuracy_score(np.array(true_targets), np.array(vader_preds).astype(int))
	print("Correct classification rate:", vader_accuracy)
	print('\n')

	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)
	#conf_matrix = confusion_matrix(np.array(true_targets), np.array(vader_preds).astype(int))

	#plt.figure(figsize=(12, 10))
	#sns.heatmap(conf_matrix, annot=True, fmt="d", annot_kws={"size": 16});
	#plt.title('Confusion Matrix: (VADER Vocabulary-Based Sentiment Analyzer) \n', fontsize=20)
	#plt.ylabel('True label', fontsize=15)
	#plt.xlabel('Predicted label', fontsize=15)
	#plt.show()
	
	# Performance Evaluation across Lexicon-based Sentiment Analyzers~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	#Visualize confusion matrix as a heatmap
	#sns.set(font_scale=3)

	#model_types = 'Vocabulary-Based Sentiment Analyzers: \n Pattern, AFINN, SentiWordNet, and VADER'

	#threshold = 0.1
	#conf_matrix_pattern = confusion_matrix(np.array(true_targets), (np.array(pattern_preds)>threshold).astype(int))

	#threshold = 0.0
	#conf_matrix_afinn = confusion_matrix(np.array(true_targets), (np.array(afinn_preds)>threshold).astype(int))

	#conf_matrix_sentiwordnet = confusion_matrix(np.array(true_targets), np.array(sentiwordnet_preds).astype(int))

	#conf_matrix_vader = confusion_matrix(np.array(true_targets), np.array(vader_preds).astype(int))

	#fig = plt.figure(constrained_layout=True, figsize=(26,26))

	#gs = GridSpec(2, 2, figure=fig)
	#ax1 = fig.add_subplot(gs[0, 0])
	#ax2 = fig.add_subplot(gs[0, -1])
	#ax3 = fig.add_subplot(gs[-1, 0])
	#ax4 = fig.add_subplot(gs[-1, -1])

	#sns.heatmap(conf_matrix_pattern, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax1)
	#ax1.set_xlabel('Predicted Label: (Pattern)')
	#ax1.set_ylabel('True Label')

	#sns.heatmap(conf_matrix_afinn, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax2)
	#ax2.set_xlabel('Predicted Label: (AFINN) ')
	#ax2.set_ylabel('True Label')

	#sns.heatmap(conf_matrix_sentiwordnet, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax3)
	#ax3.set_xlabel('Predicted Label: (SentiWordNet) ')
	#ax3.set_ylabel('True Label')

	#sns.heatmap(conf_matrix_vader, annot=True, fmt="d", annot_kws={"size": 16}, ax=ax4)
	#ax4.set_xlabel('Predicted Label: (VADER) ')
	#ax4.set_ylabel('True Label')

	#fig.suptitle('Confusion Matrices: {} \n'.format(model_types))

	#plt.show()
	#plt.tight_layout()
	
	# Final Performance Comparison with DBOW DNN Sentiment Classifiers~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	accuracy_scores = dict([('DBOW_DNN_SMOTE',dbow_dnn_smote_accuracy), 
                       ('DBOW_DNN_ADASYN',dbow_dnn_adasyn_accuracy),
                       ('PATTERN',pattern_accuracy),
                       ('AFINN',afinn_accuracy),
                       ('SentiWordNet',sentiwordnet_accuracy),
                       ('VADER',vader_accuracy)])
					   
	sorted_accuracy_scores = [(k, accuracy_scores[k]) for k in sorted(accuracy_scores, key=accuracy_scores.get, reverse=True)]
	
	print('The Sorted Accuracy Scores of the Implemented Classifiers are: \n')
	for k,v in sorted_accuracy_scores:
		print(k + ':', v, '\n')
	
	print('The accuracy score of the best performing sentiment classifier is {}!'.format(sorted_accuracy_scores[0]))	

	return None
	
####################################################################################################################################
	
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
	print('NLTK', nltk.__version__)
	print('Pattern', pattern.__version__)
	print('Gensim', gensim.__version__)
	print('StatsModels', statsmodels.__version__)
	print('ImbLearn', imblearn.__version__)
	print('Sklearn', sklearn.__version__)
	print('Keras', keras.__version__)
	#print('Matplotlib', matplotlib.__version__)
	#print('Seaborn', sns.__version__)
	#print('Plotly', plotly.__version__)
	#print('Cufflinks', cf.__version__)

	print('\n')
	print("~"*70)
	print('\n')

	main()

	print('\n')
	print('END OF PROGRAM')	
