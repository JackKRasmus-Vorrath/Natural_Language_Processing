#!/usr/bin/env python

import platform
import sys
import os

import re

import numpy as np

import bs4
from bs4 import BeautifulSoup

import urllib
from urllib.request import urlopen, Request
import gzip

import nltk
from nltk import word_tokenize
from nltk.text import Text
from nltk.corpus import PlaintextCorpusReader

#environment and package versions
print('\n')
print("_"*80)
print('The environment and package versions used in this script are:')
print('\n')
print(platform.platform())
print('Python', sys.version)
print('OS', os.name)
print('Regex', re.__version__)
print('Numpy', np.__version__)
print('Beautiful Soup', bs4.__version__)
print('Urllib', urllib.request.__version__)
print('NLTK', nltk.__version__)

#return vocab size, long vocab size, and lexical diversity score
def text_scores(text):
	#list of all lower-cased alphabetical tokens in text
	text_lower_alpha = list(w.lower() for w in text if w.isalpha())

	#length of all lower-cased alphabetical tokens
	token_size = len(text_lower_alpha)

	#length of all unique lower-cased alphabetical tokens
	vocab_size = len(set(text_lower_alpha))


	#list of all lower-cased alphabetical tokens in text of length GT 8
	text_lower_alpha_long = list(w.lower() for w in text if w.isalpha() and len(w) > 8)

	#length of all lower-cased alphabetical tokens of length GT 8
	token_size_long = len(text_lower_alpha_long)

	#length of all unique lower-cased alphabetical tokens of length GT 8
	vocab_size_long = len(set(text_lower_alpha_long))


	#diversity score: distinct types / n_tokens
	div_score = vocab_size/token_size

	return {"vocab_size":vocab_size, "vocab_size_long":vocab_size_long, "div_score":div_score}

#return dictionary result for the text with the max normalized vocab score	
def search_norm_vocab_scores(max_score, dict_list):
    return [x for x in dict_list if x['norm_vocab'] == max_score]
	
#return dictionary result for the text with the max normalized long vocab score	
def search_norm_vocab_long_scores(max_score, dict_list):
    return [x for x in dict_list if x['norm_vocab_long'] == max_score]

#return dictionary result for the text with the max text difficulty score		
def search_difficulty_scores(max_score, dict_list):
    return [x for x in dict_list if x['text_difficulty'] == max_score]
	
	
if __name__ == '__main__':

	#open the page with all the e-books
	html_page = urlopen("http://www.gutenberg.org/wiki/Children%27s_Instructional_Books_(Bookshelf)#Graded_Readers")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')
	
	base_urls = [ ("http:" + tag.attrs['href'], tag.text.replace('\n',' ') ) 
                            for tag in soup.findAll('a', attrs={'href': re.compile("^//www.")}) ]
	
	#Scrape each eBook hyperlink to find actual UTF-8 text file url link, and eBook title
	level_2_urls_text_utf8 = []
	for url in base_urls:
    
		soup = BeautifulSoup(urlopen(url[0]).read(), 'html.parser')
		href_tags = soup.find_all(href=True, text=u'Plain Text UTF-8')		
		update_url = 'http:' + href_tags[0].attrs['href']	
		level_2_urls_text_utf8.append((update_url, url[1]))
		
	max_vocab = 0
	max_vocab_text = None

	max_vocab_long = 0
	max_vocab_long_text = None

	score_list = []
	
	print("_"*80)
	print('\n')
	print('The titles, vocab scores, long vocab scores, and lexical diversity scores of all texts:')

	i = 0
	j = 0
	squint = '>_<'
	#for every (url,title) tuple ...
	for url in level_2_urls_text_utf8:
    #read in the raw text
		raw_text = None
		try:
			raw_text = urlopen(url[0]).read()
		except:
			i += 1
			print('\n')
			print(squint + ' I found {} URLs that I couldn\'t read!'.format(i))
			try:
				num_id = re.findall('\d+', url[0])[0]
				hyperlink = 'http://www.gutenberg.org/files/' + num_id + '/' + num_id + '-0.txt'
				print(hyperlink)
				url = (hyperlink, url[1])
				raw_text = urlopen(url[0]).read()
			except:
				print('It didn\'t work!')
				continue
		#decode the raw text
		decoded_text = None
		try:
			decoded_text = raw_text.decode("utf-8")
		except:
			j += 1
			print('\n')
			print(squint + ' I found {} texts that I couldn\'t decode!'.format(j))
			charset = urlopen(url[0]).info().get_content_charset()
			print('Content Character Set is: {}'.format(charset))
			try:
				req = Request(url[0])
				raw_text = urlopen(req)
				raw_text = raw_text.read()
				raw_text = gzip.decompress(raw_text)
				decoded_text = raw_text.decode('utf-8')
			except:
				print('It didn\'t work!')
				continue
		#remove all front and back matter	
		clean_text = re.findall('(?<=START OF ).*?(?=END OF)', decoded_text, re.DOTALL)
		#tokenize the text
		word_tokens = word_tokenize(str(clean_text))
		#instantiate each text as an nltk.text() object, named with its corresponding title
		text = Text(word_tokens, name=url[1])
		
		#generate scores using the defined function above
		scores = text_scores(text)
		#create title field in the resulting dictionary
		scores['title'] = text.name
		
		#update max vocab size
		if scores['vocab_size'] > max_vocab:
			max_vocab = scores['vocab_size']
			max_vocab_text = scores['title']
		
		#update max long vocab size
		if scores['vocab_size_long'] > max_vocab_long:
			max_vocab_long = scores['vocab_size_long']
			max_vocab_long_text = scores['title']
		
		#append the resulting dictionary to a list
		score_list.append(scores)
		
		#print the title and scores for each text
		print('\n')
		print("_"*40)
		print(scores['title'])
		print("Vocab_Size:" + ' ' + str(scores["vocab_size"]))
		print("Long_Vocab_Size:" + ' ' + str(scores["vocab_size_long"]))
		print("Lexical_Diversity_Score:" + ' ' + str(scores["div_score"]))
	
	#create fields for normalized vocab score, normalized long vocab scores, and weighted text difficulty score for each text
	for scores in score_list:
		scores['norm_vocab'] = np.sqrt(scores['vocab_size']/max_vocab)
		scores['norm_vocab_long'] = np.sqrt(scores['vocab_size_long']/max_vocab_long)
		scores['text_difficulty'] = np.average(np.array([scores['norm_vocab'],
														 scores['norm_vocab_long'],
														 scores['div_score']]),
											  weights=[0.33, 0.33, 0.33])
	
	#find and print the dictionary result of the text with the highest normalized vocab score
	print('\n')
	print("_"*80)
	print('The dictionary result of the text with the highest normalized vocab score:')
	print('\n')
	norm_vocab_scores = [x['norm_vocab'] for x in score_list]
	highest_norm_vocab_score = max(norm_vocab_scores)
	print(search_norm_vocab_scores(highest_norm_vocab_score, score_list))
	print('\n')
	
	#find and print the dictionary result of the text with the highest normalized long vocab score
	print("_"*80)
	print('The dictionary result of the text with the highest normalized long vocab score:')
	print('\n')
	norm_vocab_long_scores = [x['norm_vocab_long'] for x in score_list]
	highest_norm_vocab_long_score = max(norm_vocab_long_scores)
	print(search_norm_vocab_long_scores(highest_norm_vocab_long_score, score_list))
	print('\n')
	
	#find and print the dictionary result of the text with the highest difficulty score
	print("_"*80)
	print('The dictionary result of the text with the highest difficulty score:')
	print('\n')
	difficulty_scores = [x['text_difficulty'] for x in score_list]
	highest_difficulty_score = max(difficulty_scores)
	print(search_difficulty_scores(highest_difficulty_score, score_list))
	print('\n')
	
	#print the difficulty scores and titles of all texts, sorted by ascending difficulty score
	print("_"*80)
	print('\n')
	print('The difficulty scores and titles of all texts, sorted by ascending difficulty score:')
	print('\n')
	score_list_sorted_by_difficulty = sorted(score_list, key=lambda k: k['text_difficulty'])
	for d in score_list_sorted_by_difficulty:
		print("_"*50)
		print(d['text_difficulty'], d['title'])
		print('\n')
	
	#return the text difficulty rank and dictionary result of the first reader analyzed in HW_1
	print("_"*80)
	print("The text difficulty rank of McGuffey's Second Eclectic Reader is:")
	McGuffey_2ndGrade_Rank = next((index for (index, d) in enumerate(score_list_sorted_by_difficulty) 
									if d['title'] == "McGuffey's Second Eclectic Reader"), None) + 1
	print(McGuffey_2ndGrade_Rank)
	print('\n')
	print("The dictionary result of McGuffey's Second Eclectic Reader is:")
	print('\n')
	print([x for x in score_list_sorted_by_difficulty if x['title'] == "McGuffey's Second Eclectic Reader"])
	print('\n')
	
	#return the text difficulty rank and dictionary result of the second reader analyzed in HW_1
	print("_"*80)
	print("The text difficulty rank of McGuffey's Fourth Eclectic Reader is:")
	McGuffey_4thGrade_Rank = next((index for (index, d) in enumerate(score_list_sorted_by_difficulty) 
									if d['title'] == "McGuffey's Fourth Eclectic Reader"), None) + 1
	print(McGuffey_4thGrade_Rank)
	print('\n')
	print("The dictionary result of McGuffey's Fourth Eclectic Reader is:")
	print('\n')
	print([x for x in score_list_sorted_by_difficulty if x['title'] == "McGuffey's Fourth Eclectic Reader"])
	print('\n')
	
	#return the text difficulty rank and dictionary result of the third reader analyzed in HW_1
	print("_"*80)
	print("The text difficulty rank of McGuffey's Sixth Eclectic Reader is:")
	McGuffey_6thGrade_Rank = next((index for (index, d) in enumerate(score_list_sorted_by_difficulty) 
									if d['title'] == "McGuffey's Sixth Eclectic Reader"), None) + 1
	print(McGuffey_6thGrade_Rank)
	print('\n')
	print("The dictionary result of McGuffey's Sixth Eclectic Reader is:")
	print('\n')
	print([x for x in score_list_sorted_by_difficulty if x['title'] == "McGuffey's Sixth Eclectic Reader"])
	print('\n')
	print('\n')
	print('END OF PROGRAM')