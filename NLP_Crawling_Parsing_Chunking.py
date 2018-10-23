#!/usr/bin/env python

import platform
import sys
import os

import re

import bs4
from bs4 import BeautifulSoup

import urllib
from urllib.request import urlopen

import textwrap
from textwrap import fill

import spacy

def get_movie_reviews(soup_broth):
	
	base_urls = [ ("https://www.imdb.com" + tag.attrs['href'], tag.text.replace('\n',' ') ) 
							for tag in soup_broth.findAll('a', attrs={'href': re.compile("^/title/.*_tt")}) ]
							
	level_2_urls = []
	for url, title in base_urls:
		soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
		update_url = [("https://www.imdb.com" + tag.attrs['href']) 
							for tag in soup.findAll('a', attrs={'href': re.compile("^/title/.*tt_urv")})]
		level_2_urls.append((update_url[0], title))
		
	level_3_urls = []
	for url, title in level_2_urls:
		soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
		update_url = [("https://www.imdb.com" + soup.find('a', href=re.compile("^/review/.*tt_urv"))['href'])]
		level_3_urls.append((update_url[0], title))
		
	level_4_urls = []
	for url, title in level_3_urls:
		soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
		update_url = [("https://www.imdb.com" + soup.find('a', href=re.compile("^/review/.*rw_urv"))['href'])]
		level_4_urls.append((update_url[0], title))
		
	level_5_text = []
	for url, title in level_4_urls:
		soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
		review_text = [(soup.find('div', {'class' : re.compile("^text")}).text)]
		level_5_text.append((review_text[0], title))
		
	return level_5_text

def show_chunks(input_text):

	nlp = spacy.load('en')

	#print with text wrapping
	format = '%s'

	for text, title in input_text:
		chunk_list = []
		doc = nlp(text)
		for np in doc.noun_chunks:
			chunk_list.append(np)
			#print(np.text, np.root.text, np.root.dep_, np.root.head.text)
		try:
			pieces = [format % (word) for word in chunk_list]
			output = ' | '.join(pieces)
			np_chunks = fill(output)
			print(np_chunks)
			print('\n')
			print('Film title:  {}'.format(title))
			print('\n')
			print("-"*70)
			print('\n')
		except:
			continue
			
	return None
	
def main():
	
	print('QUESTION 1, Part I: Web Crawling: Extraction of Movie Review Permalinks \n')
	print("-"*70)
	print('Retrieving Reviews of the 100 Best Recent Horror Movies on IMDB \n')
	print('Please wait a few minutes... \n')
	
	#open the base URL webpage
	html_page = urlopen("https://www.imdb.com/list/ls059633855/")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')

	review_text = get_movie_reviews(soup)
	print('Reviews have been retrieved! Web Crawling process complete! \n')
	
	print('QUESTION 2, Part I: Shallow Parsing: Noun-Chunking the Review Texts of Good Horror Movies \n')
	print("-"*70)
	print('\n')
							
	show_chunks(review_text)
		
	print("_"*70)
	print('\n')
	
	print('QUESTION 1, Part II: Web Crawling: Extraction of Movie Review Permalinks \n')
	print("-"*70)
	print('Retrieving Reviews of the 100 Worst Horror Movies on IMDB \n')
	print('Please wait a few minutes... \n')
		
	#open the base URL webpage
	html_page = urlopen("https://www.imdb.com/list/ls061324742/")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')
	
	review_text = get_movie_reviews(soup)
	print('Reviews have been retrieved! Web Crawling process complete! \n')
	
	print('QUESTION 2, Part II: Shallow Parsing: Noun-Chunking the Review Texts of Bad Horror Movies \n')
	print("-"*70)
	print('\n')
	
	show_chunks(review_text)
	
	print("_"*70)
	print('\n')
	
	print('QUESTION 3: Summary Report of Crawling and NP-Chunking Procedure \n')
	print("-"*70)
	report = "The best 100 and worst 100 horror movies were selected from the IMDB website, which hosts pages where these films have been collected according to the user ratings they have received. BeautifulSoup was used to crawl from the base URLs to the permanent link for the first review of each film. The first review was selected because the page hosting all reviews for a given movie is ordered by default from most to least 'helpful'--a qualification determined according to the criteria of the proprietary algorithm used by IMDB. Choosing the 'most helpful' review for each movie also ensured that the written text was informative, clearly written, and grammatically and typographically correct, which facilitated the subsequent NP-chunking procedure. Tokenization was performed and noun phrases were chunked using the built-in shallow parsing noun-chunker provided with the SpaCy Python library. The majority of the time, the default 'English' model included with SpaCy was robust enough to correctly interpret and POS-tag unfamiliar words and proper nouns (e.g., character and actor names), obviating the need for manual updates of the built-in working lexicon. The decision to scrape reviews from as broad a selection of movies as possible would have otherwise rendered impractical any piecewise rule-based system for manually updating the vocabulary of each film. The title for each movie and the noun phrases output by the SpaCy NP-chunker were printed in the order in which the films appeared on the respective base URL from which the reviews were retrieved."
	format = '%s'
	pieces = [format % (word) for word in report]
	output = ''.join(pieces)
	write_up = fill(output)
	print(write_up)
	
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
	print('Regex', re.__version__)
	print('Beautiful Soup', bs4.__version__)
	print('Urllib', urllib.request.__version__)
	print('SpaCy', spacy.__version__)
	print('\n')
	print("~"*70)
	print('\n')
	
	main()
	
	print('\n')
	print('END OF PROGRAM')
	
	
	