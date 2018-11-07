#!/usr/bin/env python

import platform
import sys
import os
import time

import unicodedata
import re

import bs4
from bs4 import BeautifulSoup
import requests

import textwrap
from textwrap import fill

import pattern
from pattern.vector import Document, Model, TFIDF

def get_title_soup(input_url, header_details):

    source_code = requests.get(input_url, headers=header_details)
    plain_text = source_code.content
    soup_broth = BeautifulSoup(plain_text, "lxml")
    
    return soup_broth

def get_titles(input_url):
    
    title_list = []
    
    header_info = {'User-Agent': 'Mozilla/5.0'}   
    soup = get_title_soup(input_url, header_info)
    links = soup.findAll('a', {'class': re.compile("^a-link-normal s-access-detail-page")})
    for link in links:
        title_list.append(link.get('title'))
        
    next_page_url = [ ("https://www.amazon.com" + tag.attrs['href']) 
                    for tag in soup.findAll('a', {'href': re.compile("^/s/ref=sr_pg_2")}) ]
    
    time.sleep(60)
    
    header_info = {'User-Agent': 'Safari/537.36'}    
    more_soup = get_title_soup(next_page_url[0], header_info)   
    links = more_soup.findAll('a', {'class': re.compile("^a-link-normal s-access-detail-page")})
    for link in links:
        title_list.append(link.get('title'))
        
    return title_list
	
def get_capsule_soup(input_url, header_details):

    source_code = requests.get(input_url, headers=header_details)
    soup_broth = BeautifulSoup(source_code.text, "lxml")
    
    return soup_broth
	
def get_capsules(input_url):
    
    text_list = []
    
    header_info = {'User-Agent': 'Chrome/23.0.1271.97'}
    
    soup = get_capsule_soup(input_url, header_info)

    for st in soup.find_all(class_='st'):
        text_list.append(st.text)
        
    level_2_url = input_url + '&start=10'
    
    header_info = {'User-Agent': 'AppleWebKit/537.11'}
    
    more_soup = get_capsule_soup(level_2_url, header_info)
        
    for st in more_soup.find_all(class_='st'):
        text_list.append(st.text)
        
    level_3_url = input_url + '&start=20'
    
    header_info = {'User-Agent': 'Mozilla/5.0'}
    
    even_more_soup = get_capsule_soup(level_3_url, header_info)
        
    for st in even_more_soup.find_all(class_='st'):
        text_list.append(st.text)
        
    return text_list
	
def main():
	
	##############################################################################################
	print('QUESTION 1, Part I: Web Crawling: Extraction of Book Titles')
	print("-"*70)
	print('\n')
	print('Retrieving Book Titles from the first two pages of Amazon search results! \n')
	print('Please wait a minute... \n')
	
	print("~"*70)
	
	#open the base URL webpage
	level_1_url = "https://www.amazon.com/s?url=search-alias%3Daps&field-keywords=Martin+Heidegger"
	
	all_titles = get_titles(level_1_url)
	
	#print with text wrapping
	format = '%s'

	pieces = [format % (ttl) for ttl in all_titles]
	output = ' | '.join(pieces)
	ttls = fill(output)
	print('The scraped book titles are:')
	print("_"*40)
	print('\n')
	print('\n\n'.join(ttls.split('|')))
	print('\n')
	
	##############################################################################################
	print('QUESTION 1, Part II: Pairwise Text Cosine Similarity Scores of Book Titles')
	print("-"*70)
	print('\n')	
	
	doc_list = []
	for i in range(len(all_titles)):
		doc_list.append(Document(all_titles[i], type=" ".join(all_titles[i].split())))
		
	m = Model(documents=doc_list, weight=TFIDF)
	
	cos_similarities = [(m.similarity(x, y), m.documents[i].type, m.documents[j].type) for i,x in enumerate(m.documents) for j,y in enumerate(m.documents) if i != j]
	
	unique_cos_sim = [tuple(x) for x in set(map(frozenset, cos_similarities)) if len(tuple(x)) == 3]
	
	print('The number of calculated book title cosine similarity scores is: {} \n'.format(len(unique_cos_sim)))
	
	print('All non-zero book title cosine similarity scores, from smallest to largest: \n')
	for tup in sorted(unique_cos_sim):
		if tup[0] != 0:
			print(tup[0])
	print('\n')
	
	print("~"*70)
	
	#print with text wrapping
	format = '%s'

	pieces = [format % (sim,) for sim in sorted(unique_cos_sim, key=lambda t: t[0], reverse=True)[:5]]
	output = ' | '.join(pieces)
	sims = fill(output)
	print('The cosine similarity scores of the five most similar book titles are: \n')
	print('\n\n'.join(sims.split('|')))
	print('\n')
	
	print("~"*70)

	pieces = [format % (sim,) for sim in sorted(unique_cos_sim, key=lambda t: t[0], reverse=False)[:5]]
	output = ' | '.join(pieces)
	sims = fill(output)
	print('The cosine similarity scores of the five most dissimilar book titles are: \n')
	print('\n\n'.join(sims.split('|')))
	print('\n')
	
	#############################################################################################
	print('QUESTION 1, Part III: Most Similar and Dissimilar Book Titles and Search Rankings')
	print("-"*70)
	print('\n')	
	
	print('The most similar pair of book titles is: \n')
	print(max(unique_cos_sim))
	print('\n')
	
	print('The most dissimilar pair of book titles is: \n')
	print(min(unique_cos_sim))
	print('\n')
	
	print("~"*70)
	
	doc_types = [doc.type for doc in m.documents]
	
	print('The search ranking of the first element of the most similar book title pair is: \n')
	print(doc_types.index(max(unique_cos_sim)[1]))
	print('\n')
	
	print('The search ranking of the second element of the most similar book title pair is: \n')
	print(doc_types.index(max(unique_cos_sim)[2]))
	print('\n')
	
	print('The search ranking of the first element of the most dissimilar book title pair is: \n')
	print(doc_types.index(min(unique_cos_sim)[1]))
	print('\n')
	
	print('The search ranking of the second element of the most dissimilar book title pair is: \n')
	print(doc_types.index(min(unique_cos_sim)[2]))
	print('\n')
	
	#############################################################################################
	print('QUESTION 2, Part I: Web Crawling: Extraction of Search Capsules')
	print("-"*70)
	print('\n')
	
	orig_query = 'Ponderings XII–XV: Black Notebooks 1939–1941 (Studies in Continental Thought)'
	
	level_1_url = "https://www.google.com/search?q=" + orig_query.replace(' ','+')
	
	all_capsules =  get_capsules(level_1_url)
	
	all_capsules_clean = []
	for cp in all_capsules:
		all_capsules_clean.append(unicodedata.normalize('NFKD', cp).encode('ascii', 'ignore').decode('utf-8'))
	
	#print with text wrapping
	format = '%s'

	pieces = [format % (cap) for cap in all_capsules_clean]
	output = ' | '.join(pieces)
	caps = fill(output)
	print('The scraped capsules are:')
	print("_"*40)
	print('\n')
	print('\n\n'.join(caps.split('|')))
	print('\n')
	
	##############################################################################################
	print('QUESTION 2, Part II: Pairwise Text Cosine Similarity Scores of Search Capsules')
	print("-"*70)
	print('\n')
	
	query_list = []
	for i in range(len(all_capsules_clean)):
		query_list.append(Document(all_capsules_clean[i], type=" ".join(all_capsules_clean[i].split())))
	
	m = Model(documents=query_list, weight=TFIDF)
	
	cos_similarities = [(m.similarity(x, y), m.documents[i].type, m.documents[j].type) for i,x in enumerate(m.documents) for j,y in enumerate(m.documents) if i != j]
	
	unique_cos_sim = [tuple(x) for x in set(map(frozenset, cos_similarities)) if len(tuple(x)) == 3]
	
	resorted_cos_sim = []
	for i in range(len(unique_cos_sim)):
		resorted_cos_sim.append(sorted(tuple(str(e) for e in unique_cos_sim[i])))
		resorted_cos_sim[i][0] = float(resorted_cos_sim[i][0])
		resorted_cos_sim[i] = tuple(resorted_cos_sim[i])
	
	print('The number of calculated capsule cosine similarity scores is: {} \n'.format(len(resorted_cos_sim)))
	
	print('All non-zero capsule cosine similarity scores, from smallest to largest: \n')
	for tup in sorted(resorted_cos_sim):
		if tup[0] != 0:
			print(tup[0])
	print('\n')
	
	print("~"*70)
	
	#print with text wrapping
	format = '%s'

	pieces = [format % (sim,) for sim in sorted(resorted_cos_sim, key=lambda t: t[0], reverse=True)[:5]]
	output = ' | '.join(pieces)
	sims = fill(output)
	print('The Cosine Similarity scores of the five most similar capsule pairs are: \n')
	print('\n\n'.join(sims.split('|')))
	print('\n')
	
	print("~"*70)
	
	pieces = [format % (sim,) for sim in sorted(resorted_cos_sim, key=lambda t: t[0], reverse=False)[:5]]
	output = ' | '.join(pieces)
	sims = fill(output)
	print('The Cosine Similarity scores of the five most dissimilar capsule pairs are: \n')
	print('\n\n'.join(sims.split('|')))
	print('\n')
	
	print("~"*70)
	
	print('Finding the capsule with the highest cosine similarity to the original query... \n')
	all_capsules_clean.append(orig_query)
	
	caps_and_query = []
	for i in range(len(all_capsules_clean)):
		caps_and_query.append(Document(all_capsules_clean[i], type=" ".join(all_capsules_clean[i].split())))
		
	m = Model(documents=caps_and_query, weight=TFIDF)
	
	cos_similarities = [(m.similarity(x, y), m.documents[i].type, m.documents[j].type) for i,x in enumerate(m.documents) for j,y in enumerate(m.documents) if i != j]
	
	unique_cos_sim_query = [tuple(x) for x in set(map(frozenset, cos_similarities)) if len(tuple(x)) == 3]
	
	resorted_cos_sim_query = []
	for i in range(len(unique_cos_sim_query)):
		resorted_cos_sim_query.append(sorted(tuple(str(e) for e in unique_cos_sim_query[i])))
		resorted_cos_sim_query[i][0] = float(resorted_cos_sim_query[i][0])
		resorted_cos_sim_query[i] = tuple(resorted_cos_sim_query[i])
		
	result_list = []
	for tup in resorted_cos_sim_query:
		if orig_query in tup:
			result_list.append(tup)
			
	result_tup = max(result_list, key=lambda x:x[0])
	print('The cosine similarity score of the capsule most similar to the original query is: \n')
	print(result_tup)
	print('\n')
	
	print('Finding search ranking of the capsule with the highest cosine similarity to the original query... \n')
	
	match_list = []
	for item in all_capsules_clean:
		match_list.append(item.replace('\n',''))
	
	print('The search ranking of the capsule most similar to the original query is: \n')
	print(match_list.index(result_tup[1]))
	print('\n')
	
	#############################################################################################
	print('QUESTION 2, Part III: Most Similar and Dissimilar Capsules and Search Rankings')
	print("-"*70)
	print('\n')
	
	print('The most similar pair of capsules is: \n')
	print(max(resorted_cos_sim))
	print('\n')
	
	print('The most dissimilar pair of capsules is: \n')
	print(min(resorted_cos_sim))
	print('\n')
	
	print("~"*70)
	
	doc_types = [doc.type for doc in m.documents]
	
	print('The search ranking of the first element of the most similar capsule pair is: \n')
	print(doc_types.index(max(resorted_cos_sim)[1]))
	print('\n')
	
	print('The search ranking of the second element of the most similar capsule pair is: \n')
	print(doc_types.index(max(resorted_cos_sim)[2]))
	print('\n')
	
	print('The search ranking of the first element of the most dissimilar capsule pair is: \n')
	print(doc_types.index(min(resorted_cos_sim)[1]))
	print('\n')
	
	print('The search ranking of the second element of the most dissimilar capsule pair is: \n')
	print(doc_types.index(min(resorted_cos_sim)[2]))
	print('\n')
	
	############################################################################################
	
	print('Summary Report: Document Similarity Semantic Analysis')
	print("-"*70)
	################
	report = "A crawler with changing user-agent headers was used to scrape book titles on Amazon from the first two pages of results returned when searching the philosopher, Martin Heidegger. Using TF-IDF values derived from a model incorporating the scraped results, all pairwise cosine similarity scores were calculated for the corpus documents, each of which consisted of the book title and any accompanying subtitle text. The scores were filtered for unique book title pairs and sorted by ascending cosine similarity score, so the top 5 and bottom 5 pairs could be printed in terminal. As several pairings returned a cosine similarity score of 0, the most dissimilar pair among the lowest scores could not be decisively quantified. Interestingly, search rankings of the elements of the most similar and dissimilar pairs did not appear on the same page of results. Another crawler was used to scrape capsules returned by a Google search for one of the book titles appearing in the Amazon results. Capsules from the first three pages of Google results were Unicode normalized and decoded before they were incorporated into another model, from which TF-IDF values were derived. All pairwise cosine similarity scores were calculated for the new set of corpus documents, which consisted of all text appearing in each capsule. Scores were filtered for unique capsule pairs and sorted by ascending cosine similarity score; the top 5 and bottom 5 pairs were again printed in terminal. To identify the capsule most similar to the original query, the latter was then included in the model, from which a new set of TF-IDF values and cosine similarity scores were generated. Interestingly, the ranking of the most similar capsule appeared lower in the search results than expected, on the bottom of the second page. Intuitively, the search rankings of the capsules most similar to one another did, however, appear on the same page of Google results."
	##############
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
	print('Requests', requests.__version__)
	print('Pattern', pattern.__version__)
	print('\n')
	print("~"*70)
	print('\n')
	
	main()
	
	print('\n')
	print('END OF PROGRAM')