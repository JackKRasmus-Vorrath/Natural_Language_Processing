#!/usr/bin/env python

import platform
import sys
import os

import zipfile
import re

import urllib
from urllib.request import urlopen

import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.text import Text
from nltk.tag.stanford import StanfordPOSTagger

import textwrap
from textwrap import fill

#environment and package versions
print('\n')
print("_"*70)
print('The environment and package versions used in this script are:')
print('\n')
print(platform.platform())
print('Python', sys.version)
print('OS', os.name)
print('Regex', re.__version__)
print('Urllib', urllib.request.__version__)
print('NLTK', nltk.__version__)
print('\n')
print("~"*70)
print('NOTE: The most recent version of the Stanford NLP POS Tagger is used in this script')
print('\n')
print('Running the script will download and unzip the tagger to the current working directory from the following location: \n')
print('http://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip')
print('\n')
print('For compatibility with Python, the script also requires that the path to the user\'s local Java executable is set in the OS environment \n')
print('Please also adjust the relative file paths for the .jar and .tagger model files as desired! \n')
print("~"*70)
print('\n')

def replace(string, substitutions):

    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

if __name__ == '__main__':

	#Set path to user Java executable
	java_path = "C:/Program Files/Java/jre1.8.0_131/bin/java.exe"
	os.environ['JAVAHOME'] = java_path
	
	#retrieve Stanford Core NLP POS Tagger zip file
	urllib.request.urlretrieve(r'http://nlp.stanford.edu/software/stanford-postagger-full-2017-06-09.zip', r'./stanford-postagger-full-2017-06-09.zip')
	
	#instantiate unzipper
	zfile_POS = zipfile.ZipFile(r'./stanford-postagger-full-2017-06-09.zip')
	
	#extract from zip to specified directory
	zfile_POS.extractall(r'./stanford-postagger')
	
	#set the direct path to the POS Tagger
	_model_filename = r'./stanford-postagger/stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger'
	_path_to_jar = r'./stanford-postagger/stanford-postagger-full-2017-06-09/stanford-postagger.jar'
	
	#initialize NLTK's Stanford POS Tagger API with the specified paths to the .tagger model and .jar file
	st = StanfordPOSTagger(model_filename=_model_filename, path_to_jar=_path_to_jar)
	
	print('\n')
	print("_"*70)
	print('QUESTION 1A: Stanford Core NLP POS Tagger: Longest Correct Sentence \n')
	
	#open the page with the e-book for Charles Dickens's 'A Tale of Two Cities'
	print('\n')
	print('The text being used is Charles Dickens\'s \'A Tale of Two Cities\'')
	raw_text = urlopen("http://www.gutenberg.org/files/98/98-0.txt").read()
	#UTF-8 decode the text
	decoded_text = raw_text.decode("utf-8")
	#extract only the content of the novel
	clean_text = re.findall('(?<=I. The Period).*?(?=End of the Project)', decoded_text, re.DOTALL)
	
	#remove carriage returns and newlines
	substitutions = {'\r': '', '\n': ' '}
	strip_text = replace(clean_text[0], substitutions)
	
	#sentence tokenization
	sent_tokens = sent_tokenize(strip_text)
	
	#word tokenization to find the sentence with maximum word count
	word_count = lambda sent_tokens: len(word_tokenize(sent_tokens))
	
	#longest sentence correctly tagged by Stanford POS Tagger
	max_42_sentence = sorted(sent_tokens, key=word_count)[-42]
	
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in max_42_sentence]
	output = ''.join(pieces)
	wrapped_max_42 = fill(output)

	print('\n')
	print('The longest sentence correctly tagged by the Stanford POS Tagger is: \n')
	print(wrapped_max_42)
	
	#Stanford POS Tagging output
	print('\n')
	print('The Stanford POS Tagging output of the above is: \n')
	stanfordnlp_pos_text_max_42 = st.tag(word_tokenize(max_42_sentence))
	print(stanfordnlp_pos_text_max_42)
	
	print('\n')
	print("_"*70)
	print('QUESTION 1B: Stanford Core NLP POS Tagger: Shortest Incorrect Sentence \n')
	
	#shortest sentence incorrectly tagged by Stanford POS Tagger
	min_sentence = sorted(sent_tokens, key=word_count)[1200]
	
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in min_sentence]
	output = ''.join(pieces)
	wrapped_min = fill(output)

	print('\n')
	print('The shortest sentence incorrectly tagged by the Stanford POS Tagger is: \n')
	print(wrapped_min)
	
	#Stanford POS Tagging output
	print('\n')
	print('The Stanford POS Tagging output of the above is: \n')
	stanfordnlp_pos_text_min = st.tag(word_tokenize(min_sentence))
	print(stanfordnlp_pos_text_min)
	print('\n')
	print('''Note that the Stanford POS Tagger has mistakenly tagged the numerical pronoun \'one\' as a cardinal number. \n
A better tagging might recognize the collocation with the past progressive verb tense, as in \'one knitting\'''')

	print('\n')
	print("_"*70)
	print('QUESTION 2A: NLTK POS Tagger: Longest Correct Sentence \n')
	
	#longest sentence correctly tagged by the NLTK POS Tagger
	max_42_sentence = sorted(sent_tokens, key=word_count)[-42]
	
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in max_42_sentence]
	output = ''.join(pieces)
	wrapped_max_42 = fill(output)

	print('The longest sentence correctly tagged by the NLTK POS Tagger is: \n')
	print(wrapped_max_42)
	
	#NLTK POS Tagging output
	print('\n')
	print('The NLTK POS Tagging output of the above is: \n')
	nltk_pos_text_42 = pos_tag(word_tokenize(max_42_sentence))
	print(nltk_pos_text_42)
	print('\n')
	print('Note that the output is the same as it was for the Stanford POS tagger')

	print('\n')
	print('QUESTION 2A (continued): NLTK POS Tagger: Shortest Incorrect Sentence \n')
	
	#shortest sentence incorrectly tagged by NLTK POS Tagger
	min_sentence = sorted(sent_tokens, key=word_count)[1200]
	
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in min_sentence]
	output = ''.join(pieces)
	wrapped_min = fill(output)

	print('The shortest sentence incorrectly tagged by the NLTK POS Tagger is: \n')
	print(wrapped_min)
	
	#NLTK POS Tagging output
	print('\n')
	print('The NLTK POS Tagging output of the above is: \n')
	nltk_pos_text_min = pos_tag(word_tokenize(min_sentence))
	print(nltk_pos_text_min)
	print('\n')
	print('''Note that the NLTK POS Tagger has the same difficulty with the numerical pronoun \'one\', which it mislabels as a cardinal number. \n
Out of box, both the NLTK and Stanford implementations have difficulty collocating the pronominal \'one\' with nearby words indicative of its syntactical role.''')

	print('\n')
	print("_"*70)
	print('QUESTION 2B: Differences between the Stanford Core NLP and NLTK POS Taggers \n')
	print('There are nonetheless noticeable differences in performance when comparing these two taggers, particularly with longer, syntactically complex sentences. \n')
	
	max_sentence = max(sent_tokens, key=word_count)
	print('\n')
	
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in max_sentence]
	output = ''.join(pieces)
	wrapped_max = fill(output)

	print('The longest sentence in the entire book is: \n')
	print(wrapped_max)
	
	#Stanford POS Tagging output
	print('\n')
	print('The Stanford POS Tagging output of the above is: \n')
	stanfordnlp_pos_text_max = st.tag(word_tokenize(max_sentence))
	print(stanfordnlp_pos_text_max)
	print('\n')
	
	#NLTK POS Tagging output
	print('\n')
	print('The NLTK POS Tagging output of the above is: \n')
	nltk_pos_text_max = pos_tag(word_tokenize(max_sentence))
	print(nltk_pos_text_max)
	print('\n')
	print('''Although both taggers have shared difficulties handling scare quotes without adequate preprocessing, the Stanford tagger makes only four mistakes. \n
The word \'highway\' as in \'highway robberies\' should be an adjective (JJ), not a noun (NN). \n
The word \'that\' as in \'that magnificent potentate\' is a demonstrative adjective determiner (WDT), not a subordinating conjunction (IN). \n
The word \'Court\' as in \'Court drawing-rooms\' should be a proper adjective (JJ), not a proper noun (NNP). \n
And the word \'fired\' as in \'the mob fired on the musketeers\' should be a past tense verb form (VBD), not a past participle (VBN). \n\n
By comparison, seven mistakes are made with the NLTK tagger. \n
The word \'daring\' as in \'daring burglaries\' should be an adjective (JJ), not a gerund (VBG). \n 
The word \'armed\' as in \'armed men\' should also be an adjective (JJ), not a past participle (VBN). \n
The word \'fellow-tradesman\' as in \'his fellow-tradesman\' should be a noun (NN), not an adjective (JJ). \n
The word \'rode\' as in \'[he] rode away\' should be a past tense verb form (VBD), not a base form (VB). \n
The word \'shot\' as in \'got shot\' should be a past participle (VBN), not an adjective (JJ). \n
The word \'magnificent\' as in \'that magnificent potentate\' should be an adjective (JJ), not a noun (NN). \n
And although it is capitalized, the word \'Court\', as with the Stanford tagger above, should be a proper adjective (JJ), not a proper noun (NNP). \n
In sum, while the Stanford Core NLP Tagger does appear somewhat better at handling long and complex syntax, it clearly shares a few weaknesses with the NLTK POS tagger.
''')

	print('\n')
	print("_"*70)
	print('QUESTION 3A: Hand-tagging of a Sentence from a Random News Article \n')
	print('The following sentence is taken from an article from (10/11/2018) on Hurricane Michael by the Associated Press: \n')
	news_sent = '''Mexico Beach is on the west end of what is sometimes called Florida's Forgotten Coast, 
so named because it is not heavily developed like many of the state's other shoreline areas, 
with their lavish homes and high-rise condos and hotels.'''
	#print with text wrapping
	format = '%s'
	pieces = [format % (word) for word in news_sent]
	output = ''.join(pieces)
	wrapped_news_sent = fill(output)
	print(wrapped_news_sent)
	print('\n')
	print('A manual tagging of the sentence is: \n')
	manual_tagging = [('Mexico', 'NNP'), ('Beach', 'NNP'), ('is', 'VBZ'), ('on', 'IN'), ('the', 'DT'), 
					('west', 'JJ'), ('end', 'NN'), ('of', 'IN'), ('what', 'WP'), ('is', 'VBZ'), 
					('sometimes', 'RB'), ('called', 'VBN'), ('Florida', 'NNP'), ("'s", 'POS'), 
					('Forgotten', 'NNP'), ('Coast', 'NNP'), (',', ','), ('so', 'RB'), ('named', 'VBN'), 
					('because', 'IN'), ('it', 'PRP'), ('is', 'VBZ'), ('not', 'RB'), ('heavily', 'RB'), 
					('developed', 'VBN'), ('like', 'IN'), ('many', 'DT'), ('of', 'IN'), ('the', 'DT'), 
					('state', 'NN'), ("'s", 'POS'), ('other', 'JJ'), ('shoreline', 'JJ'), ('areas', 'NNS'), 
					(',', ','), ('with', 'IN'), ('their', 'PRP$'), ('lavish', 'JJ'), ('homes', 'NNS'), 
					('and', 'CC'), ('high-rise', 'JJ'), ('condos', 'NNS'), ('and', 'CC'), ('hotels', 'NNS'), ('.', '.')]
	print(manual_tagging)
	
	print('\n')
	print("_"*70)
	print('QUESTION 3B: Using Stanford and NLTK POS Taggers on the above Sentence from a Random News Article \n')
	print('The Stanford POS Tagging output of the above is: \n')
	stanfordnlp_pos_news_text = st.tag(word_tokenize(news_sent))
	print(stanfordnlp_pos_news_text)
	
	print('\n')
	print('The NLTK POS Tagging output of the above is: \n')
	nltk_pos_news_text = pos_tag(word_tokenize(news_sent))
	print(nltk_pos_news_text)
	
	print('\n')
	print('Note that neither of the two taggers reproduced the output of the hand tagging procedure above. \n')
	
	print('\n')
	print("_"*70)
	print('QUESTION 3C: Hand Tagging vs. the Stanford and NLTK POS Taggers on the above Sentence from a Random News Article \n')
	print('''Note that both the Stanford and NLTK taggers incorrectly tagged the word \'many\' as in \'many of the [...] areas\'. \n
In this case \'many\' is a pronominal determiner (DT), not an adjective (JJ). \n
Interestingly, the NLTK tagger correctly tagged a word on which the Stanford tagger was unsuccessful. \n
The word \'shoreline\' as in \'other shoreline areas\' is an adjective (JJ), not a noun (NN).''')

	print('\n')
	print('END OF PROGRAM')
	