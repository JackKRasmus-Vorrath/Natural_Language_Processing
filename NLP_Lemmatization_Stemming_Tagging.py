#!/usr/bin/env python

import platform
import sys
import os

import re

import nltk
from nltk import word_tokenize
from nltk.text import Text
from nltk.metrics import edit_distance
from nltk import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words, wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import fuzzywuzzy
from fuzzywuzzy import fuzz

import textwrap
from textwrap import fill

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

#environment and package versions
print('\n')
print("_"*70)
print('The environment and package versions used in this script are:')
print('\n')
print(platform.platform())
print('Python', sys.version)
print('OS', os.name)
print('Regex', re.__version__)
print('NLTK', nltk.__version__)
print('FuzzyWuzzy', fuzzywuzzy.__version__)
print('SpaCy', spacy.__version__)

#function for part of speech tagging of text
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    #existential 'there' not supported by WordNet POS tagging
    elif treebank_tag.startswith('E'):
        return wordnet.ADV
    #personal pronoun
    elif treebank_tag.startswith('P'):
        return wordnet.NOUN

#function for removing prefixes
def stem_prefix(word, prefixes, lexicon):
    original_word = word
    for prefix in sorted(prefixes, key=len, reverse=True):
        # Use subn to track the no. of substitution made.
        # Allow dash in between prefix and root.
        if word.startswith(prefix):
            word, nsub = re.subn("{}[\-]?".format(prefix), "", word)
            if nsub > 0 and word in lexicon:
                return word
    return original_word

#spacy lemmatization function
def spacy_lemmatization(doc):
    # tokenize document in spacy
    doc_spacy = en_nlp(doc)    
    return [token.lemma_ for token in doc_spacy]

#function for part of speech tagging of filtered text
def get_wordnet_pos_spacy(treebank_tag):

    if treebank_tag.startswith('J'):
        return 'ADJ'
    #NB: 'King' incorrectly tagged as 'VGB'
    #      results in improper POS-tag lemmatization
    elif treebank_tag == 'VGB':
        return 'NOUN'
    elif treebank_tag == 'V':
        return 'VERB'
    elif treebank_tag.startswith('N'):
        return 'NOUN'
    #NB: SpaCy lemmatizer does not support ADV POS tags
    elif treebank_tag.startswith('R'):
        return 'ADJ'
    #NB: SpaCy lemmatizer does not support EX POS tags
    #existential 'there'
    #elif treebank_tag.startswith('E'):
        #return wordnet.ADV
    #personal pronoun
    elif treebank_tag.startswith('P'):
        return 'NOUN'

#convert list items to dictionary
def list_item_to_dict(**kwargs):
    for key in kwargs.keys():
        globals()[key]=kwargs[key]
    return kwargs

#compute similarity score for dictionary items
def comparative_score_lem(dictionary):
    results_dict = dict()
    for k,v in dictionary.items():
        doc = nlp(v)
        score = target_lem.similarity(doc)
        results_dict.update({k:score})
    
    return results_dict

#compute similarity score for dictionary items
def comparative_score_stem(dictionary):
    results_dict = dict()
    for k,v in dictionary.items():
        doc = nlp(v)
        score = target_stem.similarity(doc)
        results_dict.update({k:score})
    
    return results_dict
	

if __name__ == '__main__':

	print('\n')
	print("_"*70)
	print('QUESTION 1: Edit Distance and Fuzzy String Matching Percentage: \n')
	name = 'Jack'
	nickname = 'Jackaroo'

	print('Name is: {}'.format(name))
	print('Nickname is: {}'.format(nickname))
	print('\n')
	calculation = '''\n
	name = Jack 
	nickname = Jackaroo 

	Jack -- a, r, o, o <- edit distance = 4 

	J,a,c,k
	J,a,c,k,a,r,o,o <- total characters = 12 
			<- 8 of 12 match 
			<- fuzzy string matching = 66.67% 
		   '''
	print('The Calculation is: {}'.format(calculation))

	print('Edit Distance is:')
	print(edit_distance(name, nickname))
	print('\n')
	print('Fuzzy Matching Percentage is: \n')
	print('{}'.format(fuzz.ratio(name, nickname)), '%')
	print('\n')
	
	sents = '''It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,
it was the epoch of belief,
it was the epoch of incredulity,
it was the season of Light,
it was the season of Darkness,
it was the spring of hope,
it was the winter of despair,
we had everything before us,
we had nothing before us,
we were all going direct to Heaven,
we were all going direct the other way--
in short, the period was so far like the present period, that some of
its noisiest authorities insisted on its being received, for good or for
evil, in the superlative degree of comparison only.

There were a king with a large jaw and a queen with a plain face, on the
throne of England; there were a king with a large jaw and a queen with
a fair face, on the throne of France.'''
				
	print('The Raw Text is: \n')
	print(sents)
	print('\n')
	print('\n')
	
	print("_"*70)
	print('QUESTION 2: Stopwords and Text Identification: \n')
	
	#update stopwords list
	stopset = set(stopwords.words('english'))
	print('Three words are added to the default English Stopwords list: \n')
	print('1) us, 2) It, and 3) There \n')
	stopset.update(['us','It','There'])
	print('The Full Set of Stopwords being used is: \n\n {}'.format(stopset))
	
	#word tokenization, text object instantiation
	word_tokens = word_tokenize(str(sents))
	text = Text(word_tokens)
	print('\n')
	print('The Length of the Tokenized Text is: \n')
	print(len(text))
	print('\n')
	#filtering for alphabetical characters and excluding updated stopwords
	filtered_words = [word.lower() for word in text if word.isalpha() 
													  and word not in stopset]

	print('The Length of the Normalized Text filtered of Non-alphabetic Content and Stopwords is: \n')
	print(len(filtered_words))
	print('\n')
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in filtered_words]
	output = ', '.join(pieces)
	wrapped_filtered = fill(output)

	print('The Normalized Text filtered of Stop Words is: \n')
	print(wrapped_filtered)
	
	print('\n')
	print('\n')
	print('QUESTION 2: Write Up: \n')
	print('The interviewee correctly guessed the source of the text.')
	print('The text belongs to Charles Dickens\'s "A Tale of Two Cities".')
	print('The interviewee attributed their correct guess to how famous the first phrase is.')
	print('The first four content words were readily recognizable to anyone who has read the book.')
	print('No function words were needed to identify the source.')
	print('\n')
	print('\n')
	
	print("_"*70)
	print('QUESTION 3: Stemming and Lemmatization: \n')
	porter = PorterStemmer()
	lancaster = LancasterStemmer()
	snowball = SnowballStemmer('english')
	porter_stemming = [porter.stem(w) for w in filtered_words]
	lancaster_stemming = [lancaster.stem(w) for w in filtered_words]
	snowball_stemming = [snowball.stem(w) for w in filtered_words]
	
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in porter_stemming]
	output = ', '.join(pieces)
	wrapped_porter = fill(output)

	print('The Normalized, Filtered Text Stemmed with PorterStemmer is: \n')
	print(wrapped_porter)
	
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lancaster_stemming]
	output = ', '.join(pieces)
	wrapped_lancaster = fill(output)
	print('\n')
	print('The Normalized, Filtered Text Stemmed with LancasterStemmer is: \n')
	print(wrapped_lancaster)
	print('\n')
	
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in snowball_stemming]
	output = ', '.join(pieces)
	wrapped_snowball = fill(output)

	print('The Normalized, Filtered Text Stemmed with SnowballStemmer is: \n')
	print(wrapped_snowball)
	print('\n')
	
	lmtzr = WordNetLemmatizer()
	lemma_list = []
	for w in filtered_words:
		lemma_list.append(lmtzr.lemmatize(w))

	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemma_list]
	output = ', '.join(pieces)
	wrapped_WN_lem = fill(output)

	print('The Normalized, Filtered Text Lemmatized with WordNet is: \n')
	print(wrapped_WN_lem)
	print('\n')
	
	#lemmatizer using POS tagging
	lmtzr = WordNetLemmatizer()
	lemma_list = []
	for w,t in nltk.pos_tag(filtered_words):
		lemma_list.append(lmtzr.lemmatize(w, get_wordnet_pos(t)))
		
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemma_list]
	output = ', '.join(pieces)
	wrapped_WN_POS_lem = fill(output)

	print('The Normalized, Filtered Text POS-Tag Lemmatized with WordNet is: \n')
	print(wrapped_WN_POS_lem)
	
	#dictionary for removing prefixes

	# From https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes
	english_prefixes = {
	"anti": "",    # e.g. anti-goverment, anti-racist, anti-war
	"auto": "",    # e.g. autobiography, automobile
	#"de": "",      # e.g. de-classify, decontaminate, demotivate
	"dis": "",     # e.g. disagree, displeasure, disqualify
	"down": "",    # e.g. downgrade, downhearted
	"extra": "",   # e.g. extraordinary, extraterrestrial
	"hyper": "",   # e.g. hyperactive, hypertension
	"il": "",     # e.g. illegal
	"im": "",     # e.g. impossible
	#"in": "",     # e.g. insecure
	"ir": "",     # e.g. irregular
	"inter": "",  # e.g. interactive, international
	"mega": "",   # e.g. megabyte, mega-deal, megaton
	"mid": "",    # e.g. midday, midnight, mid-October
	"mis": "",    # e.g. misaligned, mislead, misspelt
	"non": "",    # e.g. non-payment, non-smoking
	"over": "",  # e.g. overcook, overcharge, overrate
	"out": "",    # e.g. outdo, out-perform, outrun
	"post": "",   # e.g. post-election, post-warn
	#"pre": "",    # e.g. prehistoric, pre-war
	"pro": "",    # e.g. pro-communist, pro-democracy
	"re": "",     # e.g. reconsider, redo, rewrite
	"semi": "",   # e.g. semicircle, semi-retired
	"sub": "",    # e.g. submarine, sub-Saharan
	"super": "",   # e.g. super-hero, supermodel
	"tele": "",    # e.g. television, telephathic
	"trans": "",   # e.g. transatlantic, transfer
	"ultra": "",   # e.g. ultra-compact, ultrasound
	"un": "",      # e.g. under-cook, underestimate
	"up": "",      # e.g. upgrade, uphill
	}
	
	#augmenting PorterStemmer with prefix removal
	def porter_english_plus(word, prefixes=english_prefixes):
		return porter.stem(stem_prefix(word, prefixes, lex))

	#augmenting LancasterStemmer with prefix removal
	def lancaster_english_plus(word, prefixes=english_prefixes):
		return lancaster.stem(stem_prefix(word, prefixes, lex))

	#augmenting SnowballStemmer with prefix removal
	def snowball_english_plus(word, prefixes=english_prefixes):
		return snowball.stem(stem_prefix(word, prefixes, lex))

	#augmenting Wordnet Lemmatizer with prefix removal
	def wordnet_lemma_plus(word, prefixes=english_prefixes):
		return lmtzr.lemmatize(stem_prefix(word, prefixes, lex))

	#lexicon of words for cross-referencing during stemming and lemmatization
	lex = list(wordnet.words()) + words.words()
	
	prefix_removed = []
	for w in filtered_words:
		prefix_removed.append(stem_prefix(w, english_prefixes, lex))
	
	lemma_list = []
	for w in prefix_removed:
		lemma_list.append(lmtzr.lemmatize(w))

	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemma_list]
	output = ', '.join(pieces)
	wrapped_WN_pre_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text with Prefixes Removed and Lemmtaized with WordNet is: \n')
	print(wrapped_WN_pre_lem)
	
	lemma_list = []
	for w,t in nltk.pos_tag(prefix_removed):
		lemma_list.append(lmtzr.lemmatize(w, get_wordnet_pos(t)))

	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemma_list]
	output = ', '.join(pieces)
	wrapped_WN_POS_pre_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text with Prefixes Removed and POS-Tag Lemmatized with WordNet is: \n')
	print(wrapped_WN_POS_pre_lem)

	# load spacy's English-language models
	en_nlp = spacy.load('en')
	
	spacy_output = spacy_lemmatization(' '.join(filtered_words))
	
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in spacy_output]
	output = ', '.join(pieces)
	wrapped_SP_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text Lemmatized with SpaCy is: \n')
	print(wrapped_SP_lem)
	
	spacy_output = spacy_lemmatization(' '.join(prefix_removed))
	
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in spacy_output]
	output = ', '.join(pieces)
	wrapped_SP_pre_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text with Prefixes Removed and Lemmatized with SpaCy is: \n')
	print(wrapped_SP_pre_lem)
	
	lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
	
	#lemmatizer using SpaCy POS tagging
	lemma_list = []
	for w,t in nltk.pos_tag(filtered_words):
		lemma_list.append(lemmatizer(w, get_wordnet_pos_spacy(t)))
		
	lemmas = []
	for l in lemma_list:
		lemmas.append([l][0][0])
		
	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemmas]
	output = ', '.join(pieces)
	wrapped_SP_POS_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text POS-Tag Lemmatized with SpaCy is: \n')
	print(wrapped_SP_POS_lem)
	
	#lemmatizer with prefixes removed using SpaCy POS tagging
	lemma_list = []
	for w,t in nltk.pos_tag(prefix_removed):
		lemma_list.append(lemmatizer(w, get_wordnet_pos_spacy(t)))
		
	lemmas = []
	for l in lemma_list:
		lemmas.append([l][0][0])

	#with wrapping
	format = '%s'
	pieces = [format % (word) for word in lemmas]
	output = ', '.join(pieces)
	wrapped_SP_POS_pre_lem = fill(output)
	print('\n')

	print('The Normalized, Filtered Text with Prefixes Removed and POS-Tag Lemmatized with SpaCy is: \n')
	print(wrapped_SP_POS_pre_lem)
	
	
	nlp = spacy.load('en_core_web_lg')
	
	best_rep_lem = '''good, time, bad, time, age, wisdom, age, foolishness, epoch, belief,
epoch, credulity, season, light, season, dark, spring, hope,
winter, despair, everything, nothing, go, direct, heaven, go,
direct, way, short, period, far, like, present, period, noisy,
authority, insist, receive, good, evil, superlative, degree,
comparison, king, large, jaw, queen, plain, face, throne, england,
king, large, jaw, queen, fair, face, throne, france'''
	
	best_rep_stem = '''good, time, bad, time, age, wis, age, fool, epoch, lief,
epoch, cred, season, light, season, dark, spring, hope,
winter, espair, everything, nothing, go, rect, heaven, go,
rect, way, short, od, far, like, sen, od, noise,
auth, sist, ceive, good, evil, lat, gree,
par, king, large, jaw, queen, plain, face, throne, england,
king, large, jaw, queen, fair, face, throne, france'''
	
	print('\n')
	print('\n')
	print("_"*70)
	print('A handcrafted ideal representation of the lemmatized text is: \n')
	print(best_rep_lem)
	
	print('\n')
	print('\n')
	print('A handcrafted ideal representation of the stemmed text is: \n')
	print(best_rep_stem)
	print('\n')
	print('\n')
	
	target_lem = nlp(best_rep_lem)
	target_stem = nlp(best_rep_stem)

	rep_list = [wrapped_filtered, wrapped_porter, wrapped_lancaster, wrapped_snowball, 
				wrapped_WN_lem, wrapped_WN_pre_lem, wrapped_WN_POS_lem, wrapped_WN_POS_pre_lem, 
				wrapped_SP_lem, wrapped_SP_pre_lem, wrapped_SP_POS_lem, wrapped_SP_POS_pre_lem]

	for l in range(len(rep_list)):
		rep_list[l] = rep_list[l].replace('\n', ' ')
		
	i = 0
	rep_dict = list_item_to_dict(stop_filtered = rep_list[i],
				   porter_stemmed = rep_list[i+1],
				   lancaster_stemmed = rep_list[i+2],
				   snowball_stemmed = rep_list[i+3],
				   WordNet_lemmed = rep_list[i+4],
				   WordNet_prefix_lemmed = rep_list[i+5],
				   WordNet_POS_lemmed = rep_list[i+6],
				   WordNet_POS_prefix_lemmed = rep_list[i+7],
				   SpaCy_lemmed = rep_list[i+8],
				   SpaCy_prefix_lemmed = rep_list[i+9],
				   SpaCy_POS_lemmed = rep_list[i+10],
				   SpaCy_POS_prefix_lemmed = rep_list[i+11])
				   
	all_scores_lem = comparative_score_lem(rep_dict)
	all_scores_stem = comparative_score_stem(rep_dict)
	
	sorted_scores_lem = [(k, all_scores_lem[k]) for k in sorted(all_scores_lem, 
                                        key=all_scores_lem.get, reverse=True)]
	print("_"*70)
	print('With respect to the ideal lemmatization, a list of procedures used, sorted by descending similarity score, is: \n')
	for s in sorted_scores_lem:
		print(s)
	print('\n')
		
	sorted_scores_stem = [(k, all_scores_stem[k]) for k in sorted(all_scores_stem, 
                                        key=all_scores_stem.get, reverse=True)]
	print('With respect to the ideal stemming, a list of procedures used, sorted by descending similarity score, is: \n')
	for s in sorted_scores_stem :
		print(s)
		
	print('\n')
	print('\n')
	print("_"*70)
	print('The Lemmatization procedure closest to the ideal output is: \n')
	if str(sorted_scores_lem[0][0]).endswith('lemmed'):
		print('The {}'.format(sorted_scores_lem[0][0][:-2]) + 'atization Procedure')
	else:
		print('The {}'.format(sorted_scores_lem[0][0][:-2]) + 'ing Procedure')
	
	print('\n')
	print('The output of the best performing (SpaCy) Lemmatization procedure is: \n')
	print(wrapped_SP_lem)
	
	print('\n')
	print('\n')
	print('The Stemming procedure closest to the ideal output is: \n')
	if str(sorted_scores_stem[0][0]).endswith('stemmed'):
		print('The {}'.format(sorted_scores_stem[0][0][:-2]) + 'ing Procedure')
	else:
		print('The {}'.format(sorted_scores_stem[0][0][:-2]) + 'atization Procedure')
	
	print('\n')
	print('The output of the best performing (Lancaster) Stemming procedure is: \n')
	print(wrapped_lancaster)
	
	print('\n')
	print('\n')
	print("_"*70)
	print('QUESTION 3: Write Up:')
	print('\n')
	
	print('''Note that few of the stemmers output valid morphological roots. \n
The less sophisticated stemmers often mistake words ending in \'s\' as
containing the morpheme indicating plurality. Words ending in \'e\' are also
mistakenly identified as morphemes indicating verb inflection. \n
What counts as the morphological root of a word will depend on the desired
level of granularity, and on whether part-of-speech is respected. \n
The root of a word like \'authority\' is arguably \'auth\', deriving from the
Latin \'auctor\' (originator), \'augeo\' (I originate), and the
Ancient Greek \'auxo\', from \'auxano\' (to make grow). Virtually none of the 
built-in stemmers or lemmatizers will return this kind of result, particularly 
because lemmatizers will only return words corresponding to the headword of the
lexicon with which it cross-references its output. \n
With respect to the previous example of \'authority\', the exception is the 
Lancaster Stemmer, which correctly identifies the root morpheme \'auth\'. 
Meanwhile, it also makes the typical stemming mistakes concerning plurality 
and inflection mentioned above. \n
Granting these limitations, nearly all the words of the output could be said 
to at least contain the valid morphological roots. \n
Taking \'incredulity\' as another example, none of the procedures output
the correct morphological root \'cred\', from Proto-Italic \'krezdo\' by way
of the Proto-Indo-European root \'ker\' (heart, as in: take to heart, trust,
or believe). But none of the procedures go so far as to lose the root. \n 
Again, the Lancaster Stemmer was the closest to capturing the root, 
returning \'incred\' as its output. If prefix removal had been applied, 
it would have found the correct morpheme. \n\n\n
If the best-performing SpaCy lemmatization procedure is taken as the model, 
the percent of the outputted tokens that contain or consist of valid 
morphological roots would be nearly 100%, the exception being the word \'noisiest\', 
a superlative form which it failed to reduce to the word \'noisy\'. 
\n\nAt 58 of 59 total words, the percentage of correctly outputted tokens returned 
by the best-performing procedure was: \n
{}%
'''.format((58/59)*100))
	print('\n')
	print('''If a stemming closest to the true morphological root is the goal,
and the Lancaster stemming procedure is taken as the model, the percent of
the outputted tokens that correspond to valid morphological roots would be
significantly lower. \n\nAt 26 of 59 total words, the percentage of correctly
outputted tokens returned by the best-performing procedure was: \n
{}%
'''.format((26/59)*100))
			
	print('\n')
	print('END OF PROGRAM')
	