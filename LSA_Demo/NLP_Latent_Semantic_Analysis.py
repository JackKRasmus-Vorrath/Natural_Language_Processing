from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#processes XML file (record of patient info wrapped in tags)

info = open('./Veterans_Behavioral_Health.xml', 'r+').read()

soup = BeautifulSoup(info, 'lxml')		#lxml parser
conditionTxt = soup.findAll('measure_name')
conditionDocs = [x.text for x in conditionTxt]
#conditionDocs.pop(0)
conditionDocs = [x.lower() for x in conditionDocs]

stopset = set(stopwords.words('english'))
stopset.update(['with', 'on', 'to', 'were', 'appropriate', 'justification', ])

conditionTxt[0] 

vectorizer = TfidfVectorizer(stop_words = stopset, use_idf = True, ngram_range = (1, 3))
X = vectorizer.fit_transform(conditionDocs)

X[0]

# <1x80 sparse matrix of type '<type 'numpy.float64'>' with 9 stored elements in
# Compressed Sparse Row format>

print(X[0])

#(0, 23)	0.27959121985
#(0, 41)	0.356641132074
#(0, 56)	0.356641132074
#(0, 75)	0.177421729714
#(0, 24)	0.356641132074
#(0, 42)	0.356641132074
#(0, 57)	0.356641132074
#(0, 25)	0.356641132074
#(0, 43)	0.356641132074

# X: matrix where M is # docs, N # terms

# X = USV^(t)

#decomposition of three matrices U, S, T, picking a value k (# of concepts to keep)
#solving for 3 matrices whose product is approximately X

#U: m x k matrix (docs x concepts)
#S: k x k diagonal matrix (elements = amount of variation from each concept)
#V: n x k (transposed) matrix (terms x concepts)

X.shape
#(1143, 80)
#(documents x terms)

lsa = TruncatedSVD(n_components = 5, n_iter = 100)		#singular value decomposition

lsa.fit(X)

#engine for performing SVD:
#TruncatedSVD(algorithm = 'randomized', n_components=27, n_iter=100, random_state=None, tol=0.0)

#algorithm = SVD solution method; n_components = # of concepts to select; 
#n_iter = # of iterations in calculating SVD
#random_state = seed of random number generator; tol = precision, with 0 = machine precision

lsa.components_[0]

#first row for V: the tf-idf scores of all the terms that go with concept 0 (i.e., row 0)
# array([ 0.15894345, 0.15894345, 0.15894345, ..., 0.1967413, 0.12547969, 0.12547969])

#a measure of how important the term is to that concept; 
#position in the row corresponds to document position of the term



terms = vectorizer.get_feature_names()		#tf-idf vectorizer returns list of concept terms in same order
for i, comp in enumerate(lsa.components_):	#for all term tf-idf scores for each concept,
	termsInComp = zip(terms, comp)			#associate the corresponding words with the term tf-idf scores
	sortedTerms = sorted(termsInComp, key = lambda x:x[1], reverse = True)[:6]	#return 6 terms per concept in desc importance
	print("Concept %d:" % i)				#print concept number before list of terms
	for term in sortedTerms:				#for every term in the sorted concept list
		print(term[0])						#print it
	print(" ")								#separate the concept lists
	






