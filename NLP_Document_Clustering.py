#!/usr/bin/env python

import platform
import sys
import os
from time import time
from itertools import groupby

import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import bs4
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen

import re
import string
import textwrap
from textwrap import fill

import pycontractions
from pycontractions import Contractions
import autocorrect
from autocorrect import spell

import textacy
from textacy.preprocess import remove_punct

import spacy
from spacy.tokenizer import Tokenizer

import gensim
from gensim.models import Doc2Vec

import sklearn
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold

import scipy
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

#VISUALIZATION LIBRARIES (used for interactive development)
#import matplotlib
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~HELPER FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_movie_reviews(soup_broth):
    
    print('Retrieving all baseline URLs... \n')

    base_urls = [ ("https://www.imdb.com" + tag.attrs['href'], tag.text.replace('\n',' ') ) 
                            for tag in soup_broth.findAll('a', attrs={'href': re.compile("^/title/.*_tt")}) ]
    
    print('Retrieving all second-level URLs... \n')

    level_2_urls = []
    for url, title in base_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + tag.attrs['href']) 
                            for tag in soup.findAll('a', attrs={'href': re.compile("^/title/.*tt_urv")})]
        level_2_urls.append((update_url[0], title))
        
    print('Retrieving all third-level URLs... \n')

    level_3_urls = []
    for url, title in level_2_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + soup.find('a', href=re.compile("^/review/.*tt_urv"))['href'])]
        level_3_urls.append((update_url[0], title))
        
    print('Retrieving all fourth-level URLs... \n')

    level_4_urls = []
    for url, title in level_3_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        update_url = [("https://www.imdb.com" + soup.find('a', href=re.compile("^/review/.*rw_urv"))['href'])]
        level_4_urls.append((update_url[0], title))
        
    print('Retrieving all fifth-level URLs... \n')

    level_5_text = []
    for url, title in level_4_urls:
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        review_text = [(soup.find('div', {'class' : re.compile("^text")}).text)]
        level_5_text.append((review_text[0], title))
        
    print('All reviews retrieved! \n')

    return level_5_text
	
def clean_component(review, contract_model, stop_words, tokenizer, puncts):
    """Text Cleaner: Expand Contractions, Tokenize, Remove Stopwords, Punctuation, Lemmatize, Spell Correct, Lowercase"""
    
    rev_contract_exp = list(contract_model.expand_texts([review], precise=True))
    
    doc_tok = tokenizer(rev_contract_exp[0])

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
	
def get_tagged_documents(input_review_texts, contract_model, stop_words, tokenizer, puncts, sentence_labeler):
    print('Creating Tagged Documents... \n')
    
    all_content = []
    j=0
    for rev, ttl in input_review_texts:
        print('Cleaning review #{}'.format(j+1))
        clean_rev = clean_component(rev, contract_model, stop_words, tokenizer, puncts)
        all_content.append(sentence_labeler(clean_rev, [ttl]))
        j += 1

    print('Total Number of Movie Review Document Vectors: ', j)
    
    return all_content
	
def pretty_print(input_text, ttl):
    
    format = '%s'
    pieces = [format % (word) for word in input_text]
    output = ' '.join(pieces)
    write_up = fill(output)
    print('\n')
    print('_'*70)
    print(write_up)
    print('\n')
    print(ttl)
    print('-'*70)
    
    return None
	
def get_most_similar(input_rev, all_revs, d2v_model):
    
    most_sim_tag = d2v_model.docvecs.most_similar(input_rev)[0][0]

    for rev in all_revs:
        if rev[1][0] == most_sim_tag:
            print('\n')
            print('The Review Document Most Similar to {} is: '.format(input_rev))
            pretty_print(rev[0], rev[1])
            print('Their Cosine Similarity Score is: ')
            print(d2v_model.docvecs.similarity(input_rev, rev[1][0]))
            print('_'*70)
            print('\n')
            
    return None
	
def get_most_dissimilar(input_rev, all_revs, d2v_model):
    
    sim_list = []
    for rev, ttl in all_revs:
        sim_list.append((d2v_model.docvecs.similarity(input_rev, ttl[0]), ttl[0]))

    least_sim_tag = sorted(sim_list)[0][1]

    for rev in all_revs:
        if rev[1][0] == least_sim_tag:
            print('\n')
            print('The Review Document Most Dissimilar to {} is: '.format(input_rev))
            pretty_print(rev[0], rev[1])
            print('Their Cosine Similarity Score is: ')
            print(d2v_model.docvecs.similarity(input_rev, rev[1][0]))
            print('_'*70)
            print('\n')
            
    return None
	
def doc2vec_similarity(tagged_reviews, review_type):

    d2v_model = Doc2Vec(tagged_reviews, vector_size = 5000, window = 5, 
                                                min_count = 5, workers = 8, dm = 1, 
                                                alpha=0.025, min_alpha=0.001)

    d2v_model.train(tagged_reviews, total_examples=d2v_model.corpus_count, epochs=300)

    print('According to IMDB, the Most Helpful Review on the 100 {} Horror Movies List is: '.format(review_type))

    pretty_print(tagged_reviews[0][0], tagged_reviews[0][1])

    print('Most Similar Review Documents: \n')
    print(d2v_model.docvecs.most_similar(tagged_reviews[0][1][0]))
    print('_'*70)

    get_most_similar(tagged_reviews[0][1][0], tagged_reviews, d2v_model)

    get_most_dissimilar(tagged_reviews[0][1][0], tagged_reviews, d2v_model)

    print('According to IMDB, the Least Helpful Review on the 100 {} Horror Movies List is: '.format(review_type))

    pretty_print(tagged_reviews[-1][0], tagged_reviews[-1][1])

    print('Most Similar Review Documents: \n')
    print(d2v_model.docvecs.most_similar(tagged_reviews[-1][1][0]))
    print('_'*70)

    get_most_similar(tagged_reviews[-1][1][0], tagged_reviews, d2v_model)

    get_most_dissimilar(tagged_reviews[-1][1][0], tagged_reviews, d2v_model)

    return d2v_model
	
def get_docs_closest_to_centroids(data_length, n_clust, clust_labels, centroid_input, doc_vecs, all_reviews, ttls_with_labels):
    
    all_data = [ i for i in range(data_length) ]
    n_in_clust = [len(list(group)) for key, group in groupby(sorted(clust_labels))]

    num_clusters = n_clust
    m_clusters = clust_labels

    centers = centroid_input

    closest_data = []
    for i in range(num_clusters):
        center_vec = centers[i]
        data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

        one_cluster_tf_matrix = np.zeros( (  n_in_clust[i] , centers.shape[1] ) )
        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = doc_vecs[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        center_vec = np.expand_dims(center_vec, axis=0)

        closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
        
        closest_idx_in_one_cluster_tf_matrix = closest[0]
        
        closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
        data_id = all_data[closest_data_row_num]

        closest_data.append(data_id)

    closest_data_idxs = list(set(closest_data))

    for i in range(len(closest_data_idxs)):

        print('The Review Document most Representative of Grouping {} is: {} \n'.format(ttls_with_labels[closest_data_idxs[i]][1], 
                                                                         ttls_with_labels[closest_data_idxs[i]][0][0]))
        print('The Review Content is: \n')
        pretty_print(all_reviews[closest_data_idxs[i]][0], all_reviews[closest_data_idxs[i]][1])
        print('\n')

    return None
	
def kmeans_3D(input_d2v_model): 

    X = input_d2v_model.docvecs.vectors_docs

    #perform PCA with 3 Components
    pca = PCA(n_components=3)
    pca.fit(X)
    x_pca = pca.transform(X)

    print(np.amax(x_pca, axis=0))
    print(np.amin(x_pca, axis=0))

    np.random.seed(5)

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
            range=[-20, 20],
            title='PC_1',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        ),
        yaxis=dict(
            range=[-20, 20],
            title='PC_2',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        ),
        zaxis=dict(
            range=[-20,20],
            title='PC_3',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)',
            showticklabels=False, ticks=''
        )
    )

    centers = [[1, 1], [-1, -1], [1, -1]]
    X = x_pca

    estimators = {'KMeans': KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=111).fit(X)
                  }
    fignum = 1
    for name, est in estimators.items():
        est.fit(X)
        labels = est.labels_

        trace = go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                             showlegend=False,
                             mode='markers',
                             marker=dict(
                                    color=labels.astype(np.float),
                                    line=dict(color='black', width=1)
            ))
        fig.append_trace(trace, 1, fignum)

        fignum = fignum + 1

    fig['layout'].update(height=1000, width=800,
                         margin=dict(l=10,r=10))

    fig['layout']['scene'].update(scene)

    iplot(fig)
    
    return None
	
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(labels[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return None
	
def plot_spectral_embed_agglom_clusters(dist_matrix, n_clust):
    
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(dist_matrix)
    
    for linkage in ('ward', 'average', 'complete'):
        clustering = AgglomerativeClustering(affinity='euclidean',
                                    compute_full_tree='auto',
                                    connectivity=None,
                                    linkage=linkage,
                                    memory=None,
                                    n_clusters=n_clust,
                                    ).fit(dist_matrix)
        t0 = time()
        clustering.fit(X_red)
        print("%s :\t%.2fs" % (linkage, time() - t0))    
        
        plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
        
        print('Spectral Embedded Agglomerative Cluster Labels (Levels={} | {} Distance) are: \n\n {} \n'.format(n_clust,
                                                                                                           linkage,
                                                                                                           clustering.labels_))   
    return None
	
def dendro_static(dist_matrix, all_titles):

    dendro = dendrogram(
             dist_matrix, 
             labels=all_titles,
             no_plot=True,
             color_threshold=3.5,
             count_sort = "ascending"
             )

    icoord = np.array( dendro['icoord'] )
    dcoord = np.array( dendro['dcoord'] )
    color_list = np.array( dendro['color_list'] )

    plt.subplots(figsize=(35, 18))
    plt.yticks(fontsize = 20)
    plt.title("Hierachical Clustering - (Ward Distance)", fontsize=30)

    for xs, ys in zip(icoord, dcoord):
        color = plt.cm.Spectral( ys/8.0 )
        plt.plot(xs, ys, color)

    dendrogram(
             dist_matrix, 
             labels= all_titles,
             color_threshold= 3.5,
             count_sort = "ascending",
             #leaf_rotation= 85.,
             leaf_font_size= 20
             )
    
    return None
	
def dendro_interactive(dist_matrix, all_titles):

    fig = ff.create_dendrogram(dist_matrix, orientation='left', labels=all_titles)
    fig['layout'].update({'width':800, 'height':800})
    fig['layout'].update(font=dict(
            family='Old Standard TT, serif',
            size=9,
            color='black',
        ),  margin=dict(l=150,
            r=5,
            b=10,
            t=10,
            pad=4
    )          
                        )
    iplot(fig, filename='dendrogram_with_labels')
    
    return None
	
def dendro_heatmap(dist_matrix, all_titles):

    # Initialize figure by creating upper dendrogram
    figure = ff.create_dendrogram(dist_matrix, orientation='bottom', labels=all_titles)
    for i in range(len(figure['data'])):
        figure['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(dist_matrix, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    figure['data'].extend(dendro_side['data'])

    figure['layout']['yaxis']['ticktext'] = figure['layout']['xaxis']['ticktext']
    figure['layout']['yaxis']['tickvals'] = np.asarray(dendro_side['layout']['yaxis']['tickvals'])

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))
    data_dist = pdist(dist_matrix)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale = 'YIGnBu'
        )
    ]

    heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    figure['data'].extend(heatmap)

    # Edit Layout
    figure['layout'].update({'width':1200, 'height':800,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    figure['layout'].update(font=dict(
            family='Old Standard TT, serif',
            size=9,
            color='black',
        ),  margin=dict(l=5,
            r=5,
            b=150,
            t=5,
            pad=4
    ))

    # Edit xaxis
    figure['layout']['xaxis'].update({'domain': [.15, 1],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'tickangle':45,
                                      'ticks':""})
    # Edit xaxis2
    figure['layout'].update({'xaxis2': {'domain': [0, .15],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    # Edit yaxis
    figure['layout']['yaxis'].update({'domain': [0, .85],
                                      'mirror': False,
                                      'showgrid': False,
                                      'showline': False,
                                      'zeroline': False,
                                      'showticklabels': False,
                                      'ticks': ""})
    # Edit yaxis2
    figure['layout'].update({'yaxis2':{'domain':[.825, .975],
                                       'mirror': False,
                                       'showgrid': False,
                                       'showline': False,
                                       'zeroline': False,
                                       'showticklabels': False,
                                       'ticks':""}})

    # Plot!
    iplot(figure, filename='dendrogram_with_heatmap')
    
    return None
	
	
def main():
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	nlp = spacy.load('en')
	cont = Contractions('../GoogleNews-vectors-negative300.bin.gz')
	cont.load_models()

	#stopwords = spacy.lang.en.STOP_WORDS
	#spacy.lang.en.STOP_WORDS.add("e.g.")
	#nlp.vocab['the'].is_stop
	nlp.Defaults.stop_words |= {"(a)", "(b)", "(c)", "etc", "etc.", "etc.)", "w/e", "(e.g.", "no?", "s", 
							   "film", "movie","0","1","2","3","4","5","6","7","8","9","10","e","f","k","n","q",
								"de","oh","ones","miike","http","imdb", }
	stopwords = list(nlp.Defaults.stop_words)
	tokenizer = Tokenizer(nlp.vocab)
	punctuations = string.punctuation

	LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
	
	#####################################REVIEWS OF 100 BEST MOVIES on IMDB#########################################
	
	#open the base URL webpage
	html_page = urlopen("https://www.imdb.com/list/ls059633855/")

	#instantiate beautiful soup object of the html page
	soup = BeautifulSoup(html_page, 'lxml')

	review_text = get_movie_reviews(soup)
	
	all_good_movie_reviews = get_tagged_documents(review_text, cont, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	print('\n')
	d2v_model_best = doc2vec_similarity(all_good_movie_reviews, 'Best')
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~K-MEANS CLUSTERING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	all_rev_text = [rev for rev, ttl in all_good_movie_reviews]
	all_rev_ttls = [ttl[0] for rev, ttl in all_good_movie_reviews]
	
	numclust = 4
	kmeans_model = KMeans(n_clusters=numclust, init='k-means++', max_iter=300, random_state=111)  
	X = kmeans_model.fit(d2v_model_best.docvecs.vectors_docs)
	kmeans_clust_labels = kmeans_model.labels_.tolist()
	
	output_list = list(zip(kmeans_clust_labels, all_rev_ttls))
	print('The Groupings assigned by K-Means Clustering are: \n\n'.format(numclust))

	for i in sorted(output_list):
		print(i)
	print('\n')
		
	ttls_and_labels_kmeans_clust =[]
	i=0
	for rev, ttl in all_good_movie_reviews:
		ttls_and_labels_kmeans_clust.append((ttl, kmeans_clust_labels[i]))
		i +=1
		
	kmeans_clust_centroids = np.array(kmeans_model.cluster_centers_)
	
	get_docs_closest_to_centroids(data_length=100, n_clust=4, clust_labels=kmeans_clust_labels,
                                 centroid_input=kmeans_clust_centroids, doc_vecs=d2v_model_best.docvecs.vectors_docs,
                                 all_reviews=all_good_movie_reviews, ttls_with_labels=ttls_and_labels_kmeans_clust)
								 
	#pca = PCA(n_components=2).fit(d2v_model_best.docvecs.vectors_docs)
	#datapoint = pca.transform(d2v_model_best.docvecs.vectors_docs)
	
	#plt.figure
	#label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#ff0000"]
	#color = [label1[i] for i in kmeans_clust_labels]
	#plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

	#centroids = kmeans_model.cluster_centers_
	#centroidpoint = pca.transform(centroids)
	#plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
	#plt.show()
	
	#kmeans_3D(d2v_model_best)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~AGGLOMERATIVE HIERARCHICAL CLUSTERING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	tfidf_input = [' '.join(t) for t in all_rev_text]

	#define vectorizer parameters
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2,
									   use_idf=True, ngram_range=(1,3))

	tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_input)
	
	dist = 1 - cosine_similarity(tfidf_matrix)
	linkage_matrix = ward(dist)
	
	cl = linkage_matrix
	numclust = 5
	hier_clust_labels = fcluster(cl, numclust, criterion='maxclust')

	hier_clust_labels = hier_clust_labels - 1

	output_list = list(zip(hier_clust_labels, all_rev_ttls))
	print('The Levels assigned by a {}-Tiered Hierarchical Clustering are: \n\n'.format(numclust))

	for i in sorted(output_list):
		print(i)
	print('\n')
		
	ttls_and_labels_hier_clust =[]
	i=0
	for rev, ttl in all_good_movie_reviews:
		ttls_and_labels_hier_clust.append((ttl, hier_clust_labels[i]))
		i +=1
		
	hier_clust_codebook = []

	for i in range(hier_clust_labels.min(), hier_clust_labels.max()+1):
		hier_clust_codebook.append(d2v_model_best.docvecs.vectors_docs[hier_clust_labels == i].mean(0))

	hier_clust_centroids = np.vstack(hier_clust_codebook)
	
	get_docs_closest_to_centroids(data_length=100, n_clust=5, clust_labels=hier_clust_labels, centroid_input=hier_clust_centroids,
                              doc_vecs=d2v_model_best.docvecs.vectors_docs, all_reviews=all_good_movie_reviews, 
                              ttls_with_labels=ttls_and_labels_hier_clust)
							  
	#plot_spectral_embed_agglom_clusters(dist, 5)
	
	#dendro_static(linkage_matrix, all_rev_ttls)
	
	#dendro_interactive(dist, all_rev_ttls)
	
	#dendro_heatmap(dist, all_rev_ttls)
	
	#####################################REVIEWS OF 100 WORST MOVIES on IMDB#########################################
	
	#open the base URL webpage
	html_page_2 = urlopen("https://www.imdb.com/list/ls061324742/")

	#instantiate beautiful soup object of the html page
	soup_2 = BeautifulSoup(html_page_2, 'lxml')

	review_text_2 = get_movie_reviews(soup_2)
	
	all_bad_movie_reviews = get_tagged_documents(review_text_2, cont, stopwords, tokenizer, punctuations, LabeledSentence1)
	
	print('\n')
	d2v_model_worst = doc2vec_similarity(all_bad_movie_reviews, 'Worst')
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~K-MEANS CLUSTERING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	all_rev_text_bad = [rev for rev, ttl in all_bad_movie_reviews]
	all_rev_ttls_bad = [ttl[0] for rev, ttl in all_bad_movie_reviews]
	
	numclust = 4
	kmeans_model_worst = KMeans(n_clusters=numclust, init='k-means++', max_iter=300, random_state=111)  
	X_worst = kmeans_model_worst.fit(d2v_model_worst.docvecs.vectors_docs)
	kmeans_clust_labels_worst = kmeans_model_worst.labels_.tolist()
	
	output_list = list(zip(kmeans_clust_labels_worst, all_rev_ttls_bad))
	print('The Groupings assigned by K-Means Clustering are: \n\n'.format(numclust))

	for i in sorted(output_list):
		print(i)
	print('\n')
		
	ttls_and_labels_kmeans_clust_bad =[]
	i=0
	for rev, ttl in all_bad_movie_reviews:
		ttls_and_labels_kmeans_clust_bad.append((ttl, kmeans_clust_labels_worst[i]))
		i +=1
		
	kmeans_clust_centroids_worst = np.array(kmeans_model_worst.cluster_centers_)
	
	get_docs_closest_to_centroids(data_length=100, n_clust=4, clust_labels=kmeans_clust_labels_worst,
                                 centroid_input=kmeans_clust_centroids_worst, doc_vecs=d2v_model_worst.docvecs.vectors_docs,
                                 all_reviews=all_bad_movie_reviews, ttls_with_labels=ttls_and_labels_kmeans_clust_bad)
								 
	#pca_worst = PCA(n_components=2).fit(d2v_model_worst.docvecs.vectors_docs)
	#datapoint_worst = pca_worst.transform(d2v_model_worst.docvecs.vectors_docs)
	
	#plt.figure
	#label1 = ["#FFFF00", "#008000", "#0000FF", "#800080", "#ff0000"]
	#color = [label1[i] for i in kmeans_clust_labels_worst]
	#plt.scatter(datapoint_worst[:, 0], datapoint_worst[:, 1], c=color)

	#centroids = kmeans_model_worst.cluster_centers_
	#centroidpoint = pca_worst.transform(centroids)
	#plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
	#plt.show()
	
	#kmeans_3D(d2v_model_worst)
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~AGGLOMERATIVE HIERARCHICAL CLUSTERING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	tfidf_input_bad = [' '.join(t) for t in all_rev_text_bad]
	
	#define vectorizer parameters
	tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2,
									   use_idf=True, ngram_range=(1,3))

	tfidf_matrix_bad = tfidf_vectorizer.fit_transform(tfidf_input_bad)

	dist_bad = 1 - cosine_similarity(tfidf_matrix_bad)
	linkage_matrix_bad = ward(dist_bad)
	
	cl = linkage_matrix_bad
	numclust = 5
	hier_clust_labels_bad = fcluster(cl, numclust, criterion='maxclust')

	hier_clust_labels_bad = hier_clust_labels_bad - 1

	output_list = list(zip(hier_clust_labels_bad, all_rev_ttls_bad))
	print('The Levels assigned by a {}-Tiered Hierarchical Clustering are: \n\n'.format(numclust))

	for i in sorted(output_list):
		print(i)
	print('\n')
		
	ttls_and_labels_hier_clust_bad =[]
	i=0
	for rev, ttl in all_bad_movie_reviews:
		ttls_and_labels_hier_clust_bad.append((ttl, hier_clust_labels_bad[i]))
		i +=1
	
	hier_clust_codebook_bad = []

	for i in range(hier_clust_labels_bad.min(), hier_clust_labels_bad.max()+1):
		hier_clust_codebook_bad.append(d2v_model_worst.docvecs.vectors_docs[hier_clust_labels_bad == i].mean(0))

	hier_clust_centroids_bad = np.vstack(hier_clust_codebook_bad)
	
	get_docs_closest_to_centroids(data_length=100, n_clust=5, clust_labels=hier_clust_labels_bad, 
                              centroid_input=hier_clust_centroids_bad,
                              doc_vecs=d2v_model_worst.docvecs.vectors_docs, all_reviews=all_bad_movie_reviews, 
                              ttls_with_labels=ttls_and_labels_hier_clust_bad)
							  
	#plot_spectral_embed_agglom_clusters(dist_bad, n_clust=5)
	
	#dendro_static(linkage_matrix_bad, all_rev_ttls_bad)
	
	#dendro_interactive(dist_bad, all_rev_ttls_bad)
	
	#dendro_heatmap(dist_bad, all_rev_ttls_bad)
	
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
    print('Numpy', np.__version__)
    print('Beautiful Soup', bs4.__version__)
    print('Urllib', urllib.request.__version__) 
    print('Regex', re.__version__)
    print('Textacy', textacy.__version__)
    print('SpaCy', spacy.__version__)
    print('Gensim', gensim.__version__)
    print('Sklearn', sklearn.__version__)
    print('Scipy', scipy.__version__)
    #print('Matplotlib', matplotlib.__version__)
    #print('Plotly', plotly.__version__)
    #print('Cufflinks', cf.__version__)

    print('\n')
    print("~"*70)
    print('\n')

    main()

    print('\n')
    print('END OF PROGRAM')