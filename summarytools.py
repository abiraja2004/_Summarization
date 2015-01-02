# -*- coding: utf-8 -*-
"""Created on Wed Nov 26 15:11:30 2014

@author: JDD46
"""

###############################################################################
#
# The scripts that follow are far from production grade. This file contains
# basic implementations that were used by me for testing the effectiveness of
# summarization algorithms, and I hope that you can use it as a basis for 
# making a TF-IDF implementation in Scandium.
#
# Basic organization of file:
# I created 3 main object classes that store the information necessary for
# analysis. Of course, in the production implementation, all this data should
# be available through solr queries against a given project database, I leave
# it to the experts to properly make that implementation. The DataSet class 
# holds data about a dataset as a whole. The Post class holds data about a
# particular document within a given dataset. The Word class holds data about
# each unique word found within a dataset. Based on my understanding of how
# Scandium's implementaion in Riak and Solr works, DataSet objects basically
# correpond to the Riak project buckets, Post objects correspond to
# individual documents within a bucket (holding the data captured during
# ingestion), and Word objects contain data that can be obtained about words
# via solr queries.
#
# The methods in the DataSet class are what do the actual calculations. 
# extract_from() and extract_data() are just methods I created to get the data
# into a workable form. calc_scores(), build_svec_matrix(), and remove_rt are
# the methods that may actually be of interest to you guys.
#
###############################################################################

import csv
import string
import math
import numpy as np
import scipy.spatial.distance as dist
import nltk
import re
import cPickle as pickle
from nltk.corpus import stopwords
from operator import attrgetter as ag

stops = stopwords.words('english') + stopwords.words('spanish') + ['http', 
        'rt', 'n\'t', 'lol']

class DataSet(object):
    'Base class to store data representations for a particular dataset.'
    
    def __init__(self, setname='NO_NAME', pos_tag=0):
        self.setname = setname
        self.pos_tag = pos_tag
        self.word_dict = {}
        self.word_list = []
        self.wl_short = []
        self.posts = []
        self.retweets = []
        self.cosines = None
        self.co_occur_matrix = None
        self.reader = None
        self.has_data = False # Include checks to see if data has already been read.
        self.postcount = 0
        self.wordcount = 0
        
        if self.pos_tag != 0 and self.pos_tag != 1:
            print 'ERROR: Invalid input for POS tagging. 0 = No, 1 = Yes.'
        
    def save(self, fname=None):
        if fname is None:
            fname = self.setname
        fname = fname + '.pkl'
        with open(fname, 'wb') as f:
            pickler = pickle.Pickler(f, -1)
            pickler.dump(self)
    
    def extract_from(self, in_fname, col):
        """Returns a CSV reader object that reads from the file "out_fname".
        
        First creates reader object for file "in_fname", then creates a writer
        object that writes to file "out_fname". Only the data specified in 
        columnncol is written to the new file. Lastly, deletes the reader and 
        writer that were created by the method, creates a new reader for
        "out_fname", and returns that reader.
        """
        # Open passed file name.
        try:
            rawcsvfile = open(in_fname, 'rU')
        except IOError : 
            print 'Error: Not a valid file name. Try again.'
        else:
            out_fname = in_fname[:-4] + '_clean.csv'
            # Create CSV reader object to read from passed CSV file. 
            rawreader = csv.reader(rawcsvfile, dialect='excel')
            # Create write file and CSV writer object for it.
            with open(out_fname, 'wb') as f:
                writer = csv.writer(f, dialect='excel')
                # Write all entries contained in specified column to the file.
                for row in rawreader:
                    if len(row) < 2 or len(row[col]) > 250:
                        continue
                    # Remove non-ASCII characters from tweet before writing.
                    tweet = [filter(lambda x: x in string.printable, row[col])]
                    writer.writerow(tweet)
            rawcsvfile.close()
            del rawreader, writer
            f = open(out_fname, 'rU')
            self.reader = csv.reader(f, dialect='excel')
            
    def extract_data(self):
        """Extracts data from passed csv_reader object. Creates Post and Word 
        objects as they are observed.
        
        """
        if self.pos_tag == 0:
            # Read thru CSV, tokenize and keep only words, cast to lowercase.
            for row in self.reader:
                self.posts.append(Post(text=row[0]))
                clean = (filter(lambda x: x not in (string.punctuation + 
                        string.digits), row[0])).lower().strip().split()
                currentpost = self.posts[self.postcount]
                for word in clean:
                    if word not in self.word_dict:
                        self.word_dict[word] = Word(name=word)
                    currentpost.words_in[word] = currentpost.words_in.get(word,
                                                                         0) + 1
                currentpost.add(self.word_dict)
                currentpost.word_rank = [self.word_dict[word] for word in 
                                         currentpost.words_in]
                self.wordcount += currentpost.wordcount
                self.postcount += 1
        else:
             # Read through the CSV, tokenize and keep only words, cast to lowercase.
            for row in self.reader:
                self.posts.append(Post(text=row[0]))
                tokens = nltk.word_tokenize(row[0].lower().strip())
#                tokens = (filter(lambda x: x not in (string.punctuation + 
#                        string.digits), row[0])).lower().strip().split()
                tagged = nltk.pos_tag(tokens)
                clean = filter(lambda x: x[0][0] not in (
                        string.punctuation + string.digits), tagged)
                currentpost = self.posts[self.postcount]
                for word in clean:
                    if word not in self.word_dict:
                        self.word_dict[word] = Word(name=word[0], pos=word[1])
                    currentpost.words_in[word] = currentpost.words_in.get(word, 0) + 1
                currentpost.add(self.word_dict)
                currentpost.word_rank = [self.word_dict[word] for word in currentpost.words_in]
                self.wordcount += currentpost.wordcount
                self.postcount += 1
            
        # Create word_list, which is simply a list filled with the Word objects
        #   in word_dict.
        self.word_list = self.word_dict.values()
        self.reader = None
        self.save()
            
    def calc_scores(self):
        """Calculates tf-idf weights for words, scores posts.
        """
        # Go through each Word object in word_dict, calculating the tf-idf
        #   score for each from the tf and idf values stored in each object,
        #   and storing the tf-idf scores in each object.
        
        # Calculate the tf-idf weight of each word.
        for word in self.word_list:
            word.calctfidf(self.postcount, self.wordcount)
        # Calculate the tf-idf score of each sentence.
        for post in self.posts:
            post.calcscore(self.word_dict)
            post.word_rank.sort(key=lambda x: x.tfidf, reverse=True)  
        # Sort the post and word lists, descending by tf-idf score.
        self.posts.sort(key=lambda x: x.score, reverse=True)
        self.word_list.sort(key=lambda x: x.tfidf, reverse=True)
        max_score = self.posts[0].score
        # Apply an index to each word and post (this is important for the
        # retweet removal process.)
        for idx, word in enumerate(self.word_list):
            word.index = idx
        for idx, post in enumerate(self.posts):
            post.score /= max_score
            post.index = idx
        self.save()
        
    def build_svec_matrix(self):
        """Calculates the pairwise cosine similarity of each sentence, using
        the sentence vector representation.
        
        This is done to facilitate retweet removal. This function uses a Numpy
        array in which the columns represent each unique word in the data set
        and the rows represent each post observed. The values in the matrix
        correspond to the number of occurrences of a word in each post. In
        other words, the array is simply a matrix that is composed of vector 
        representations of each observed sentence, with words serving as 
        vector dimensions.
        
        Take, for example, the sentence, "I took the dog to the park." For this
        example, we'll say that the sentence is the fourth in the data set
        (index = 3) and the indexes corresponding to the words in the
        sentence are:
        I : 2
        took : 14
        the : 0
        dog : 25
        to : 5
        park : 31
        Since the index of this sentence is 3, the vector representation of 
        this sentence is simply the fourth row in the Numpy array, with the
        first three rows representing the sentences with indexes 0-2. That
        means in the fourth row of the array, we will find a 1 in column
        indexes 2, 14, 25, 5, and 31 (since all of those words occurred once in
        the sentence) and a 2 in column index 0 (since 'the' occurred twice).
        Every other value in the row will be 0.
        
        This representation then allows the use of Numpy's pdist function that
        calculates the pairwise cosine similairty between each sentence in the
        data set. Cosine similarity is a value between 0 and 1, where the more
        similar two sentences are (based on the words they contain), the higher
        the similarity value. The output of the calculation is a new square 
        matrix where both the rows and columns represent the sentences in the 
        data set and the values are similarity score between each pair of
        sentences. This approach is able to detect tweets that are likely
        retweets (contain a very high percentage of the same words as another 
        tweet) but remains flexible if some other text is included in the post
        along with the retweeted text, as is often the case. I've found that 
        methods that rely on cues, like 'RT' or string sequences aren't as
        reliable. Additionally, Numpy is actually written in C, so it is much
        more memory- and speed-efficient than anything you could make in
        Python. As you can imagine, the biggest problem that I didn't have time
        to fix is that once the dataset becomes big enough, the matrix becomes
        too large to deal with in memory. I would be really interested if you
        guys were able to come up with a function that takes this general
        approach to automated retweet removal, but is scalable. Perhaps Spark
        can handle a problem like this without too much actual adjustment...
        
        """
        # Numpy nd arrays are most efficient when the size of the array is
        # static, so I prescripted the size.
        svec_matrix = np.zeros((len(self.posts), len(self.word_dict)), 
                                    dtype=float)
        # This is where the values are entered into the array. Note that this
        # function is only meant to be run after calc_scores(), in which the 
        # Post and Word lists are sorted. Those lists need to be sorted and the
        # objects given index values in order for this function to work
        # properly.
        for post in self.posts:
            for word in post.words_in:
                svec_matrix[post.index, self.word_dict[word].index] = \
                post.words_in[word]
        # Here, we use pdist to calculate the similarities between each pair
        # of sentences. It's a huge amount of computation for large data sets, 
        # but it runs pretty quickly... Not instant by any means, though.
        self.cosines = \
                np.triu(dist.squareform(1 - dist.pdist(svec_matrix, 'cosine')))
    
        self.save()        
    
    def remove_rt(self):
        """Removes retweets from DataSet.
        
        Also adjusts Word and Post objects accordingly, so that the cleaned up
        set of tweets can be rescored.
        
        This function holds on to one copy of a retweeted tweet (the highest-
        scored one... This relies on the fact that the posts and words were 
        sorted before creating the Numpy cosine similarity array.)
        """
        # Find where we likely have retweets (similarity score above a high
        # threshold). The coordinates of these high values correpond to 
        # sentence indexes.
        coords = \
        zip(np.asarray(np.where(self.cosines >= 0.85)[0], dtype=int).tolist(), 
            np.asarray(np.where(self.cosines >= 0.85)[1], dtype=int).tolist())
        # This step compiles a list of the tweets that need to be removed,
        # making sure that one copy of each tweet is not marked as a retweet
        # and those that are marked are only marked once.
        to_rt = {}           
        for coord in coords:
            if coord[1] not in to_rt:
                to_rt[coord[1]] = None
                self.posts[coord[0]].retweets += 1
        to_rt = to_rt.keys()
        to_rt.sort(reverse=True)
        # Now actually mark the retweets. The remove method actually marks the 
        # post and takes care of the housekeeping.
        for i in to_rt:
            self.posts[i].remove(self.word_dict)
            self.postcount -= 1
            self.wordcount -= self.posts[i].wordcount
            self.retweets.append(self.posts.pop(i))               
        self.calc_scores()
        self.build_svec_matrix()
        
class Word(object):
    'Base class to store word data relevant to summarization.'
    
    def __init__(self, name='', pos='-NONE-'):
        self.name = name
        self.pos = pos
        self.is_stopword = False
        self.index = -1
        self.co_occur = {}
        self.tf = 0
        self.idf = 0
        self.tfidf = 0
        
        # If it's a stopword, we're not even going to consider it in
        # calculations.        
        if self.name in stops:
            self.is_stopword = True
            self.tf, self.idf, self.tfidf = 0, 0, 0
            
    def __repr__(self):
        return 'Word ' + str((self.name, self.pos))
            
    def calctfidf(self, postcount, wordcount):
        """Calculates 
        """
        if self.idf != 0:    
            self.tfidf = (float(self.tf) / float(wordcount)) * math.log(
            (float(postcount) / float(self.idf)), 2)
        else:
            self.tfidf = 0
        
class Post(object):
    'Base class to store Twitter post data relevant to summarization.'
    
    def __init__(self, text=''):
        self.text = text
        self.index = -1
        self.score = 0
        self.wordcount = 0
        self.retweets = 0
        self.words_in = {}
        self.word_rank = []
        self.is_junk = True
        
    def add(self, word_dict):
        """This makes sure that the data for the calling Post object and for
        all of the words it contains is properly adjusted when a Post is
        unmarked as junk.
        """
        if self.is_junk:
            self.is_junk = False
            for word in self.words_in:
                if not word_dict[word].is_stopword:
                    word_dict[word].tf += self.words_in[word]
                    self.wordcount += self.words_in[word]
                    word_dict[word].idf += 1
                    for compare in self.words_in:
                        if not word_dict[compare].is_stopword:
                            word_dict[word].co_occur[compare] = word_dict[word
                            ].co_occur.get(compare, 0) + 1
    
    def remove(self, word_dict):
        """This makes sure that the data for the calling Post object and for
        all of the words it contains is properly adjusted when a Post is
        marked as junk.
        """
        if not self.is_junk:
            self.is_junk = True
            self.wordcount = 0
            for word in self.words_in:
                if not word_dict[word].is_stopword:
                    word_dict[word].tf -= self.words_in[word]
                    word_dict[word].idf -= 1
                    for compare in self.words_in:
                        if not word_dict[compare].is_stopword:
                            word_dict[word].co_occur[compare] -= 1
            self.index = -1
            self.score = 0
                
    def calcscore(self, word_dict):
        """Calculated the tf-idf score of the calling post object.
        """
        weightsum = 0
        for item in self.words_in:
            weightsum += word_dict[item].tfidf * self.words_in[item]
        self.score = weightsum / max(7, self.wordcount)
        
class Summary(object):
    'Base class that holds important summary data.'
    def __init__(self, dataset):
        self.dataset = dataset
        self.summary = []
        
    def select(self, num=10, threshold=0.77):
        selected = [0]
        sel_array = np.zeros(num, len(self.dataset.posts))
        sel_array[0, :] += self.dataset.cosines[0, :]
        while len(selected) < num:
            
            for index in sel_ind:
                
    
        
def open_dataset(fname):
    with open(fname, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        return unpickler.load()