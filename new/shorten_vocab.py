from nltk import word_tokenize
import sys
import collections
import pandas as pd
import csv
import numpy as np
import pickle

#LOAD EMBEDDINGS
embedding_size = 50

words = pd.read_table('../glove.6B.'+str(embedding_size)+'d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
words_matrix = words.as_matrix()
print words_matrix.shape

#Add vectors for <UNK>, <EOS> and <PAD> and <go>
extra_words = ['<pad>', '<unk>', '<eos>', '<go>']
bla = np.random.rand(len(extra_words), embedding_size)
words_matrix = np.concatenate((bla, words_matrix), axis=0)


vocab =  extra_words + list(words.index)
vocab_size = len(vocab)

tups = zip(vocab, range(vocab_size))
dictionary = dict(tups)
rtups = zip(range(vocab_size), vocab)
rdictionary = dict(rtups)
print len(dictionary.keys())


with open('../sample.enc') as f:
    elines = f.read().decode('latin-1')
    elines = elines.split('\n')
with open('../sample.dec') as f:
    dlines = f.read().decode('latin-1')
    dlines = dlines.split('\n')

#Anything above 20 for enc or above 21 for dec is thrown away
i=0
nelines = []
ndlines = []
while i<len(elines):
    elines[i] = [tk.lower() for tk in word_tokenize(elines[i])]
    dlines[i] = ['<go>'] + [tk.lower() for tk in word_tokenize(dlines[i])]
    if len(elines[i])<21 and len(dlines[i])<22:
        nelines += [elines[i]]
        ndlines += [dlines[i]]
    i+=1

del elines, dlines
elines = nelines
dlines = ndlines

#CORP_VOCAB stores the corpus vocab (and the extra words)
corp_vocab = {'a'}
for i in range(len(elines)):
    corp_vocab.update(elines[i])
    corp_vocab.update(dlines[i])

corp_vocab.update(extra_words)

cwords = list(corp_vocab & set(vocab))

print 'common vocab:' , len(cwords)
print cwords[:10]
fwordsid = []
for tk in cwords:
    fwordsid.append(dictionary[tk])

fwordsid = sorted(fwordsid)

#final embeddings
fembeddings = words_matrix[fwordsid, :]
print fembeddings.shape

#final dictionaries
fdictionary = {}
frdictionary = {}
for i in fwordsid:
    fdictionary[rdictionary[i]] = fwordsid.index(i)
for i in fdictionary.keys():
    frdictionary[fdictionary[i]] = i


pickle.dump((fembeddings, fdictionary, frdictionary), open('embeddings.pkl', 'w'))
