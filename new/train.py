from nltk import word_tokenize
import sys
import collections
import pandas as pd
import csv
import numpy as np
import pickle

max_seq_len = 20

#LOAD EMBEDDINGS
embeddings, dictionary, reversed_dictionary = pickle.load(open('embeddings.pkl'))
vocab_size = len(dictionary.keys())
embedding_size = embeddings.shape[1]

with open('../sample.enc') as f:
    elines = f.read().decode('latin-1')
    elines = elines.split('\n')
with open('../sample.dec') as f:
    dlines = f.read().decode('latin-1')
    dlines = dlines.split('\n')

print 'data loaded'

#TOKENIZE THE WORDS IN LINES
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

print elines[0]
print dlines[0]

print 'dataset built'


def tokToId(inp, dictionary):
    for iline, line in enumerate(inp):
        nline = []
        for tk in line:
            if tk in dictionary:
                nline += [dictionary[tk]]
            else:
                nline += [dictionary['<unk>']]
        inp[iline] = nline
    return inp

def padSeq(inp, mxlen, markEOS):
    for iline, line in enumerate(inp):
        #if len(line)>=mxlen:
        #    line = line[:mxlen - 1]
        #    if markEOS:
        #        line.append('<EOS>')
        #    else:
        #        pass
        #else:
        if markEOS:
            line.append('<eos>')
        for _ in range(mxlen - len(line) + 2):
            line.append('<pad>')
        inp[iline] = line
    return inp

def epadSeq(inp, mxlen, markEOS):
    for iline, line in enumerate(inp):
        #if len(line)>=mxlen:
        #    line = line[:mxlen - 1]
        for _ in range(mxlen - len(line)):
            line.append('<pad>')
        inp[iline] = line[::-1]
    return inp


def prep_data(enc, dec, dictionary):

    enc = epadSeq(enc, max_seq_len, False)
    dec = padSeq(dec, max_seq_len, True)

    print enc[0]
    print dec[0]
    
    enc = tokToId(enc, dictionary)
    dec = tokToId(dec, dictionary)
    
    return enc, dec

enc, dec = prep_data(elines, dlines, dictionary)

print 'data processed'
print len(enc[0])
print len(dec[0])

import tensorflow as tf #I know this needs to go up
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
import numpy as np
vocabulary_size = len(dictionary.keys())
batch_size = 10
lsize = 40

print 'making graph'
#Graph
#with tf.variable_scope("myrnn", reuse=None) as scope:

cell = tf.contrib.rnn.BasicLSTMCell(lsize)

inputs = tf.placeholder(tf.int32, shape=(None, max_seq_len))
labels = tf.placeholder(tf.int32, shape=(None, max_seq_len+1))
decoder_inputs = tf.placeholder(tf.int32, shape=(None, max_seq_len+1))

w_t = tf.get_variable("proj_w",[vocabulary_size, lsize], dtype=tf.float32)
w = tf.transpose(w_t)
b = tf.get_variable("proj_b",[vocabulary_size], dtype=tf.float32)
output_projection = (w, b)
#output_projection = None

inputs_series = tf.unstack(inputs, axis=1)
decoder_input_series = tf.unstack(decoder_inputs, axis=1)
labels_series = tf.unstack(labels, axis=1)

#print w_t

outputs, states = embedding_attention_seq2seq(
    inputs_series, decoder_input_series, cell,
    vocabulary_size,
    vocabulary_size,
    embedding_size, output_projection=output_projection,
    feed_previous=True)

for i in outputs:
    print i.name

def sampled_loss(labels, inputs):
    labels = tf.reshape(labels, [-1, 1])
    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_w_t = tf.cast(w_t, tf.float32)
    local_b = tf.cast(b, tf.float32)
    local_inputs = tf.cast(inputs, tf.float32)
    return tf.cast(
        tf.nn.sampled_softmax_loss(
            weights=local_w_t,
            biases=local_b,
            labels=labels,
            inputs=local_inputs,
            num_sampled=512,
            num_classes=vocabulary_size),
        tf.float32)

loss = tf.reduce_mean([tf.reduce_sum(sampled_loss(label, output)) for output, label in zip(outputs, labels_series)])
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
print decoder_inputs.name
print 'graph ok'

#for x in tf.get_default_graph().as_graph_def().node:
#    if x.name.endswith('embedding'):
#        print x

graph = tf.get_default_graph()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

#Embedding initialization
enc_emb = [v for v in tf.global_variables() if v.name == "embedding_attention_seq2seq/rnn/embedding_wrapper/embedding:0"][0]
dec_emb = [v for v in tf.global_variables() if v.name == "embedding_attention_seq2seq/embedding_attention_decoder/embedding:0"][0]
sess.run(enc_emb.assign(embeddings))
sess.run(dec_emb.assign(embeddings))

for ep in range(40):
    #if i>0:
        #scope.reuse_variables()
    for i in range(len(enc)//batch_size):
        inp = enc[i:i+batch_size]
        flabel = np.array(dec[i:i+batch_size])
        dec_inp = flabel[:, :-1]
        label = flabel[:, 1:]
        #print inp
        #print label
        try:
            sess.run(train_step, {inputs: inp, decoder_inputs: dec_inp, labels: label})
            print 'hoin'
        except:
            continue
    saver.save(sess, './freshckpts/ckpt_'+str(ep)+'.tfmodel')
    print ep
    exit()
