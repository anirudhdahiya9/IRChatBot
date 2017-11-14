# coding: utf-8
from nltk import word_tokenize
import sys
import collections
import pickle
import tensorflow as tf #I know this needs to go up
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq
import numpy as np

max_seq_len = 20

#functions to interface tokens and token ID's
def load_dicts(dictpath, rdictpath):
    dictionary = pickle.load(open(dictpath))
    rdict = pickle.load(open(rdictpath))
    return dictionary, rdict

def tokToId(inp, dictionary):
    for iline, line in enumerate(inp):
        line = [dictionary[tk] if tk in dictionary.keys() else dictionary['<unk>'] for tk in line]
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

def prep_data(enc, dictionary):
    print enc
    enc = epadSeq([enc], max_seq_len, False)
    print enc
    enc = tokToId(enc, dictionary)
    
    return enc

#Load dictionaries
#dictpath, rdictpath = ['dic.pkl', 'rdic.pkl']
_, dictionary, reversed_dictionary = pickle.load(open('embeddings.pkl'))

print 'data processed'

vocabulary_size = len(dictionary.keys())
#embedding_size = 100
#batch_size = 10
#lsize = 40


print 'making graph'

#Graph
sess = tf.Session()
saver = tf.train.import_meta_graph('./freshckpts/ckpt_0.tfmodel.meta')
saver.restore(sess,tf.train.latest_checkpoint('./freshckpts/'))

graph = tf.get_default_graph()
inputs = graph.get_tensor_by_name("Placeholder:0")
dec_inputs = graph.get_tensor_by_name("Placeholder_2:0")
#fetch tensor names for prediction step
tnames = ['embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_1/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_2/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_3/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_4/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_5/mul_2:0',
'embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/basic_lstm_cell_6/mul_2:0'] 

tnames = ['embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_1/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_2/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_3/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_4/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_5/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_6/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_7/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_8/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_9/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_10/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_11/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_12/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_13/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_14/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_15/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_16/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_17/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_18/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_19/AttnOutputProjection/BiasAdd:0',
'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/AttnOutputProjection_20/AttnOutputProjection/BiasAdd:0']

outputs = [graph.get_tensor_by_name(tname) for tname in tnames]

w = graph.get_tensor_by_name("transpose:0")
b = graph.get_tensor_by_name("proj_b:0")

predictions = [tf.argmax(tf.nn.softmax(tf.matmul(out, w) + b), 1) for out in outputs]

go_id = dictionary['<go>']
while True:
    getsent = [tk.lower() for tk in word_tokenize(raw_input('>>'))]
    inp = prep_data(getsent, dictionary)
    decoder_inputs = [[go_id] + [0]*max_seq_len]
    print decoder_inputs
    tkids = sess.run(predictions, {inputs: inp, dec_inputs : decoder_inputs})
    #for tkid in tkids:
    #    print tkid[0]
    
    sentence = []
    for tkid in tkids:
        try:
            sentence.append(reversed_dictionary[tkid[0]])
        except:
            continue
    outsent = []
    print sentence
    for tk in sentence:
        if tk not in ['<go>', '<eos>', '<pad>']:
            outsent.append(tk)
    print 'BOT : ', ' '.join(outsent)

