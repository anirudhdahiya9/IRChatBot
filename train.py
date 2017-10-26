from nltk import word_tokenize
import sys
import collections

with open('../train.enc') as f:
    elines = f.read().decode('latin-1')
    elines = elines.split('\n')
with open('../train.dec') as f:
    dlines = f.read().decode('latin-1')
    dlines = dlines.split('\n')

print 'data loaded'
# In[19]:

def build_dataset(words, n_words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  #data = list()
  #unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      #unk_count += 1
    #data.append(index)
  #count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return dictionary, reversed_dictionary

#Anything above 7 for enc or above 8 for dec is thrown away
i=0
nelines = []
ndlines = []
while i<len(elines):
    elines[i] = word_tokenize(elines[i])
    dlines[i] = ['<GO>'] + word_tokenize(dlines[i])
    if len(elines[i])<8 and len(dlines[i])<9:
        nelines += [elines[i]]
        ndlines += [dlines[i]]
    i+=1

del elines, dlines
elines = nelines
dlines = ndlines

total = []
for subl in elines:
    total += subl

for subl in dlines:
    total+=subl

#print elines[:5]
#print dlines[:5]
    
#print total[:30]

dictionary, reversed_dictionary = build_dataset(total, 56000)
print dictionary.keys()[:10]
print len(dictionary.keys())

print 'dataset built'

# In[21]:

max_seq_len = 7
vocab_size = len(dictionary)

def tokToId(inp, dictionary):
    for iline, line in enumerate(inp):
        line = [dictionary[tk] for tk in line]
        inp[iline] = line
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
            line.append('<EOS>')
        for _ in range(mxlen - len(line) + 2):
            line.append('<PAD>')
        inp[iline] = line
    return inp

def epadSeq(inp, mxlen, markEOS):
    for iline, line in enumerate(inp):
        #if len(line)>=mxlen:
        #    line = line[:mxlen - 1]
        for _ in range(mxlen - len(line)):
            line.append('<PAD>')
        inp[iline] = line[::-1]
    return inp


def prep_data(enc, dec, dictionary):
    
    dictionary['<PAD>'] = vocab_size
    dictionary['<EOS>'] = vocab_size+1

    enc = epadSeq(enc, max_seq_len, False)
    dec = padSeq(dec, max_seq_len, True)
    
    enc = tokToId(enc, dictionary)
    dec = tokToId(dec, dictionary)
    
    #print enc[:10]
    #print dec[:10]
    
    return enc, dec

enc, dec = prep_data(elines, dlines, dictionary)

print 'data processed'

import tensorflow as tf #I know this needs to go up
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq
vocabulary_size = len(dictionary.keys())
embedding_size = 100
batch_size = 10
lsize = 40


# In[72]:
print 'making graph'
#Graph
#with tf.variable_scope("myrnn", reuse=None) as scope:

cell = tf.contrib.rnn.BasicLSTMCell(lsize)

inputs = tf.placeholder(tf.int32, shape=(None, max_seq_len))
labels = tf.placeholder(tf.int32, shape=(None, max_seq_len+2))

w_t = tf.get_variable("proj_w",[vocabulary_size, lsize], dtype=tf.float32)
w = tf.transpose(w_t)
b = tf.get_variable("proj_b",[vocabulary_size], dtype=tf.float32)
output_projection = (w, b)
#output_projection = None

inputs_series = tf.unstack(inputs, axis=1)
labels_series = tf.unstack(labels, axis=1)

#print w_t

outputs, states = embedding_rnn_seq2seq(
    inputs_series, inputs_series, cell,
    vocabulary_size,
    vocabulary_size,
    embedding_size, output_projection=output_projection,
    feed_previous=True)

#print outputs[0]

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

print 'graph ok'
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)
#with tf.variable_scope("myrnn") as scope:
for ep in range(40):
    #if i>0:
        #scope.reuse_variables()
    for i in range(len(enc)//batch_size):
        inp = enc[i:i+batch_size]
        label = dec[i:i+batch_size]
        #print inp
        #print label
        try:
            sess.run(train_step, {inputs: inp, labels: label})
        except:
            continue
    saver.save(sess, 'ckpt_'+str(ep)+'.tfmodel')
    print ep
