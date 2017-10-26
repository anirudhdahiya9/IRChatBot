import Stemmer
from Stemmer import Stemmer
import re
from collections import defaultdict

stoplist = defaultdict(int)
with open('stopwords.txt', 'r') as f:
    contents = f.read().splitlines()
for key in contents:
    stoplist[key]=1

stoplist["ref"]=1
stoplist["/ref"]=1
stoplist["references"]=1
stoplist["reflist"]=1

def removestopwords(data):
    processedlist = [key for key in data if stoplist[key]!=1]
    return processedlist

def tokenizedata(data):
    newdata = re.sub('[^a-zA-Z]',' ', data)
    tokenizedwords = newdata.split()
    return tokenizedwords

def stemmer(data):
    mystemmer = Stemmer("english")
    stemmedtext = [mystemmer.stemWord(key) for key in data]
    return stemmedtext

def processtext(textasstring):

    data = textasstring.lower()
    tokenizedlist = tokenizedata(data)
    stopwordsremoved = removestopwords(tokenizedlist)
    stemmedata = stemmer(stopwordsremoved)
    termf = defaultdict(int)
    for key in stemmedata:
        termf[key]+=1
    return termf

