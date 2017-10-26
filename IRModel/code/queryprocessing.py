from Textprocessing import stemmer,tokenizedata,removestopwords
from collections import defaultdict
import sys
import bz2
import re
import math
from datetime import datetime


offset = []


def ranking(results, documentFreq, numberOfdocs):

    idf = defaultdict(float)
    listOfDocuments = defaultdict(float)

    for key in documentFreq:
        idf[key] = math.log(float(numberOfdocs)/float(documentFreq[key]))

    for word in results:
        fieldWisePostingList = results[word]
        for key in fieldWisePostingList:
            if len(key) > 0:
                postingList = fieldWisePostingList[key]
                if key == 'question':
                    factor = 10
                if key == 'answer':
                    factor = 10
                for i in range(0, len(postingList), 2):
                     listOfDocuments[postingList[i]] += float(float(postingList[i + 1])*float(factor)*float(idf[word]))

    return listOfDocuments


def findFileNumber(low, high, offset,pathOfFolder, word, f):
    while low <= high:
        mid = (low + high) / 2
        f.seek(offset[mid])
        testWord = f.readline().strip().split(' ')
        if word == testWord[0]:
            return testWord[1:], mid
        elif word > testWord[0]:
            low = mid + 1
        else:
            high = mid - 1
    return [], -1

def findtitle(low, high, offset,pathOfFolder, word, f):    #because titlefile has int to be compared
    while low <= high:
        mid = (low + high) / 2
        f.seek(offset[mid])
        testWord = f.readline().strip().split(' ')
        if int(word) == int(testWord[0]):
            return testWord[1:], mid
        elif int(word) > int(testWord[0]):
            low = mid + 1
        elif int(word) < int(testWord[0]):
            high = mid - 1
    return [], -1

def findposList(fileName, fileNumber, field, pathOfFolder, word, fieldFile):
    fieldOffset = []
    tempdf = []
    offsetFileName = pathOfFolder + 'offset' + str(field) + str(fileNumber) + '.txt'
    #print offsetFileName
    with open(offsetFileName, 'rb') as fieldOffsetFile:
        for line in fieldOffsetFile:
            offset, docfreq = line.strip().split(' ')
            fieldOffset.append(int(offset))
            tempdf.append(int(docfreq))
    fileList, mid = findFileNumber(0, len(fieldOffset), fieldOffset, pathOfFolder, word, fieldFile)
    return fileList, tempdf[mid]


def queryMultifield(queryWords, listOfFields, pathOfFolder, fVocabulary):
    postingList = defaultdict(dict)
    df = {}
    for i in range(len(queryWords)):
        word = queryWords[i]
        key = listOfFields[i]
        #print word , key
        #print fVocabulary
        returnedList, mid = findFileNumber(0, len(offset), offset, sys.argv[1], word, fVocabulary)

        if len(returnedList) > 0:
            fileNumber = returnedList[0]
            #print fileNumber
            fileName = pathOfFolder +  str(key) + str(fileNumber) + '.bz2'
            #print fileName
            fieldFile = bz2.BZ2File(fileName, 'rb')
            returnedList, docfreq = findposList(fileName, fileNumber, key, pathOfFolder, word, fieldFile)
            #print returnedList
            postingList[word][key] = returnedList
            df[word] = docfreq
    return postingList, df


def querySimple(queryWords, pathOfFolder, fVocabulary):
    postingList = defaultdict(dict)
    df = {}
    listOfField = ['question', 'answer']
    for word in queryWords:
        returnedList, _ = findFileNumber(0, len(offset), offset, sys.argv[1], word, fVocabulary)
        if len(returnedList) > 0:
            fileNumber = returnedList[0]
            df[word] = returnedList[1]
            for key in listOfField:
                fileName = pathOfFolder + str(key) + str(fileNumber) + '.bz2'
                fieldFile = bz2.BZ2File(fileName, 'rb')
                returnedList, _ = findposList(fileName, fileNumber, key, pathOfFolder, word, fieldFile)
                postingList[word][key] = returnedList
    return postingList, df


def main():
    if len(sys.argv) != 2:
         print "Usage :: python query.py pathOfFolder"
         sys.exit(0)

    with open(sys.argv[1] + 'offset.txt', 'rb') as f:
       for line in f:
         offset.append(int(line.strip()))

    '''titledict = {}
    with open(sys.argv[1] + 'title.txt', 'rb') as f:
       for line in f:
         temp = line.strip().split(' ')
         titledict[temp[0]]=' '.join(temp[1:])
    '''

    titleoffset = []
    with open(sys.argv[1] + 'titleoffset.txt', 'rb') as f:
       for line in f:
         titleoffset.append(int(line.strip()))

    #print titleoffset

    f = open(sys.argv[1] + 'totaldocs.txt', 'r')
    numberOfdocs = int(f.read().strip())
    f.close()

    stoplist = defaultdict(int)
    with open('stopwords.txt', 'r') as f:
        contents = f.read().splitlines()
    for key in contents:
        stoplist[key] = 1
    print "Cheers!! You can start Searching -->"

    while True:
        query = raw_input()
        #print query
        if(query=="quit"):
            break
        fVocabulary = open(sys.argv[1] + 'vocabularyList.txt', 'r')
        x1 = datetime.now()

        query = query.strip()
        query = query.lower()
        queryWords = tokenizedata(query)
        queryWords = removestopwords(queryWords)
        queryWords = stemmer(queryWords)
        results, documentFrequency = querySimple(queryWords, sys.argv[1], fVocabulary)


        #print results
        results = ranking(results, documentFrequency, numberOfdocs)

        titleFile = open(sys.argv[1] + 'title.txt', 'rb')
        dict_Title = {}

        for key in sorted(results.keys()):  # find top ten links
            title, _ = findtitle(0, len(titleoffset), titleoffset, sys.argv[1], key, titleFile)
            dict_Title[key] = ' '.join(title)

        #print results
        if len(results) > 0:
            results = sorted(results, key=results.get, reverse=True)
            #print results
            if len(results) > 10:
                results = results[:10]
                #print results
            for key in results:
                print dict_Title[key]
        else:
           print "Phrase Not Found"

        print "Time Taken to process  the query" , datetime.now()-x1

if __name__ == "__main__":
    main()


