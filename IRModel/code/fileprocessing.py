import sys
import bz2
import heapq
import os
import operator
from collections import defaultdict
import threading


class writeParallel(threading.Thread):

    def __init__(self, field, data, offset, countFinalFile, pathOfFolder):
        threading.Thread.__init__(self)
        self.data = data
        self.field = field
        self.count = countFinalFile
        self.offset = offset
        self.pathOfFolder = pathOfFolder

    def run(self):
        filename = self.pathOfFolder + self.field + str(self.count)
        with bz2.BZ2File(filename + '.bz2', 'wb', compresslevel=7) as f:
            f.write('\n'.join(self.data))
            #f.write('\n')

        filename = self.pathOfFolder + 'offset' + self.field + str(self.count) + '.txt'
        with open(filename, 'wb') as f:
            f.write('\n'.join(self.offset))
            #f.write('\n')

def writeFinalIndex(data, countFinalFile, pathOfFolder, offsetSize):
    question = defaultdict(dict)
    answer = defaultdict(dict)
    uniqueWords = []
    offset = []

    for key in sorted(data.keys()):
        listOfDoc = data[key]
        temp = []
        flag = 0
        for i in range(0, len(listOfDoc), 3):
            word = listOfDoc
            docid = word[i]
            if word[i + 1] != '0':
                question[key][docid] = int(word[i + 1])
                flag = 1
            if word[i + 2] != '0':
                answer[key][docid] = int(word[i + 2])
                flag = 1

        if flag == 1:
            string = key + ' ' + str(countFinalFile) + ' ' + str(len(listOfDoc) / 3)
            uniqueWords.append(string)
            offset.append(str(offsetSize))
            offsetSize = offsetSize + len(string) + 1

    questionData = []
    answerData = []


    questionOffset = []
    answerOffset = []


    previousquestion = 0
    previousanswer = 0

    for key in sorted(data.keys()):
        # print key
        if key in question:
            string = key + ' '
            sortedField = question[key]
            sortedField = sorted(sortedField, key=sortedField.get, reverse=True)     #sorted by highesttermfrequency in a doc
            for doc in sortedField:
                string += doc + ' ' + str(question[key][doc]) + ' '
            questionOffset.append(str(previousquestion) + ' ' + str(len(sortedField)))
            previousquestion += len(string) + 1
            questionData.append(string)

        if key in answer:
            string = key + ' '
            sortedField = answer[key]
            sortedField = sorted(sortedField, key=sortedField.get, reverse=True)
            for doc in sortedField:
                string += doc + ' ' + str(answer[key][doc]) + ' '
            answerOffset.append(str(previousanswer) + ' ' + str(len(sortedField)))
            previousanswer += len(string) + 1
            answerData.append(string)


    thread1 = writeParallel('question', questionData, questionOffset, countFinalFile, pathOfFolder)
    thread2 = writeParallel('answer', answerData, answerOffset, countFinalFile, pathOfFolder)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    with open(pathOfFolder + "vocabularyList.txt", "ab") as f:
        f.write('\n'.join(uniqueWords))
        f.write('\n')

    with open(pathOfFolder + "offset.txt", "ab") as f:
        f.write('\n'.join(offset))
        f.write('\n')

    return countFinalFile, offsetSize


def mergeallindexfiles(folderpath, countfile):
        listOfWords = {}
        indexFile = {}
        topOfFile = {}
        flag = [0] * countfile
        data = defaultdict(list)
        heap = []
        countFinalFile = 0
        offsetSize = 0
        for i in xrange(countfile):
            fileName = folderpath + 'index' + str(i) + '.txt.bz2'
            indexFile[i] = bz2.BZ2File(fileName, 'rb')
            flag[i] = 1
            topOfFile[i] = indexFile[i].readline().strip()
            listOfWords[i] = topOfFile[i].split(' ')
            if listOfWords[i][0] not in heap:
                heapq.heappush(heap, listOfWords[i][0])

        count = 0
        while any(flag) == 1:
            temp = heapq.heappop(heap)
            count += 1
            for i in xrange(countfile):
                if flag[i]:
                    if listOfWords[i][0] == temp:
                        data[temp].extend(listOfWords[i][1:])

                        topOfFile[i] = indexFile[i].readline().strip()

                        if topOfFile[i] == '':
                            flag[i] = 0
                            indexFile[i].close()
                            os.remove(folderpath + 'index' + str(i) + '.txt.bz2')
                        else:
                            listOfWords[i] = topOfFile[i].split(' ')
                            if listOfWords[i][0] not in heap:
                                heapq.heappush(heap, listOfWords[i][0])

            if(len(data)>=10000):
                countFinalFile, offsetSize = writeFinalIndex(data, countFinalFile, folderpath, offsetSize)
                countFinalFile+=1
                data = defaultdict(list)

        countFinalFile, offsetSize = writeFinalIndex(data, countFinalFile, folderpath, offsetSize)
        countFinalFile += 1
        print countFinalFile


def writeinfile(filepath, index, dict_id, countfile, titleoffset):
        data = []
        previousTitleOffset = titleoffset

        for key in sorted(index):
            string = str(key) + ' '
            temp = index[key]
            string += ' '.join(temp)
            data.append(string)

        filename = filepath + 'index' + str(countfile) + '.txt.bz2'  # compress and write into file
        with bz2.BZ2File(filename, 'wb', compresslevel=9) as f:
            f.write('\n'.join(data))
            #f.write('\n')

        data = []
        dataOffset = []
        for key in sorted(dict_id.keys()):
            data.append(str(key) + ' ' + dict_id[key])
            dataOffset.append(str(previousTitleOffset))
            previousTitleOffset += len(str(key) + ' ' + dict_id[key])

        filename = filepath + 'title.txt'
        with open(filename, 'ab') as f:
            f.write('\n'.join(data))
            f.write('\n')
        return previousTitleOffset



