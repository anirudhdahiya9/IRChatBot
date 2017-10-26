import unicodedata
from collections import defaultdict
import sys
from fileprocessing import writeinfile
from fileprocessing import mergeallindexfiles
import json
from Textprocessing import processtext
from datetime import datetime


index = defaultdict(list)
dict_id = {}
countdoc = 0
countfile = 0
offset = 0
def createindex(question,answer):

        global index
        global dict_id
        global countdoc
        global countfile
        global offset
        vocabularylist = list(set(question.keys() + answer.keys()))

        for key in vocabularylist:
            list1 = ""
            list1 = list1 + str(countdoc) + " "
            list1 = list1 + str(question[key]) + " "
            list1 = list1 + str(answer[key])
            index[key].append(list1)

        print countdoc
        countdoc = countdoc+1;

        if(countdoc%5000==0):
            offset = writeinfile(sys.argv[2], index, dict_id, countfile, offset)
            countfile = countfile+1
            index = defaultdict(list)
            dict_id = {}

        return



if __name__ == "__main__":

    x = datetime.now()

    if len(sys.argv) != 3:
        print "Usage :: ./index.sh /path/to/sample.xml /folderWhereIndexisTobeCreated"
        sys.exit(0)

    with open('ehealthforumQAs.json', 'r') as data_file1:
        json_data1 = data_file1.read()
    data1 = json.loads(json_data1)

    for doc in data1:
        temp = []
        temp.append(doc["question"])
        temp.append(doc["answer"])
        dict_id[countdoc] = ' '.join(temp).encode('utf-8')
        createindex(processtext(doc["question"]), processtext(doc["answer"]))

    with open('healthtapQAs.json', 'r') as data_file2:
        json_data2 = data_file2.read()
    data2 = json.loads(json_data2)

    for doc in data2:
        temp = []
        temp.append(doc["question"])
        temp.append(doc["answer"])
        dict_id[countdoc] = ' '.join(temp).encode('utf-8')
        createindex(processtext(doc["question"]), processtext(doc["answer"]))

    with open('icliniqQAs.json', 'r') as data_file3:
        json_data3 = data_file3.read()

    data3 = json.loads(json_data3)

    for doc in data3:
        temp = []
        temp.append(doc["question"])
        temp.append(doc["answer"])
        dict_id[countdoc] = ' '.join(temp).encode('utf-8')
        createindex(processtext(doc["question"]), processtext(doc["answer"]))

    with open('questionDoctorQAs.json', 'r') as data_file4:
        json_data4 = data_file4.read()
    data4 = json.loads(json_data4)

    for doc in data4:
        temp = []
        temp.append(doc["question"])
        temp.append(doc["answer"])
        dict_id[countdoc] = ' '.join(temp).encode('utf-8')
        createindex(processtext(doc["question"]), processtext(doc["answer"]))

    with open('webmdQAs.json', 'r') as data_file5:
        json_data5 = data_file5.read()
    data5 = json.loads(json_data5)

    for doc in data5:
        temp = []
        temp.append(doc["question"])
        temp.append(doc["answer"])
        dict_id[countdoc] = ' '.join(temp).encode('utf-8')
        createindex(processtext(doc["question"]), processtext(doc["answer"]))

    with open(sys.argv[2] + 'totaldocs.txt', 'wb') as f:
        f.write(str(countdoc))

    offset = writeinfile(sys.argv[2], index, dict_id, countfile, offset)
    countfile = countfile+1

    mergeallindexfiles(sys.argv[2], countfile)

    titleOffset=[]

    with open(sys.argv[2]+'title.txt','rb') as f:
      titleOffset.append('0')
      prevoff = 0;
      for line in f:
        titleOffset.append(str(len(line)+prevoff))
        prevoff = len(line)+prevoff
    titleOffset = titleOffset[:-1]

    with open(sys.argv[2]+'titleoffset.txt','wb') as f:
      f.write('\n'.join(titleOffset))

    print "Time taken to create Index"
    print datetime.now()-x

