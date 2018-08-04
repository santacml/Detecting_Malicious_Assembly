#!/usr/bin python3
from assemblyScanner import Scanner
import pickle
import random
import numpy
import time


class Vectorizor(object):
    def __init__(self, maxVal=1, maxLineLen=10):
        # works through multiple vectorizings -> multiple pgms 
        # self.values = {}
        # self.takenVals = []
        # self.maxVal=maxVal
        # self.maxLineLen = maxLineLen
        
        # self.vars = []     # don't need this anymore?
        self.bannedVars = []
        
        
        self.files = []
        self.maxFileLen = 0
        self.maxLineLen = 0
        # self.absoluteMaxFileLen = 50000
        
        # self.vocabSize = 5000  #idk, this will be an issue
        # self.takenNums = []
        
        self.currTokNum = 0
        self.tokToNum = {}
        # self.currTokNum, self.tokToNum = pickle.load(open("tokToNum.pkl", "rb"))
        self.currTokNum =  self.currTokNum + 1
        # print("loaded tokToNum")
        # print(self.currTokNum)
        
        # pruning less frequency reduces extraneous prunes
        # but it also takes waaaaaaay longer to do all the math
        self.pruneFrequency = 5
        self.numFilesScanned = 0
        
        self.answers = []
        
        # self.tokFreqs = [0,0]  # list of frequencies -> index is frequency!
        
        self.accumFreqs = numpy.asarray([0])
        self.availableNums = []
        
        
    def dumpVocab(self):
        pickle.dump([self.currTokNum, self.tokToNum], open("tokToNum.pkl", "wb"))
        
    def itemReplacer(self, testItem):
        item = testItem
        
        if len(testItem) > 1 :
            if testItem[-1] == "h":
                testItem = testItem[:-1]
            try:
                int(testItem, 16)
                # if we can convert it from hex, then say item is a hex num
                item = "hexnum"
            except ValueError:
                pass
        if item.startswith("sub_"):
            item = "sub"
        elif item.startswith("off_"):
            item = "off"
        elif item.startswith("unk_"):
            item = "unk"
        elif item.startswith("loc_"):
            item = "loc"
        elif item.startswith("dword_"):
            item = "dword"
        elif item.startswith("word_"):
            item = "word"
        elif item.startswith("locret_"):
            item = "locret"
        elif item.startswith("var_"):
            item = "var"
        elif item.startswith("byte_"):
            item = "byte"
        elif item.startswith("arg_"):
            item = "arg"
        elif item.startswith("nullsub_"):
            item = "nullsub"
        elif item.startswith("?"):
            item = "?thing"
        elif item.startswith("__"):
            item = "__thing"
        return item
        
    def findItem(self, item):
        result = self.tokToNum.get(item, 0)
        if not result:
            # print(item)
            if len(self.availableNums) > 0:
                result = self.availableNums.pop(0)
                
                self.tokToNum[item] = result
                
            else:
                self.currTokNum = self.currTokNum + 1
                self.tokToNum[item] = self.currTokNum
                
                # self.vars.append(item) # just for padding, idk
                
                result = self.currTokNum
            
            # self.tokFreqs.append(0)
        
        # self.tokFreqs[result] = self.tokFreqs[result] + 1
        return result
        
    def testHex(self, testItem):
        try:
            if len(testItem) != 2: return None
            return int(testItem, 16)
        except ValueError:
            return None
        
    def vectorize(self, fileName, answer):
        newLines = []
        with open(fileName, 'rb') as f:
            for line in f:
                line = line.lower()
                if line[0:6] != b".text:" and line[0:5] != b"code:" and line[0:4] != b"UPX1": # filter off only text
                    continue
                else:
                    # line = line[6:] # get rid of ".text"
                    line = line[14:] # get rid of ".text" and hex location
                
                try:
                    line = line.decode('ascii')
                except UnicodeDecodeError:
                    continue
                # print(line.split())
                line = line.split()
                
                
                newLine = []
                if len(line) == 0: continue
                # print(line)
                convert = self.testHex(line.pop(0))
                while len(line) > 0 and not convert == None: 
                    newLine.append(convert)
                    convert = self.testHex(line.pop(0))
                # print(newLine, line)
                if len(newLine) == 0: continue
                newLines.append(newLine)
                if len(newLine) > self.maxLineLen: self.maxLineLen = len(newLine)
        
        if len(newLines) > self.maxFileLen: self.maxFileLen = len(newLines) 
        
        if len(newLines) == 0:
            print("DID NOT FIND ANY MACHINE CODE.")
            return
        self.files.append(newLines)
        self.answers.append(answer) # maintains order with files
        self.numFilesScanned += 1
        print("num files scanned:", self.numFilesScanned)
        print("file length", len(newLines))
        print("max file len", self.maxFileLen)
        print("max line len", self.maxLineLen)
        # print("num vars",self.currTokNum) # doesn't make sense as we are just keeping hex!
        return [newLines, answer]
        

answers = {}
import csv
with open("trainLabels.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        name = row[0]
        answer = [int(row[1])]
        answers[name] = answer
        
    
filePrefix = r"D:\DAAACUMENTS\Research\2_Malware_Detection\kaggle_dataset" 
    
files = []
import glob, os, gzip
files.extend(glob.glob(filePrefix + r"\fullData\*.asm"))

# random.shuffle(files)
# THIS IS EITHER M OR R!!!!!!!!!!!!!!!!

#first, turn into tokens, then call vectorizor

# '''
f = None
vectorizor = Vectorizor()
numScanned = 0
for file in files:
    # if numScanned < 0: 
        # numScanned += 1
        # continue # us this to manual override to a count
        
    print(file)
    name = os.path.basename(file)[:-4] # get rid of .asm, better way, idc
    answer = answers[name]
    
    # lines = Scanner().scan(file)
    # vectorizor.vectorize(lines, answer)
    
    toPickle = vectorizor.vectorize(file, answer) # WILL HAVE [NEWLINES, ANSWER]
    # print(0/0)
    # print(vectorizor.files)
    
    # every 100 files start a new thing. Results in 100 folders? I guess.
    if numScanned % 500 == 0:
        if f: f.close()
        # f = open('tempAssemblyVectorized' + str(numScanned) + '.pklz', 'wb').truncate() # clear file
        f = gzip.open(filePrefix + r"\tempAssemblyVectorized\tempAssemblyVectorized" + str(numScanned) + '.pklz', 'w+b')  # this should clear
        # f.truncate()  # not in gzip
        # vectorizor.dumpVocab()
    
    if vectorizor.files:
        # pickle.dump(vectorizor.files[0], f) # just dump the one file we've scanned
        pickle.dump(toPickle, f) # just dump the one file we've scanned
    else:
        print("ERROR file:", file) # this will happen if there are no files :) good job past me
    
    vectorizor.files = []  # idk, whatever. ducktaaaaaaaaaaaape
    
    numScanned += 1
    
f.close()
    

print("FINAL LENGTHS:")
print("Num of files:", len(finalData))
print("Num of lines per file:", len(finalData[0]))
print("Vocab size", len(vectorizor.vars))
print("banned vars", len(vectorizor.bannedVars))


# '''

cutoffLen = 10000
def padFileLines(fileLines):
    maxLineLen = vectorizor.maxLineLen
    maxFileLen = cutoffLen
    emptyLine = [0] * maxLineLen
    for line in fileLines[0]:
        for n in range(0,maxLineLen-len(line)): line.append(0)
        # print(line)
        
    fileLines[0].extend([emptyLine] * (maxFileLen - len(fileLines[0]))) 
    
    return fileLines
    

# '''
fullAssemblyFileName = r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_dataset\tempAssemblyVectorized\assemblyVectorized10klines.pklz"
writeMe = gzip.open(fullAssemblyFileName, 'w+b')

validFileName = r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_dataset\tempAssemblyVectorized\validationAssemblyVectorized10klines.pklz"

fileCount = 0
for x in range(0,10500, 500):
    # I only want one copy of all these, keep it here
    zippedFile = r"D:\DAAACUMENTS\Research\2_Malware_Detection\kaggle_dataset\tempAssemblyVectorized\tempAssemblyVectorized" + str(x) + ".pklz"
    readMe = gzip.open(zippedFile, "r")
    while True:
        try:
            # grab 500 data samples, 100 validation
            if fileCount == 500:
                writeMe = gzip.open(validFileName, 'w+b')  # this should clear
            if fileCount == 600: 
                break
            
            fileLines = pickle.load(readMe)
            if len(fileLines[0]) > cutoffLen: 
                continue
            
            pickle.dump(padFileLines(fileLines), writeMe)
            fileCount += 1
            print("file:", fileCount, "len", len(fileLines[0]))
        except EOFError:
            print("starting next file", x)
            break

'''
# finalData = pickle.load(open("assemblyData.pkl", "rb"))

# shuffle everything, but maintain data to answer
shuffledData = []
for n in range(0, len(answers)):
    shuffledData.append([finalData[n],answers[n]])
    
random.shuffle(shuffledData)
random.shuffle(shuffledData)

    
# split datasets
# train = shuffledData[0:250]
# valid = shuffledData[250:275]
# test = shuffledData[275:300]

splitPoint = int((vectorizor.numFilesScanned*2)/3)
del vectorizor
del finalData
train = shuffledData[0:splitPoint]
valid = shuffledData[splitPoint:]
test = shuffledData[0:0]



# for each set, create data and answer vectors again
toPickle = []
for dataset in [train, valid, test]:
    data = []
    ans = []
    for n in range(0,len(dataset)):
        data.append(dataset[n][0])
        ans.append(dataset[n][1])
    
    toPickle.append([data,ans])

del shuffledData
del train
del valid
del test
del data
del dataset
# print(dir())
pickle.dump(tuple(toPickle), open("shuffledData.pkl", "wb"))
# '''


# newTrain, newValid, newTest = pickle.load(open("shuffledData.pkl", "rb"))

# trainx, trainy = newTrain
# print(trainx[0] == train[0][0])
# print(trainy[0] == train[0][1])

# print(len(trainx[1][1]))

# import numpy
# trainx = numpy.asarray(trainx)
# print(trainx.shape)

# assert(trainx[0] == trainx[1])









































