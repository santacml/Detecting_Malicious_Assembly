#!/usr/bin python3
from assemblyScanner import Scanner
import pickle
import gzip
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
        
        self.vars = []
        self.bannedVars = []
        
        
        self.files = []
        self.maxFileLen = 0
        self.maxLineLen = 0
        self.absoluteMaxFileLen = 50000
        
        # self.vocabSize = 5000  #idk, this will be an issue
        # self.takenNums = []
        
        self.currTokNum = 1
        self.tokToNum = {}
        # self.currTokNum, self.tokToNum = pickle.load(gzip.open("tokToNum.pklz", "rb"))
        # self.currTokNum =  self.currTokNum + 1
        # print("loaded tokToNum")
        # print(self.currTokNum)
        
        # pruning less frequency reduces extraneous prunes
        # but it also takes waaaaaaay longer to do all the math
        self.pruneFrequency = 5
        self.numFilesScanned = 0
        
        self.answers = []
        
        
        
    def dumpVocab(self):
        pickle.dump([self.currTokNum, self.tokToNum], gzip.open("tokToNum.pklz", "wb"))
        
    def vectorize(self, fileLines, answer):
        self.answers.append(answer)
        temp_newvars = []
        newLines = []
        
        if len(fileLines) > self.maxFileLen: self.maxFileLen = len(fileLines) 
        
        
        
        for line in fileLines:
            newLine = []
            if len(line) > self.maxLineLen: self.maxLineLen = len(line)
            for tempItem in line:
                if tempItem == ",": continue
                
                item = tempItem
                # if len(tempItem) == 8 and tempItem[1] == "x" and tempItem[0] == "0":
                # if len(tempItem) > 1 and tempItem[1] == "x" and tempItem[0] == "0":
                if len(tempItem) > 1:
                    try:
                        int(tempItem, 16)
                        # if we can convert it from hex, then say item is a hex num
                        item = "hexnum"
                    except ValueError:
                        pass
                    # print(tempItem)
                    # continue
                
                result = self.tokToNum.get(item, 0)
                
                if not result:
                    # print(item)
                    self.currTokNum = self.currTokNum + 1
                    self.tokToNum[item] = self.currTokNum
                    result = self.currTokNum
                    
                    self.vars.append(item) # just for padding, idk
                    temp_newvars.append(item)
                
                newLine.append(result)
            
            # print(newLine)
            newLines.append(newLine)
            # print(line)
            # print(newLine)
            
                
            
        # print("new vars", temp_newvars)
        self.files.append(newLines)
        
            
            
        # pruning - what if I just turn this off
        self.numFilesScanned += 1
        # if self.numFilesScanned % self.pruneFrequency == 0:
            # self.prune()
        print("num files scanned:", self.numFilesScanned)
        print("file len",  len(newLines))
        # print("vars", len(self.vars))
        print("num vars",self.currTokNum)
        
        return (newLines, answer)
        
        
    def padAll(self):
        print("number of files:", len(self.files))
        print("max line len:", self.maxLineLen)
        print("max file len:", self.maxFileLen)
        
        while (self.maxFileLen % 500 != 0):
            self.maxFileLen = self.maxFileLen + 1
            
        print("new max file len:", self.maxFileLen)
        emptyLine = [0] * self.maxLineLen
        for fileLines in self.files:
            for line in fileLines:
                for n in range(0,self.maxLineLen-len(line)): line.append(0)
                
            fileLines.extend([emptyLine] * (self.maxFileLen - len(fileLines))) 
            
        
        return self.files
    
files = []
import glob
files.extend(glob.glob(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_dataset\ASM_MALWARE\*.asm"))
regware = glob.glob(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_dataset\ASM_REGWARE\*.asm")
files.extend(regware) 


random.shuffle(files)

tempFileName = "tempAssemblyVectorized.pklz"

# '''

f = gzip.open(tempFileName, "wb")
vectorizor = Vectorizor()
for file in files:
    print("vectorizing", file)
    answer = -1
    # if file[57] == "M": 
        # answer = [0]
    # else:
        # answer = [1]
    if "ASM_MALWARE" in file:
        answer = [0]
    else:
        answer = [1]
    
    lines = Scanner(file).scan()
    
    toPickle = vectorizor.vectorize(lines, answer)  # contains   newLines, answer
    
    
    if vectorizor.files:
        # pickle.dump(vectorizor.files[0], f) # just dump the one file we've scanned
        pickle.dump(toPickle, f) # just dump the one file we've scanned
    else:
        print("ERROR file:", file) # this will happen if there are no files :) good job past me
    
    vectorizor.files = []  # idk, whatever. ducktaaaaaaaaaaaape
    

f.close()
# pad everything to be same row and col size
# finalData = vectorizor.padAll()



print("FINAL LENGTHS:")
print("Num of files:", vectorizor.numFilesScanned)
print("Num of lines per file:", vectorizor.maxFileLen)
print("Num of toks per line:", vectorizor.maxLineLen)
print("Vocab size", len(vectorizor.vars))
# print("banned vars", len(vectorizor.bannedVars))
# '''

def padFileLines(fileLines):
    maxLineLen = vectorizor.maxLineLen
    maxFileLen = vectorizor.maxFileLen
    emptyLine = [0] * maxLineLen
    for line in fileLines[0]:
        for n in range(0,maxLineLen-len(line)): line.append(0)
        # print(line)
        
    fileLines[0].extend([emptyLine] * (maxFileLen - len(fileLines[0]))) 
    
    return fileLines
    

# '''
fullAssemblyFileName = "assemblyVectorized.pklz"
writeMe = gzip.open(fullAssemblyFileName, 'w+b')

validFileName = "validationAssemblyVectorized.pklz"

readMe = gzip.open(tempFileName, "rb")

fileCount = 0
while True:
    try:
        # grab 500 data samples, 100 validation
        splitPoint = int((vectorizor.numFilesScanned * 2)/3)
        if fileCount == splitPoint:
            writeMe = gzip.open(validFileName, 'w+b')  # this should clear
        
        fileLines = pickle.load(readMe)
        
        pickle.dump(padFileLines(fileLines), writeMe)
        fileCount += 1
        print("file:", fileCount, "len", len(fileLines[0]))
    except EOFError:
        break






# '''


'''
# finalData = pickle.load(gzip.open("assemblyData.pklz", "rb"))

# shuffle everything, but maintain data to answer
shuffledData = []
for n in range(0, len(answers)):
    shuffledData.append([finalData[n],answers[n]])
    
random.shuffle(shuffledData)
random.shuffle(shuffledData)


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
pickle.dump(tuple(toPickle), gzip.open("shuffledData.pklz", "wb"))
# '''


# newTrain, newValid, newTest = pickle.load(gzip.open("shuffledData.pklz", "rb"))

# trainx, trainy = newTrain
# print(trainx[0] == train[0][0])
# print(trainy[0] == train[0][1])

# print(len(trainx[1][1]))

# import numpy
# trainx = numpy.asarray(trainx)
# print(trainx.shape)

# assert(trainx[0] == trainx[1])









































