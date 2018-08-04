
files = []
import pickle, glob, os, gzip
# files.extend(glob.glob(r"D:\DAAACUMENTS\Research\2_Malware_Detection\kaggle_dataset\compressed\*.pklz"))

# print(files)
# 0/0

'''
num files: 10732
max file len 480941
max line len 18

import pickle
objs = []
while 1:
    try:
        objs.append(pickle.load(f))
    except EOFError:
        break
'''

# cutoffLen = 500000
cutoffLen = 10000

def padFileLines(fileLines):
    maxLineLen = 18
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
# '''
    
    
    
    
    
    
    