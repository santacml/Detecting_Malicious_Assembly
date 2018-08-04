import pickle, gzip, glob, os

answerFile = gzip.open("tempAnswers.pklz", "rb")
answers = pickle.load(answerFile)
print(answers)