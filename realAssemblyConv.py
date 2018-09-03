import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # gets rid of AVX message
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'  # log tensor allcoation?

# I have a GPU so idc

# kills GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import numpy
import random
import pickle
import gzip
import keras
from keras import optimizers
from keras import backend as K
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing import sequence
from keras.preprocessing.sequence import TimeseriesGenerator

import tensorflow as tf

vocab_size = 903
batch_size = 1
input_length = 107071
line_length = 25
num_samples = 160
num_samples_valid = 80

ANSWERS = []

def loadDataGeneratorBatches(assemblyVectorizedFile):
    readMe = open(assemblyVectorizedFile, "rb")
    while True:
        # for IDK in range(0, num_samples):  # if this works regardless of who calls, then idk I think this works
        try:
            xArr = []
            ans = []
            for x in range(0,batch_size):
                fileArr = pickle.load(readMe)
                xArr.append(fileArr[0])
                ans.append(fileArr[1])
                # ANSWERS.append(fileArr[1])    # use this for confusion matrix ( need answers from generator)
                
            
            train_x = numpy.asarray(xArr, dtype="float32") 
            
                # print("ANSWER IS", num)
            train_y = numpy.asarray(ans, dtype="float32")
            
            yield (train_x, train_y)
            
        except EOFError:
            del readMe
            readMe = open(assemblyVectorizedFile, "rb")
            

# import tensorflow as tf
# config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

model = Sequential()

model.add(MaxPooling1D(10, input_shape=(input_length,line_length)))
model.add(TimeDistributed(Embedding(vocab_size,100)))
model.add(Conv2D(200, 3, activation='tanh'))
model.add(MaxPooling2D(3))
model.add(Conv2D(200, 3, activation='tanh'))
model.add(MaxPooling2D(3))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))






print("Compiling Model and training")
print()

'''
model = None
# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
# '''



'''
from keras.models import load_model
model = load_model(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_networks\21.MaxEmb ConvNet 240 Files\Conv-37.hdf5")
# adamTest = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
# sgd = optimizers.SGD(lr=.001, momentum=0.9, nesterov=True)
# model.compile(optimizer='sgd',
              # loss='binary_crossentropy',
              # metrics=['accuracy'])
print(model.summary())
# '''

'''
# pred = model.predict(valid_x, batch_size=batch_size)

valid_gen = loadDataGeneratorBatches(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_dataset\assemblyVectorized.pkl")
pred = model.predict_generator(valid_gen, 
                    steps = int(num_samples/batch_size))
pred = numpy.rint(pred)

valid_y = ANSWERS[:num_samples]

TP = 0
FP = 0
TN = 0
FN = 0
for x in range(0, pred.shape[0]):
    val = valid_y[x][0]
    predVal = pred[x][0]
    
    if val == predVal:
        if val == 0: # 0 is malware!!
            TP = TP + 1
        else:
            TN = TN + 1
    else:
        if val == 0:
            FN = FN + 1
        else:
            FP = FP + 1

print("TP", TP)
print("FP", FP)
print("TN", TN)
print("FN", FN)
print(0/0)
# '''
    
# print(pred)
# print(numpy.rint(pred))
# print(numpy.rint(pred).shape)
# print(valid_y)

# '''

# import tensorflow as tf
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())


csv_logger = CSVLogger(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_networks\trainingSeqConv.log")
filepath = r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_networks\Conv-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath)


# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = loadData()
# model.fit(train_x, train_y, 
          # epochs=50, 
          # batch_size=batch_size, 
          # callbacks=[csv_logger, checkpoint],
          # validation_data=(valid_x, valid_y))
          
# '''
# train_gen = loadDataGeneratorTrainSeqSep()
# valid_gen = loadDataGeneratorValidSeqSep()

train_gen = loadDataGeneratorBatches(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_dataset\assemblyVectorized.pkl")
valid_gen = loadDataGeneratorBatches(r"D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_dataset\validationAssemblyVectorized.pkl")

model.fit_generator(train_gen,
                    epochs=50,
                    callbacks=[csv_logger, checkpoint],
                    steps_per_epoch=int(num_samples/batch_size),
                    validation_data=valid_gen,
                    validation_steps = int(num_samples_valid/batch_size),
                    )
          
# '''
# print("Testing output, fitting number:", i)
# getOutput = K.function([model.layers[0].input],
                       # [model.layers[1].output])
# layer_output = getOutput([train_x[0:50]])[0]
# print(layer_output)
# print()



# print("Evaluating")
# scores = model.evaluate(valid_x, valid_y, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


















