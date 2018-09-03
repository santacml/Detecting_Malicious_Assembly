import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # gets rid of AVX message
# os.environ['TF_CPP_MIN_VLOG_LEVEL']='3'  # log tensor allcoation?

# I have a GPU so idc

# kills GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import numpy
import random
import pickle, gzip, glob
import keras
from keras import optimizers
from keras import backend as K
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.preprocessing import sequence
from keras.preprocessing.sequence import TimeseriesGenerator

# pip install --ignore-installed --upgrade tensorflow-gpu
# C:\Users\msant_000\.keras
# C:\Users\msant_000\.theanorc

# from tempfile import mkdtemp
# import os.path as path
# filename = path.join(mkdtemp(), 'C:\NUMPYFILE.dat')

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# for only first cmd
# vocab size = 559
# file len = 28671


# vocab_size = 10097
# num_samples = 160
# vocab_size = 909
# vocab_size = 620
vocab_size = 202
batch_size = 5
# input_length = 500000
input_length = 10000
line_length = 18
window_size = 20
# num_samples = 10732
num_samples = 500
# num_samples_valid = 40
steps_per_epoch = num_samples/batch_size           

num_samples_valid = 100
valid_steps = num_samples_valid/batch_size  # should be this


def loadDataGeneratorBatches(assemblyVectorizedFile):
    readMe = gzip.open(assemblyVectorizedFile, "rb")
    while True:
        # for IDK in range(0, num_samples):  # if this works regardless of who calls, then idk I think this works
        try:
            xArr = []
            answers = []
            for x in range(0,batch_size):
                fileArr = pickle.load(readMe)
                xArr.append(fileArr[0])
                answers.append(fileArr[1])
                
            
            train_x = numpy.asarray(xArr, dtype="float32") 
            
            ans = []
            for x in range(0,batch_size):
                num = answers.pop(0)[0]
                answer = [0]*9
                answer[num-1] = 1
                ans.append(answer)
                # print("ANSWER IS", num)
            train_y = numpy.asarray(ans, dtype="float32")
            
            yield (train_x, train_y)
            
        except EOFError:
            del readMe
            readMe = gzip.open(assemblyVectorizedFile, "rb")
            
        
import tensorflow as tf
# config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

model = Sequential()

model.add(MaxPooling1D(10, input_shape=(input_length,line_length)))
model.add(TimeDistributed(Embedding(vocab_size,100)))
model.add(Conv2D(400, 4, activation='tanh'))
model.add(MaxPooling2D(data_format="channels_first", pool_size=(1,3)))
model.add(Conv2D(300, 3, activation='tanh'))
model.add(MaxPooling2D(data_format="channels_first", pool_size=(2,3)))
# model.add(Dropout(0.5))
# model.add(GlobalMaxPooling2D())
model.add(GlobalAveragePooling2D())


model.add(Dense(9, activation='sigmoid'))


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
model = load_model(r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_networks\6. Back to where it was\Conv-32.hdf5")
# adamTest = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
# sgd = optimizers.SGD(lr=.001, momentum=0.9, nesterov=True)
# model.compile(optimizer='sgd',
              # loss='binary_crossentropy',
              # metrics=['accuracy'])
print(model.summary())
# '''


train_gen = loadDataGeneratorBatches(r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_dataset\tempAssemblyVectorized\assemblyVectorized10klines.pklz")
valid_gen = loadDataGeneratorBatches(r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_dataset\tempAssemblyVectorized\validationAssemblyVectorized10klines.pklz")

# import tensorflow as tf
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
              # loss='binary_crossentropy',
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# print(0/0)
csv_logger = CSVLogger(r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_networks\trainingSeqConv.log")
filepath = r"D:\DAAACUMENTS\Research\3_NAECON2018\kaggle_networks\Conv-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath)


# model.fit(train_x, train_y, 
          # epochs=50, 
          # batch_size=batch_size, 
          # callbacks=[csv_logger, checkpoint],
          # validation_data=(valid_x, valid_y))
          
model.fit_generator(train_gen,
                    epochs=50,
                    callbacks=[csv_logger, checkpoint],
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=valid_gen,
                    validation_steps = valid_steps,
                    )
          
          

# print("Testing output, fitting number:", i)
# getOutput = K.function([model.layers[0].input],
                       # [model.layers[1].output])
# layer_output = getOutput([train_x[0:50]])[0]
# print(layer_output)
# print()



# print("Evaluating")
# scores = model.evaluate(valid_x, valid_y, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))


















