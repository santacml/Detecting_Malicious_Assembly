import os
os.environ["PATH"] += os.pathsep + r'C:\Users\msant_000\Anaconda3\envs\keras\Library\bin\graphviz'

from keras.utils import plot_model
from keras.models import load_model
model = load_model(r'D:\DAAACUMENTS\Research\3_NAECON2018\realpgm_networks\21.MaxEmb ConvNet 240 Files\Conv-50.hdf5')
plot_model(model, to_file='model.png', show_shapes =True)




 