import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.layers import Wrapper
from keras.utils.generic_utils import has_arg
import numpy as np
'''
class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
'''
class DFTLayer(Layer):

    def __init__(self, **kwargs):
        super(DFTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel', 
                                      # shape=(input_shape[1], self.output_dim),
                                      # initializer='uniform',
                                      # trainable=True)
        super(DFTLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        input_shape = K.int_shape(input)
        print("INPIUT SHAPE", input_shape)
        
        # input = K.reshape(input, input_shape[1] + input_shape[2] + input_shape[0])
        input = tf.cast(input, tf.complex64)
        out = K.fft2d(input)
        # out = input
        out = tf.cast(out, tf.float32)
        # out = K.reshape(input, (input_shape[0], input_shape[1], input_shape[2]))
        
        out_shape = K.int_shape(out)
        print("OUt SHAPE", out_shape)
        return out

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], self.output_dim)
        return input_shape

        
class OnceOnLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# '''
class CPUWrapper(Wrapper):
    def __init__(self, layer, **kwargs):
        super(CPUWrapper, self).__init__(layer, **kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = InputSpec(shape=input_shape)
        child_input_shape = input_shape
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(CPUWrapper, self).build()

    def compute_output_shape(self, input_shape):
        # essentially just get rid of timesteps
        child_input_shape = input_shape
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return child_output_shape

    
    def call(self, inputs, training=None):
        with K.device('/cpu:0'):
            out = self.layer.call(inputs)
        return out


class TimeDistributedRAM(Wrapper):
    def __init__(self, layer, **kwargs):
        super(TimeDistributedRAM, self).__init__(layer, **kwargs)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = InputSpec(shape=input_shape)
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributedRAM, self).build()

    def compute_output_shape(self, input_shape):
        # essentially just get rid of timesteps
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, inputs, training=None):
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        uses_learning_phase = False

        input_shape = K.int_shape(inputs)
        if False and input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, _):
                global uses_learning_phase
                output = self.layer.call(x, **kwargs)
                if hasattr(output, '_uses_learning_phase'):
                    uses_learning_phase = (output._uses_learning_phase or
                                           uses_learning_phase)
                return output, []

            _, outputs, _ = K.rnn(step, inputs,
                                  initial_states=[],
                                  input_length=input_shape[1],
                                  unroll=False)
            y = outputs
        else:
            '''
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            input_uid = _object_list_uid(inputs)
            inputs = K.reshape(inputs, (-1,) + input_shape[2:])
            self._input_map[input_uid] = inputs
            # (num_samples * timesteps, ...)
            y = self.layer.call(inputs, **kwargs)
            if hasattr(y, '_uses_learning_phase'):
                uses_learning_phase = y._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])
            '''
            input_length = input_shape[1]
            output_shape = self.compute_output_shape(input_shape)
            
            # self.gpuInputVar = tf.zeros( (input_shape[0],) + input_shape[2:])
            self.gpuInputVar = None
            # with K.device('/cpu:0'):
                # self.cpuOutputVar = tf.zeros(output_shape)
            
            
            # tsteps = None
            # with K.device('/cpu:0'):
                # tsteps = K.unstack(inputs, axis=1)
                # outList = []
            # ins = K.reshape(inputs, (input_length, input_shape[0]) + input_shape[2:])
            
            numDims = len(input_shape)
            with K.device('/cpu:0'):
                ins = tf.transpose(inputs, (1, 0) + tuple([i for i in range(2, numDims)]))
            
            def foo(timestepPlusDim):
                self.gpuInputVar = timestepPlusDim
                # with tf.device('/gpu:0'):
                    # self.gpuInputVar = tf.squeeze(timestepPlusDim)
                # with tf.device('/cpu:0'):
                    # return self.layer.call(self.gpuInputVar)
                
                with tf.device('/gpu:0'):
                    out = self.layer.call(self.gpuInputVar)
                
                with tf.device('/cpu:0'):
                    return out
            
            with tf.device('/cpu:0'):
                y = tf.map_fn(foo, 
                              ins,
                              back_prop = True,
                              swap_memory=True
                )
            
            
            
            '''
            # this should be on the device we are otherwise using... gpu...
            # for i, timestep in enumerate(tsteps):
            for timestep in tsteps:
                # out = None
                # with tf.device('/gpu:0'):
                # test = tsteps[i]
                # with tf.device('/cpu:0'):
                # gpuVar = tf.identity(tsteps[i])
                gpuVar = timestep * 1
                
                # with K.device('/cpu:0'):
                    # out = self.layer.call(gpuVar, **kwargs)
                out = self.layer.call(gpuVar, **kwargs)
                
                with tf.device('/cpu:0'):
                    cpuVar = out * 1
                    # outList.append(self.layer.call(tsteps[i] , **kwargs))
                    outList.append(cpuVar)
                    
                # tsteps[i] = K.reshape(tsteps[i], (-1, 1) + output_shape[2:])
            # '''
            
            
            # ins = K.reshape(inputs, (-1,) + input_shape[2:])
            # ins = tf.transpose(inputs, (1, 0,2,3))
            # def test(timestep):
                # with tf.device('/cpu:0'):
                    # return self.layer.call(timestep)
            
            # y = tf.map_fn(test, ins)
            # print(len(outList))
                
            
            # with tf.device('/cpu:0'):
            # with K.device('/cpu:0'):
                # y = K.stack(outList, axis=1)
        
        # Apply activity regularizer if any:
        if (hasattr(self.layer, 'activity_regularizer') and
           self.layer.activity_regularizer is not None):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        return y

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# '''