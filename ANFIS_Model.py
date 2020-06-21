'''
This file contains the class for the ANFIS model itself, constructing the tensorflow
computational graph. The operations that happens inside the classes itself can be
checked in the ANFIS_Layers.py file
'''

import tensorflow as tf
import ANFIS_Layers

class Model(tf.keras.models.Model):
    def __init__(self, num_inputs, num_mf, type_mf='gaussian'):
         #the eager execution is necessary for making iteration over tensor possible
         # although it is made to use for debugging
        tf.config.experimental_run_functions_eagerly(True)
        super(Model, self).__init__()
        if type_mf == 'gaussian':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_gaussian(num_inputs, num_mf) # 1st layer instance gaussian
        elif type_mf == 'bell':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_bell(num_inputs, num_mf) # 1st layer instance bell
        elif type_mf == 'triangular':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_triangular(num_inputs, num_mf) # 1st layer instance triangular
        self.t_norm = ANFIS_Layers.TNorm(num_inputs, num_mf) # 2nd layer instance
        self.norm_fir_str = ANFIS_Layers.NormFiringStrength(num_mf**num_inputs) # 3rd layer instance
        self.dense = tf.keras.layers.Dense(num_mf**num_inputs, trainable=True) # Desne layer representing 1st part of 4th layer operations
        self.conseq_rules = ANFIS_Layers.ConsequentRules(num_inputs, num_mf) # 4th layer instance
        self.defuzz = ANFIS_Layers.DeffuzzyficationLayer(1) # 5th layer instance

    def call(self, inputs):
        x = self.fuzzyfication(inputs)
        x = self.t_norm(x)
        x = self.norm_fir_str(x)
        d = self.dense(inputs)
        x = self.conseq_rules([d, x])
        y = self.defuzz(x)
        return y
