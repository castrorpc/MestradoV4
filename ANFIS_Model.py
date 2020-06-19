'''
This file contains the class for the ANFIS model itself, constructing the tensorflow
computational graph. The operations that happens inside the classes itself can be
checked in the ANFIS_Layers.py file
'''

import tensorflow as tf
import ANFIS_Layers

class Model(tf.keras.models.Model):
    def __init__(self, num_inputs, num_mf, type_mf='gaussian'):
        super(Model, self).__init__()
        self.input_ = tf.keras.layers.Input(shape=(num_inputs,)) # Input instance
        if type_mf == 'gaussian':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_gaussian(num_inputs, num_mf) # 1st layer instance
        elif type_mf == 'bell':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_bell(num_inputs, num_mf)
        elif type_mf == 'triangular':
            self.fuzzyfication = ANFIS_Layers.FuzzyficationLayer_triangular(num_inputs, num_mf)
        self.t_norm = ANFIS_Layers.TNorm(num_inputs, num_mf) # 2nd layer instance
        self.norm_fir_str = ANFIS_Layers.NormFiringStrength(num_mf**num_inputs) # 3rd layer instance
        self.conseq_rules = ANFIS_Layers.ConsequentRules(num_inputs, num_mf) # 4th layer instance
        self.defuzz = ANFIS_Layers.DeffuzzyficationLayer(1) # 5th layer instance

    def call(self, inputs):
        x = self.fuzzyfication(inputs)
        x = self.t_norm(x)
        x = self.norm_fir_str(x)
        x = self.conseq_rules([inputs, x])
        y = self.defuzz(x)
        return y
