import tensorflow as tf
import numpy as np
import itertools


class FuzzyficationLayer(tf.keras.layers.Layer):
    '''
    This layer indicates the 1st layer of the model, which is defined by the
    fuzzyfication processes
    '''
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        self.output_dim = self.num_mf*num_inputs
        super(FuzzyficationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer must have two built in parameters for each membership
        function of the inputs of the model. This is due to the used mf used
        (gaussian), which requires two parameters: the variance and the mean
        '''
        self.parameters = self.add_weight(shape=(self.output_dim*self.num_mf, 2), trainable=False)
        super(FuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        '''
        Gaussian membership function to provide differentiable function through
        out the model (guarantee backpropagation)
        '''
        output = []
        offset = 0
        for i in range(self.num_inputs):
            for j in range(self.num_mf):
                mi = self.parameters[offset + j][0]
                sigma = self.parameters[offset + j][1]
                output.append(tf.math.exp((-(x[0][i] - mi)**2)/2*sigma**2))
            offset += self.num_mf
        return output


class TNorm(tf.keras.layers.Layer):
    '''
    This layer represents the 2nd layer of the model, which is defined by the
    application of the T-norm in the fuzzyfied values
    '''
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.output_dim = num_mf**num_inputs
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        super(TNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer has no parameters buit in them
        '''
        super(TNorm, self).build(input_shape)

    def call(self, x):
        '''
        Calling this layer must provide a differentiable set of operations, so
        we use the multiply AND operator
        '''
        # Multiply the membership value for each possible combination of inputs
        # e.g.: input1 is high (x[0]) and input2 is near (x[2])
        divided_inputs = []
        for i in range(self.num_inputs):
            divided_inputs.append(x[i*self.num_mf:(i+1)*self.num_mf])
        comb = itertools.product(*divided_inputs)
        output = []
        for i in comb:
            mult_ = 1
            for j in i:
                mult_ *= j
            output.append(mult_)
        return output


class NormFiringStrength(tf.keras.layers.Layer):
    '''
    This layer represents the 3rd layer of the model, which is defined by the
    normalization of the firing strengths to each rule (how strong it is in
    comparison to the others)
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NormFiringStrength, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer has no parameters buit in them
        '''
        super(NormFiringStrength, self).build(input_shape)

    def call(self, x):
        '''
        To normalize the values, we just devide each of them by the sum of all
        of them
        '''
        sum = tf.keras.backend.sum(x) # Sum of inputs
        output = x/sum
        return tf.concat(output, 0)


class ConsequentRules(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_mf, **kwargs):
        self.num_inputs = num_inputs
        self.num_mf = num_mf
        self.output_dim = num_mf**num_inputs
        super(ConsequentRules, self).__init__(**kwargs)

    def build(self, input_dim):
        '''
        This layer has a set of parameters similar to the ones in dense layers.
        In fact, the first part of the node operations in this layer could be
        replaced with a built in Dense layer of Keras
        '''
        self.parameters = self.add_weight(shape=(self.num_inputs, self.output_dim), trainable=True)
        super(ConsequentRules, self).build(input_dim)

    def call(self, x):
        # Separate the two inputs of this layer
        original_inputs, cons_rules_inputs = x
        # Dense operations
        original_inputs = tf.transpose(tf.matmul(original_inputs, self.parameters))
        output = []
        # Multiply result of dense with the output of layer 3
        for i in range(self.num_mf**self.num_inputs):
            output.append(original_inputs[i]*cons_rules_inputs[i])
        return output


class DeffuzzyficationLayer(tf.keras.layers.Layer):
    '''
    This layer represents the 5th layer of the model, which is defined by the
    deffuzyfication of the fuzzyfied values, turning them back to cripy values
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DeffuzzyficationLayer, self).__init__(**kwargs)

    def build(self, input_dim):
        '''
        This layer has no built in parameters
        '''
        super(DeffuzzyficationLayer, self).build(input_dim)

    def call(self, x):
        '''
        As applied in the ANFIS paper of Jyh-Shing Roger Jang, the output of the
        fifth layer of the type-3 Takagi-Sugeno type is the sum of the inputs
        '''
        return tf.keras.backend.sum(x)
