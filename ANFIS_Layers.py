import tensorflow as tf
import numpy as np

# A few necessary classes which represnt the blueprint of each layer
class FuzzyficationLayer(tf.keras.layers.Layer):
    '''
    This layer indicates the 1st layer of the model, which is defined by the
    fuzzyfication processes
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FuzzyficationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This layer must have two built in parameters for each membership
        function of the inputs of the model. This is due to the used mf used
        (gaussian), which requires two parameters: the variance and the mean
        '''
        # We have 8 parameters: 2 for each mf (2 mf for each input->2 inputs):
        # 2*2*2 = 8
        self.parameters = self.add_weight(shape=(2, 4), trainable=False)
        super(FuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        '''
        Gaussian membership function to provide differentiable function through
        out the model (guarantee backpropagation)
        '''
        output = []
        for i in range(2):
            # For the first two inputs (number of membership functions of first
            # input), apply gaussian function
            mi = self.parameters[i][0] # Names here for clarity
            sigma = self.parameters[i][1] # Names here for clarity
            output.append(tf.math.exp((-(x[0] - mi))/(2*(sigma**2)))) # Actual function
        for i in range(2, 4):
            # For the other two inputs (number of membership functions of second
            # input), do the same thing
            mi = self.parameters[0][i] # Names here for clarity
            sigma = self.parameters[1][i] # Names here for clarity
            output.append(tf.math.exp((-(x[1] - mi))/2*(sigma**2))) # Actual function

        return output


class TNorm(tf.keras.layers.Layer):
    '''
    This layer represents the 2nd layer of the model, which is defined by the
    application of the T-norm in the fuzzyfied values
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
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
        output = [x[0]*x[2], x[0]*x[3], x[1]*x[2], x[1]*x[3]]
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
        output = [i/sum for i in x] # List of inputs devided by the sum
        return output


class ConsequentRules(tf.keras.layers.Layer):
    '''
    This layer represents the 4th layer of the model, which is defined by the
    definition (through some calculation) of the consequent rules
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ConsequentRules, self).__init__(**kwargs)

    def build(self, input_dim):
        '''
        This layer has the parameter p, q and r (3) for each consequent rule
        (in this case, 4), so we have to add these paramters (first executed line)
        '''
        self.parameters = self.add_weight(shape=(3,4), trainable=True)
        super(ConsequentRules, self).build(input_dim)

    def call(self, x):
        '''This layer receives two inputs: the original inputs of the model (
        cripy values) and the output of the 3rd layer'''
        # Separate the inputs into the original of model and the one coming from previous layer
        original_inputs, cons_rules_inputs = x
        output = []
        for i in range(2):
            # For each i in the range of the outputs, apply the formula in the paper
            # Names here for clarity:
            w = cons_rules_inputs[i] # output of previous layer
            p = self.parameters[i][0] # parameter of this layer, which multiplies the input 1 of model
            q = self.parameters[i][1] # parameter of this layer, which multiplies the input 2 of model
            r = self.parameters[i][2] # parameter of this layer, which adds a bias to this layer
            output.append(w*(p*original_inputs[0] + q*original_inputs[1] + r))
        tf.concat(output, axis=0)
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
