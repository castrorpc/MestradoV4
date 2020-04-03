from tensorflow.keras.layers import Layer
from skfuzzy import gaussmf, defuzz

class mf_class():
    def __init__(self, mf):
        self.mf = mf

    def calc_mf(x):
        return gaussmf(x, self.mf.get('mean'), self.mf.get('std'))


class fuzzyficationLayer(Layer):
    def __init__(self, output_dim, classes, **kwargs):
        self.output_dim = output_dim
        self.classes = classes
        self.super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=False)
        super(fuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        output = []
        for class in self.classes:
            for i in x:
                # For each class of membership calculate the membership value of
                # each input
                membership_values = class.calc_mf(i)
                output.append(i)
        return np.array(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class conjunctionLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.super().__init__(**kwargs)

    def build(self, input_shape):
        assert isistance(input_shape, list)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1][0], self.output_dim),
                                      initializer='uniform',
                                      trainable=True))
        super(fuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        # Assert that x is a list (multiple types of inputs in ANFIS model)
        assert isistance(x, list)
        # Apply T-Norm to each of the input values
        output = np.array([np.max(z) for z in x])
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]


class defuzzyficationLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(defuzzyficationLayer, self).build(input_shape)

    def call(self, x):
        return [defuzz(i, gaussmf, 'centroid') for i in x]
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
