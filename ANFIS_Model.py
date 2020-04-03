from tensorflow.keras.Models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossEntropy as CCE
from ANFIS_Layers import *

def build_model(input_names, classes_of_inputs):
    # Number of different layers that acts as Fuzzifier layers for the different
    # types of inputs
    number_of_inputs = len(input_names)
    # Creating empty list (wasn't able to do this with list comprehension)
    inputs = []
    # For each input layer, described by its input_name and the classes of
    # memberships which the data in the crisp sets can be a member of
    for input_name, classes in zip(input_names, classes_of_inputs):
        # An input has the shape of the number of classes that the crisp set can
        # be a member of
        input = Input(shape=(np.shape(classes),), dtype='int32', name=input_name)
        # Add a fuzzification layer
        x = fuzzyficationLayer(len(classes), classes)(input)
        # Append the result of the fuzzification layers to the inputs list
        inputs.append(x)
    # Add a conjuction layer
    y = conjunctionLayer(number_of_inputs)(inputs)
    # Add a Dense (fully connected layer)
    y = Dense(number_of_inputs, activation='relu')(y)
    # And the last one is the defuzzification layer, so the result is objective
    y = defuzzyficationLayer(1)(y)

    # Instantiate model
    ANFIS_Model = Model(inputs=[x], outputs=[y])
    # Add the important the metrics, specifficaly those related to de validation data
    metrics = ['val_loss', 'val_accuracy']

    # Compile the model and return it
    ANFIS_Model.compile(optimizer='adam', loss=CCE(), metrics=metrics)
    return ANFIS_Model


# Instance of ANFIS model with placeholder parameters
ANFIS_Model = build_model(
    ['investment', 'type_of_contract', 'region', 'size_of_company', 'type_of_service'],
    [
        [mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0})],
        [mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0}), , mf_class({'mean': 0, 'std': 0})]
        [mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0}), , mf_class({'mean': 0, 'std': 0})]
        [mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0})]
        [mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0}), mf_class({'mean': 0, 'std': 0})]
    ]
)
