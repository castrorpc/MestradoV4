import tensorflow as tf
import ANFIS_Layers

def build_model(num_inputs, num_classes):
    input_ = tf.keras.layers.Input(shape=(2,)) # Input instance
    fuzzyfication = ANFIS_Layers.FuzzyficationLayer(4)(input_) # 5th layer instance
    t_norm = ANFIS_Layers.TNorm(4)(fuzzyfication) # 2nd layer instance
    norm_fir_str = ANFIS_Layers.NormFiringStrength(4)(t_norm) # 3rd layer instance
    conseq_rules = ANFIS_Layers.ConsequentRules(4)([norm_fir_str, input_]) # 4th layer instance
    defuzz = ANFIS_Layers.DeffuzzyficationLayer(1)(conseq_rules) # 5th layer instance

    ANFIS_Model = tf.keras.models.Model([input_], [defuzz])
    # Add the important the metrics, specifficaly those related to de validation data
    metrics = ['mean_absolute_error']

    # Compile the model and return it
    ANFIS_Model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)
    return ANFIS_Model


# Instance of ANFIS model with placeholder parameters
ANFIS_Model = build_model(2, 2)
