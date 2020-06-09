import tensorflow as tf
import ANFIS_Layers

def build_model(num_inputs, num_mf):
    input_ = tf.keras.layers.Input(shape=(num_inputs,)) # Input instance
    fuzzyfication = ANFIS_Layers.FuzzyficationLayer(num_inputs, num_mf)(input_) # 1st layer instance
    t_norm = ANFIS_Layers.TNorm(num_inputs, num_mf)(fuzzyfication) # 2nd layer instance
    norm_fir_str = ANFIS_Layers.NormFiringStrength(num_mf**num_inputs)(t_norm) # 3rd layer instance
    conseq_rules = ANFIS_Layers.ConsequentRules(num_inputs, num_mf)([input_, norm_fir_str]) # 4th layer instance
    defuzz = ANFIS_Layers.DeffuzzyficationLayer(1)(conseq_rules) # 5th layer instance

    ANFIS_Model = tf.keras.models.Model([input_], [defuzz])
    # Add the important the metrics, specifficaly those related to de validation data
    metrics = ['mean_absolute_error']

    # Compile the model and return it
    ANFIS_Model.compile(optimizer='adam', loss='mean_squared_error', metrics=metrics)
    return ANFIS_Model
