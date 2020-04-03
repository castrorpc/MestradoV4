import numpy as np
from tensorflow.keras import callbacks
from ANFIS_Model import ANFIS_Model

# placeholder data
X_train = np.array([1])
y_train = np.array([1])
x_test = np.array([1])
y_test = np.array([1])

# List of callbacks: ReduceLROnPlateu to better control learning when close to
# optimal solution and TensorBoard for visual avaliation
callbacks = [
    callbacks.ReduceLROnPlateu( monitor='val_loss', factor=0.1, patience=8, min_lr=0.0001),
    callbacks.TensorBoard(logdir='logs')
]

# Train model with given data and save it
ANFIS_Model.fit(X_train, y_train, validation_data=[x_test, y_test], callbacks=callbacks)
ANFIS_ModelS.save('neuro_fuzzy.model')
