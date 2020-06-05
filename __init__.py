import numpy as np
from tensorflow.keras import callbacks
from ANFIS_Model import ANFIS_Model

# placeholder data
dummy_data1 = [[3, 4], [4, 5]]
dummy_target = [5, 6]

# In the future:
'''X_train = csv.read(...)
y_train = csv.read(...)
x_test = csv.read(...)
y_test = csv.read(...)
'''


# List of callbacks: ReduceLROnPlateu to better control learning when close to
# optimal solution and TensorBoard for visual avaliation
'''callbacks = [
    callbacks.ReduceLROnPlateu( monitor='val_loss', factor=0.1, patience=8, min_lr=0.0001),
    callbacks.TensorBoard(logdir='logs')
]'''
callbacks = [
    callbacks.TensorBoard(log_dir='logs')
]

# Train model with given data and save it
ANFIS_Model.fit(X_train, y_train, validation_data=[x_test, y_test], epochs=5, callbacks=callbacks)
ANFIS_Model.save('neuro_fuzzy.model')
