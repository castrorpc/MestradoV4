import numpy as np
from tensorflow.keras import callbacks
from ANFIS_Model import build_model

model = build_model(2, 2) # Instance of the model

# placeholder data
dummy_data = [[3, 4], [4, 5]]
dummy_target = [5, 6]

# TODO: GET DATA FROM USER INPUT AND EXTRACT num_inputs AND num_mf FROM IT -> easy step


# List of callbacks: ReduceLROnPlateu to better control learning when close to
# optimal solution and TensorBoard for visual avaliation
callbacks = [
    callbacks.ReduceLROnPlateu( monitor='val_loss', factor=0.1, patience=8, min_lr=0.0001),
    callbacks.TensorBoard(logdir='logs')
]
callbacks = [
    callbacks.TensorBoard(log_dir='logs')
]

# Train model with given data
model.fit(dummy_data, dummy_target, epochs=2)
