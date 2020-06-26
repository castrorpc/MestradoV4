import ANFIS_Model
import csv
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

input_data = []
output_data = []
with open('test.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        input_data.append([float(x) for x in row[:4]])
        output_data.append(float(row[4]))


model = ANFIS_Model.Model(4, 2)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(input_data, output_data, epochs=5)
