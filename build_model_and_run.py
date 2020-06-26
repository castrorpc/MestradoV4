'''
Terminal tool for defining an ANFIS model and training it without getting to code.
This only allows a small set of control over the creation and fitting of the model,
compared to all the tools tensorflow and keras provides teste roberto
'''

import argparse
import csv
import ANFIS_Model

def read_input_data(input_file):
    with open(input_file) as f:
        reader = csv.reader(f)
        input_data, output_data = [], []
        for row in reader:
            input_data.append([float(i) for i in row[:-1]])
            output_data.append(float(row[-1]))
        num_inputs = len(input_data[0])
    return num_inputs, input_data, output_data

def train(num_inputs, num_mf, epochs, input_data, input_target, optimizer, model_name, **kwargs):
    model = ANFIS_Model.Model(num_inputs, num_mf, type_mf=type_mf)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(input_data, input_target, epochs)
    model.save(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build and train an ANFIS Model\
     based on Tekagi-Sugeno rules with the amount of inputs and membership functions\
     you choose, as long as the number of mfs for each input is the same')

    parser.add_argument('input_data', help='full path to csv file containing the\
                        input data of the model. The last column of the file will\
                        be considered the target and all the other ones features.\
                        Check for data inconsistencies before training!')

    parser.add_argument('num_mf', help='number of membership functions for the\
                        inputs of the model', type=int)

    parser.add_argument('epochs', help='number of epochs for training', type=int

    parser.add_argument('optimizer', help='optimizer used in the training', type=str)

    parser.add_argument

    parser.add_argument('output', help='name of model', type=str)

    ## TODO: Add callbacks and train_test_split as args in a nice way

    args = parser.parse_args()

    num_inputs, input_data, output_data = read_input_data(input_file=args.input_data)

    train(args.num_inputs, args.num_mf, args.epochs, input_data, output_data, args.optimizer, args.output)
