import pandas as pd
import csv
import tensorflow as tf
import ANFIS_Layers


# No arquivo excel, os números deve ter "." e não ",". 
# Usar a função do excel substituir "," por "."
# Savlar planilha ".xls" como ".csv" em qualque uma das opções
# No arquivo txt, tem que separar por "," e não por ";". 
# usar a função do Bloco de Notas substituir "," por "."

# leitura da base de dados
# base = pd.read_csv('teste_RPC_entrada_titulo.csv')

# Posso testar depois: read_excel(io [, nome_da_pasta, cabeçalho, nomes,…]). Leia um arquivo do Excel em um DataFrame do pandas.

# pega para todos os registros as primeiras quatro colunas (0,1,2 e 3) 
# input_data = base.iloc[:, 0:4]
# : significa que pega todos os registros
# .values converte para o formato numpy

# pega especificamente a coluna 4
# output_data = base.iloc[:, 4]
 
      
input_data = []
output_data = []
with open('test.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        input_data.append([float(x) for x in row[:4]])
        output_data.append(float(row[4]))
        

# conta a quantidade de registros
num_inputs = 4
# No comando ussado pelo Tito: num_inputs = len(input_data[0]), retorna 4, 
# talvez as linhas estejam trocadas com as colunas

num_mf = 2
epochs = 5
optimizer = 'adam'
output = 'saida_RPC'
type_mf= 'gaussian'

# def train(num_inputs, num_mf, epochs, input_data, input_target, optimizer, model_name, **kwargs):
#    model = ANFIS_Model.Model(num_inputs, num_mf)
#    model.compile(optimizer=optimizer, loss='mean_squared_error')
#    model.fit(input_data, input_target, epochs)
#    model.save(model_name)

# train(num_inputs, num_mf, epochs, input_data, output_data, optimizer, output)
# Na def train, model_name = output = 'saida_RPC'
# Na def train, input_darget = output_data
    
def build_model(num_inputs, num_mf):
    input_ = tf.keras.layers.Input(shape=(num_inputs,)) # Input instance
    fuzzyfication = ANFIS_Layers.FuzzyficationLayer_gaussian(num_inputs, num_mf)(input_) # 1st layer instance
    t_norm = ANFIS_Layers.TNorm(num_inputs, num_mf)(fuzzyfication) # 2nd layer instance
    norm_fir_str = ANFIS_Layers.NormFiringStrength(num_mf**num_inputs)(t_norm) # 3rd layer instance
    conseq_rules = ANFIS_Layers.ConsequentRules(num_inputs, num_mf)([input_, norm_fir_str]) # 4th layer instance
    defuzz = ANFIS_Layers.DeffuzzyficationLayer(1)(conseq_rules) # 5th layer instance

 Model = tf.keras.models.Model([input_], [defuzz])

model = Model(4, 2)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(input_data, output_data, epochs=5)




