import os
import sys

# Grabbing /model System Path for Imports
sys.path.append(str(f'{os.getcwd()}/model'))
from model import *

# Grabbing /input System Path for Imports
sys.path.append(str(f'{os.getcwd()}/input'))
from binaryop_data import *

# Grabbing /utils System Path for Imports
sys.path.append(str(f'{os.getcwd()}/utils'))
from construct_df import *

# Grabbing xor, or, and Data from Input
xor_x_train, xor_y_train = xor_data()
or_x_train, or_y_train = or_data()
and_x_train, and_y_train = and_data()

# Running the NN with Varrying Learning Rates for Xor, Or, and And
if __name__ == '__main__':

    # NN with Learning Rate of 0.1
    print('-------- XOR TABLE --------')
    preds = create_network(xor_x_train, xor_y_train, 0.1, ['r', 'xor', 0])
    df_1 = construct_xor_df()
    df_1['y1'] = pd.DataFrame(preds)
    print(df_1, '\n')

    # NN with Learning Rate of 0.01
    print('-------- XOR TABLE --------')
    preds = create_network(xor_x_train, xor_y_train, 0.01, ['g', 'xor', 0])
    df_1 = construct_xor_df()
    df_1['y1'] = pd.DataFrame(preds)
    print(df_1, '\n')
    
    # NN with Learning Rate of 0.001
    print('-------- XOR TABLE --------')
    preds = create_network(xor_x_train, xor_y_train, 0.001, ['b', 'xor', 0])
    df_1 = construct_xor_df()
    df_1['y1'] = pd.DataFrame(preds)
    print(df_1, '\n')

    
    # NN with Learning Rate of 0.1
    print('-------- OR TABLE --------')
    preds = create_network(or_x_train, or_y_train, 0.1, ['r', 'or', 1])
    df_2 = construct_or_df()
    df_2['y1'] = pd.DataFrame(preds)
    print(df_2, '\n')

    # NN with Learning Rate of 0.01 
    print('-------- OR TABLE --------')
    preds = create_network(or_x_train, or_y_train, 0.01, ['g', 'or', 1])
    df_2 = construct_or_df()
    df_2['y1'] = pd.DataFrame(preds)
    print(df_2, '\n')

    # NN with Learning Rate of 0.001
    print('-------- OR TABLE --------')
    preds = create_network(or_x_train, or_y_train, 0.001, ['b', 'or', 1])
    df_2 = construct_or_df()
    df_2['y1'] = pd.DataFrame(preds)
    print(df_2, '\n')
    
    # NN with Learning Rate of 0.1
    print('-------- AND TABLE --------')
    preds = create_network(and_x_train, and_y_train, 0.1, ['r', 'and', 2])
    df_3 = construct_and_df()
    df_3['y1'] = pd.DataFrame(preds)
    print(df_3, '\n')

    # NN with Learning Rate of 0.01
    print('-------- AND TABLE --------')
    preds = create_network(and_x_train, and_y_train, 0.01, ['g', 'and', 2])
    df_3 = construct_and_df()
    df_3['y1'] = pd.DataFrame(preds)
    print(df_3, '\n')

    # NN with Learning Rate of 0.001
    print('-------- AND TABLE --------')
    preds = create_network(and_x_train, and_y_train, 0.001, ['b', 'and', 2])
    df_3 = construct_and_df()
    df_3['y1'] = pd.DataFrame(preds)
    print(df_3, '\n')
