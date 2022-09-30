import os
import sys
import numpy as np
import pandas as pd

# Grabbing /src System Path for Imports
sys.path.append(str(f'{os.getcwd()}/input'))
from binaryop_data import *

def construct_xor_df():
    df = pd.DataFrame(xor_data()[0].reshape(4,2))
    df = df.rename(columns={0: "x1", 1: "x2"})
    df['t'] = pd.DataFrame(xor_data()[1].reshape(4,1))
    return df

def construct_or_df():
    df = pd.DataFrame(or_data()[0].reshape(4,2))
    df = df.rename(columns={0: "x1", 1: "x2"})
    df['t'] = pd.DataFrame(or_data()[1].reshape(4,1))
    return df

def construct_and_df():
    df = pd.DataFrame(and_data()[0].reshape(4,2))
    df = df.rename(columns={0: "x1", 1: "x2"})
    df['t'] = pd.DataFrame(and_data()[1].reshape(4,1))
    return df


