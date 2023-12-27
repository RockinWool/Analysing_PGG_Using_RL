import numpy as np
from .Calculate_payoff import Payoff_dealer as pad
import pandas as pd

class Data_preparing:
    def data_creator(N):
        input_data = np.random.rand(N,2)
        my_cooperate_rates = input_data[:,0]
        oponent_cooperate_rates = input_data[:,1]
        #print(input_data[1],my_cooperate_rates[1],oponent_cooperate_rates[1])
        payoff_matrix_base = np.array([[(2,1),(-1,-1)], [(-1,-1),(1,2)]])
        payoff_matrix = pad.normalize_payoff(payoff_matrix=payoff_matrix_base)
        
        output_data = pad.calculate_payoff(my_cooperate_rates,oponent_cooperate_rates,payoff_matrix)
        #print(output_data)
        df = pd.DataFrame()
        df["p1"] = input_data[:,0]
        df["p2"] = input_data[:,1]
        df["output"] = output_data
        df.to_csv("data/datablob.csv",index=False)
        return(input_data,output_data)


if __name__ == "__main__":
    N = 200
    input, output = Data_preparing.data_creator(N)
    print(input,output)


