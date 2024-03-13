import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'    
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import yaml
from scipy.stats import truncnorm
import networkx as nx
import gc
# from memory_profiler import profile
from keras import backend as K

from library.Qnetwork import QNetwork
from library.Qlearning_agent_settings import Actor, Memory
from collections import deque
from library.Qlearning_agent import Agent
from library.Qlearning_agent_strategies import _031_AI_select, _031_all_cooperate,_031_all_defect,_031_random_select,_031_tit_for_tat,_031_trigger,_031_WSLS


class Continuous_Game_Framework:
    def __init__(self,alpha,n_player):
        # Initialize Cooperating rate
        self._010_init_variables(alpha,n_player)
        self._020_run_trial()  

    def _001_numpy_deque_push(self,nparray,input):
        nparray[1:] = nparray[0:-1]
        nparray[0] = input        
        return nparray

    def _002_load_yaml(self):
        #load parmeters
        with open('./data/parameters.yaml', 'r') as yml:
            self.params = yaml.safe_load(yml)

    def _010_init_variables(self,alpha,n_player):
        #-----------Change Filename-----#
        self.csv_filename = "./data/sample.csv"
        #-------------------------------#
        self._002_load_yaml()
        self.num_steps = self.params[ "num_steps"]
        self.num_episodes = self.params["num_episodes"]
        self.num_trials = self.params["num_trials"]
        if alpha == "":
            self.alpha = self.params["alpha"]
        else:
            self.alpha = [alpha]
        if n_player=="":
            self.num_players = self.params["num_players"]
        else:
            self.num_players = [n_player]
        self._011_init_df()


    def _011_init_df(self):
        try:
            self.df_crate = pd.read_csv(self.csv_filename,index_col=0)
        except:
            data = np.zeros((self.num_trials,len(self.alpha)*len(self.num_players)))
            columns = []
            for i in self.alpha:
                for j in self.num_players:
                    columns.append("{}_{}".format(i,j))
            index = range(self.num_trials)
            self.df_crate = pd.DataFrame(data = data, columns=columns,index=index)
            self.df_crate.to_csv(self.csv_filename)
            exit(1)

    def _020_run_trial(self):
        for n_player in self.num_players:
            for alpha in self.alpha:
                for trial in range(self.num_trials):
                    print("--------------------trial No.{}, n_player = {}, alpha = {}--------------------".format(trial,n_player,alpha))
                    self._021_init_state(n_player)
                    self._022_init_agents(n_player,alpha)
                    self._023_set_b(n_player,alpha)
                    self._024_init_crate_average()
                    self._030_run_epipsodes(n_player,alpha)
                    self._025_add_crate_to_df(trial,alpha,n_player)
                    self._026_clean_memory()

    def _021_init_state(self,n_player):
        choice = [0,1]
        self.players_action = np.random.choice(choice,(self.num_steps,n_player))
   
    def _022_init_agents(self,n_player,alpha):
        self.agents = [[] for i in range(n_player)]
        name = "./data/Agents/{}AI_{}_0926".format(n_player,alpha)

        for i in range(n_player):
            try:
                self.agents[i] = Agent(load_mode=False,agent_number=i,name=name)
                print("create network")
            except:
                self.agents[i] = Agent(load_mode=False,agent_number=i,name=name)
                print("create network")
        
    def _023_set_b(self,n_player,alpha):
        self.b = alpha*n_player

    def _024_init_crate_average(self):
        self.crate_average = 0
        
    def _025_add_crate_to_df(self,trial,alpha,n_player):
        #chaange here, also require to change line 135
        self.crate_average /= (self.num_episodes-10)
        self.df_crate.at[trial,"{}_{}".format(alpha,n_player)] = self.crate_average


    def _026_clean_memory(self):
        gc.collect()
        K.clear_session()
        self.df_crate.to_csv(self.csv_filename)
        del self.players_action
        del self.agents
        del self.b
        del self.crate_average
    
    # @profile
    def _030_run_epipsodes(self,n_player,alpha):
        for episode in range(self.num_episodes):
            self._035_targetQN_learning()
            self._021_init_state(n_player)
            for t in range(self.num_steps):                    
                for agent in range(n_player):
                    _031_AI_select(GFW=self,episode=episode,agent_num=agent,n_player=n_player)
                self._032_give_reward(n_player)
                self._033_memorize_data(n_player)
                self._034_update_state()
            print('%d Episode finished after %d time steps / mean %2f' % (episode, t, self.agents[0].rewards.mean()))
            self._035_learning_main()                    
            if episode >9:
                self._037_add_crate_()
                
    def _032_give_reward(self,n_player):   
        num_of_Defector = np.sum(self.players_action[0])    
        for i in range(len(self.agents)):
            if self.players_action[0][i]==0:
                temp_reward = (n_player-num_of_Defector)*self.b/n_player -1
            elif self.players_action[0][i]==1:
                temp_reward = (n_player-num_of_Defector)*self.b/n_player 
            else:
                print("rewrd error");exit(1)
                                
            self.agents[i].rewards = self._001_numpy_deque_push(self.agents[i].rewards,temp_reward)

    def _033_memorize_data(self,n_player):
        state_base=[]
        for i in range(2):
            state_base.append(np.reshape(self.players_action[i], (1, n_player)).copy() )
            
        for i in range(len(self.agents)):
            action = self.players_action[0][i]
            
            last_action = self.players_action[1][i]
            current_state_base = np.delete(state_base[1],[i])
            num_of_Defector = np.sum(current_state_base)
            next_state_base = np.delete(state_base[0],[i])
            next_num_of_Defector = np.sum(next_state_base)
            cooperator_degree = 1-num_of_Defector/(n_player-1)
            next_cooperators_degree = 1 - next_num_of_Defector/(n_player-1)
            
            state = np.array([cooperator_degree,last_action])
            next_state = np.array([next_cooperators_degree,action])              
            self.agents[i].memory.add([state, action ,self.agents[i].rewards[0],next_state])
        
    def _034_update_state(self):
        self.players_action = np.roll(self.players_action,1,axis=0)
        self.players_action[0] = self.players_action[1]
    
    def _035_learning_main(self):
        for i in range(len(self.agents)):
            if i==0:
                self.agents[i]._60_learning_main()
            else:
                self.agents[i]._60_learning_main()
            
    def _035_targetQN_learning(self):
        for i in range(len(self.agents)):
            self.agents[i]._71_targetQN_learning()
        
    
    def _036_save(self,n_player,alpha):
        for i in range(len(self.agents)):
            self.agents[i]._071_save_with_name(agent_number=i,name = "./data/Agents/{}AI_{}_0926".format(n_player,alpha))

    def _037_add_crate_(self):
        self.crate_average += 1 - np.mean(self.players_action)

if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-alpha":
            alpha = float(sys.argv[i+1])
        elif sys.argv[i]=="-n":
            n_player = int(sys.argv[i+1])

    print("alpha={}, n_player={}".format(alpha,n_player))
    Proj = Continuous_Game_Framework(alpha,n_player)


