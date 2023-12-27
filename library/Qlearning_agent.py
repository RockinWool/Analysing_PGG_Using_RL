import yaml
import numpy as np
from .Qlearning_agent_settings import Memory
from .Qlearning_agent_settings import Actor
from .Qnetwork import QNetwork

class Agent:
    def __init__(self,load_mode = False,agent_number=0,name="") -> None:
        with open('./data/agent_parameters.yaml', 'r') as yml:
            self.agentset = yaml.safe_load(yml)
        
        # [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
        if load_mode:
            self.mainQN = QNetwork(self.agentset["learning_rate"],name=name+"{}.h5".format(agent_number))     # メインのQネットワーク
            self.targetQN = QNetwork(self.agentset["learning_rate"],name=name+"{}.h5".format(agent_number))   # 価値を計算するQネットワーク
        else:
            self.mainQN = QNetwork(self.agentset["learning_rate"],name="")     # メインのQネットワーク
            self.targetQN = QNetwork(self.agentset["learning_rate"],name="")   # 価値を計算するQネットワーク
        self.memory = Memory(max_size=self.agentset["memory_size"])
        
        self.players_action = 0
        # preparing for reward storing
        self.rewards = np.zeros(10,)
        # for decide action
        self.actor = Actor()
        
    def _60_learning_main(self):
        if (self.memory.len() > self.agentset["batch_size"]):
            self.mainQN.replay(self.memory, self.agentset["batch_size"],\
                                self.agentset["gamma"], \
                                self.targetQN)
            # print("Finish Learning")
            
            
    def _070_save(self,agent_number=0):
        self.mainQN.save(self.agentset["save_name"]+"{}.h5".format(agent_number))

    def _071_save_with_name(self,agent_number=0,name=""):
        self.mainQN.save("{}{}.h5".format(name,agent_number))
        
    def _71_targetQN_learning(self):
        self.targetQN.model.set_weights(self.mainQN.model.get_weights())

    def _ex_setTFT(self):
        self.mainQN.load_samples()
    
    
