import numpy as np

def _031_AI_select(GFW,episode,agent_num=0,n_player=0):
    
    last_action = GFW.players_action[1][agent_num]
    
    state_base = np.reshape(GFW.players_action[1], (1, n_player)).copy()
    state_base = np.delete(state_base,[agent_num])
    current_cooperators_degree = 1 - np.sum(state_base)/(n_player-1)

    state = np.array([current_cooperators_degree,last_action])
    
    state = np.reshape(state, (1, 2))
    
    GFW.players_action[0][agent_num] =GFW.agents[agent_num].actor.get_action(state, episode, GFW.agents[agent_num].mainQN) 
    del state_base
    del state
    del current_cooperators_degree
    del last_action
    del GFW
    
    
def _031_random_select(GFW,agent_num):
        move = np.random.choice([0, 1])
        GFW.players_action[0][agent_num]=move

        
def _031_tit_for_tat(GFW,agent_num,oppo_num):
    GFW.players_action[0][agent_num] = GFW.players_action[1][oppo_num]
    
def _031_all_defect(GFW,agent_num):
    GFW.players_action[0][agent_num]=1        
        
def _031_all_cooperate(GFW,agent_num):
    GFW.players_action[0][agent_num]=0

def _031_trigger(GFW,agent_num,oppo_num):
    last_action = GFW.players_action[1][agent_num]
    oppo_last_action = GFW.players_action[1][oppo_num]
    if last_action==0 and oppo_last_action==0:
        GFW.players_action[0][agent_num]=0
    else:
        GFW.players_action[0][agent_num]=1

def _031_WSLS(GFW,agent_num):
    last_reward = GFW.agents[agent_num].rewards[0]
    if last_reward>0:
        GFW.players_action[0][agent_num]=GFW.players_action[1][agent_num]
    else:
        GFW.players_action[0][agent_num]=1-GFW.players_action[1][agent_num]
               