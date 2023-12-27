import numpy as np
from collections import deque
import pickle
from keras import backend as K

# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)
    
    def to_nparray(self):
        return np.array([self.buffer[i] for i in range(self.len())])
    
    def show(self):
        print([self.buffer[i] for i in range(self.len())])
    
    def save(self,object_num=0):
        with open('./data/memory/memory{}.pickle'.format(object_num), 'wb') as p:
            pickle.dump(self.buffer, p)
        
        # with open('memory{}.pickle'.format(object_num), 'rb') as p:
        #     l = pickle.load(p)
        #     print(l)


# 状態に応じて、行動を決定するクラス
class Actor:
    def get_action_old(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)
        #epsilon = 0
        if epsilon <= np.random.uniform(0, 1):
            return_TargetQs = mainQN.model.predict(state,verbose=0)[0]
            K.clear_session()
            action = np.argmax(return_TargetQs)  # 最大の報酬を返す行動を選択する
            #print("|state{}|-----|prediction{}|action{}|".format(state,return_TargetQs,action))
        else:
            action = np.random.choice([0, 1])  # ランダムに行動する
        return action
    
    def get_action(self,state,episode,mainQN):
        # 徐々に最適行動のみをとる、Boltzmann方策
        temperature = 1/np.log(episode+1.1)
        return_TargetQs = mainQN.model.predict(state,verbose=0)[0]
        K.clear_session()
        probabilities = np.exp(return_TargetQs / temperature) / np.sum(np.exp(return_TargetQs / temperature))
        try:
            action = np.random.choice(range(len(return_TargetQs)), p=probabilities)
        except:
            action = self.get_action_old(state=state, episode=episode,mainQN=mainQN)
            print("E",end="")
            # exit(1)
        # print("|state{}|-----|prediction{}|action{}|".format(state,return_TargetQs,action))
        return action