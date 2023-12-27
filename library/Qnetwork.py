import tensorflow as tf
# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.environments import suite_gym
# from tensorflow.keras.utils import plot_model
import numpy as np
from .Qlearning_agent_settings import Actor
import pandas as pd
from keras import backend as K

class QNetwork:
    def __init__(self,learning_rate,name="",input_size = 2):
        # Initialize model
        self.model = []
        self.lr = learning_rate
        if len(name)>0:
            self.load_model(name)
        else:
            self.create_model(input_size)

    def load_model(self,name):
        self.model = tf.keras.models.load_model(name)
        
    def create_model(self,input_size):
        #    adam_learning_rate: 0.0003436452344013588
        # n_layers: 9
        # n_units_l0: 39
        # n_units_l1: 16
        # n_units_l2: 10
        # n_units_l3: 49
        # n_units_l4: 23
        # n_units_l5: 112
        # n_units_l6: 34
        # n_units_l7: 36
        # n_units_l8: 56
        # optimizer: Adam
        # weight_decay: 1.3945963806990511e-08

        optimizer = tf.keras.optimizers.Adam(self.lr)
        loss_fn = tf.keras.losses.Huber()
        # ----------Define input layer-------------#
        # the inputs are only my cooperating rate and enemy's cooperating rate.
        # p1, p2, flattened_payoff_matrix(8 factors)
        inputs = tf.keras.Input(input_size,)
        # ----------Define hidden layers-----------#
        x = tf.keras.layers.Dense(39)(inputs)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        for i in [16,10,49,23,11,34,36,56]:            
            x = tf.keras.layers.Dense(i)(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        # ----------Define output layer------------#
        # the output is up or down of my cooperating rate
        outputs = tf.keras.layers.Dense(2)(x)
        # Define model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=loss_fn, optimizer=optimizer)


    def replay_checker(self, memory, batch_size, gamma, targetQN):
        target1 = self.replay_new(memory, batch_size, gamma, targetQN)
        # target1 = self.replay_old(memory, batch_size, gamma, targetQN)
        target2 = self.replay_old(memory, batch_size, gamma, targetQN)
        print((target1-target2).mean())
        exit(1)

        # Learning the weight of the network
    def replay(self, memory, batch_size, gamma, targetQN):
        length_state = 2
        states = np.zeros((batch_size, length_state))
        next_states = np.zeros((batch_size, length_state))
        rewards = np.zeros((batch_size,1))
        actions = np.zeros(batch_size, dtype=np.int64)
        targets = np.zeros((batch_size, 2))
        rew_preds = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)
        print("Learning",end="",flush=True)
        # Using enumerate, we can get i value contemporary state and action.
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            #print("state:{}|action:{}|reward:{}|next:{}|".format(state_b, action_b, reward_b, next_state_b))
            # state_b has 2/2 dimension.
            states[i] = state_b
            next_states[i] = next_state_b
            rewards[i] = reward_b
            actions[i] = action_b
            
        # up or down から最大の報酬を返す行動を選択する
        retmainQs = self.model.predict(next_states,verbose=0)
        next_actions = np.argmax(retmainQs, axis = 1) 
        targets =  self.model.predict(states,verbose=0)
        
        rew_preds = rewards + gamma * targetQN.model.predict(next_states,verbose=0)
        K.clear_session()
        #rew_preds = rewards
        for i in range(batch_size):
            targets[i][actions[i]] = rew_preds[i][actions[i]]
            #targets[i][actions[i]] = rew_preds[i]
            
            if targets[i].mean() >1000 or targets[i].mean() < -1000:
                print("reward is too big. learning stop")
                exit(1)
            # if i%10==0:print("-",end="",flush=True)
        self.model.fit(states,targets, epochs=30, verbose=0)  # epochsは訓練データの反復回
        K.clear_session()
    
    def save(self,name = "./data/SimpleGame3.h5"):
        self.model.save(name)

    def calculate_accuracy(self,x_test,y_test):
        accuracy = tf.keras.metrics.BinaryAccuracy()
        # make predictions with the model
        y_pred = self.model.predict(x_test)
        # update the metric with the predictions and targets
        accuracy.update_state(y_test, y_pred)
        # get the current value of the metric
        acc_value = accuracy.result().numpy()
        return acc_value
    
    def load_samples(self):
        df = pd.read_csv("./data/sampledata.csv")
        inputs = np.stack([df["myCrate"],df["opoCrate"]],1)
        targets = np.stack([df["target_up"],df["target_down"]],1)
        self.model.fit(inputs,targets, epochs=100, verbose=0)  # epochsは訓練データの反復回
    
    def pred_test(self):
        inputs = np.random.rand(10,2)
        pred = self.model.predict(inputs,verbose=0)
        print(pred)
