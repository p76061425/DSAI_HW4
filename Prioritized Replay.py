import os
import gym
import numpy as np
import random
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


class SumTree(object):
    data_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object) 

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data 
        self.update(tree_idx, p) 

        self.data_pointer += 1
        if self.data_pointer >= self.capacity: 
            self.data_pointer = 0
    
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:   
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1 
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  
                leaf_idx = parent_idx
                break
            else:  
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        P = self.tree[0]
        return P

class Memory(object): 
    epsilon = 0.01 
    alpha = 0.6  
    beta = 0.4  
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,7)), np.empty((n, 1))
        pri_seg = self.tree.total_p() / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p()
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            
            b_idx[i] = idx
            b_memory[i,:] = data[:]

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Prioritized_Replay:
    def __init__(self, env, memory_size):
        self.env     = env
        self.memory  = Memory(memory_size)
        self.batch_size = 32
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.ob_size = self.env.observation_space.shape[0]
        self.model        = self.create_model()
        self.target_model = self.create_model()

        
    def create_model(self):
        def weight_loss_wrapper(isweight):
            def weight_loss(target, prdict):
                return K.mean(K.square(target - prdict) * isweight)
            return weight_loss
        
        inputs = Input(shape=(self.ob_size,))
        weights = Input(shape=(1,))
        l1   = Dense(24, activation='relu')(inputs)
        l2   = Dense(48, activation='relu')(l1)
        l3   = Dense(self.env.action_space.n)(l2)
        
        model = Model([inputs, weights], l3)
        model.compile(loss=weight_loss_wrapper(weights),optimizer=Adam(lr=self.learning_rate)) 

        return model
        
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict( [state,np.ones((1, 1))] )[0])

    def remember(self, cur_state0, cur_state1, action,reward, new_state0, new_state1, done):
        self.memory.store([cur_state0, cur_state1, action,reward, new_state0, new_state1, done])
 
    def replay(self):
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        state = batch_memory[:,:2]
        new_state = batch_memory[:,4:6]
        
        pridict_Q = self.model.predict(   [new_state,np.ones((self.batch_size, 1))]   )
        max_act = np.empty((self.batch_size))
        for i in range(self.batch_size):
            max_act[i] = np.argmax(pridict_Q[i,:])
        
        train_targets = self.target_model.predict( [state,np.ones((self.batch_size, 1))] )
        Q_future = self.target_model.predict( [new_state,np.ones((self.batch_size, 1))]  )
        targets = np.zeros((self.batch_size))
        abs_errors = np.zeros((self.batch_size))
        
        for idx,sample_q in enumerate(Q_future):
            reward = batch_memory[idx,3]
            action = batch_memory[idx,2]
            max_idx = int(max_act[idx])
            max_Q_future = sample_q[max_idx]
            targets[idx] = reward + self.gamma*max_Q_future
            train_targets[idx,int(action)] = reward + self.gamma*max_Q_future
        
        self.model.fit( [state,ISWeights]  , train_targets, epochs=1, verbose=0)
        
    
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
 
    def start(self,cur_state0, cur_state1, action,reward, new_state0, new_state1, done):
        self.remember(cur_state0, cur_state1, action,reward, new_state0, new_state1, done)
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        self.replay()        
        self.target_train() 

def sd_cal(reward_data,loop,T):
    mean_list = []
    sd_list = []
    mean = 0
    
    for t in range(T):
        mean = 0
        for file in range(loop):
            mean += reward_data[file][t]
        mean = mean/loop
        mean_list.append(mean)
        
        sd = 0
        for file in range(loop):
            sd += np.square(reward_data[file][t] - mean)
        sd = np.sqrt(sd/10)
        sd_list.append(sd)
    return mean_list, sd_list
    
def sd_max_min(mean_list,sd_list,T):
    sd_max = []
    sd_min = []
    for i in range(T):
        sd_max.append(mean_list[i]+ sd_list[i])
        sd_min.append(mean_list[i]- sd_list[i])
    return sd_max,sd_min
    
def plot_show(loop,T,PR,rdm,PRps):
    
    if(PR):
        PR_reward_data = []
        for i in range(loop):
            with open('reward/PR_reward'+str(i)+'.pickle', 'rb') as PR_R_f:
                PR_reward = pickle.load(PR_R_f)
                PR_reward_data.append(PR_reward)
    
    
    if(rdm):
        rdm_reward_data = []
        for i in range(loop):
            with open('reward/rdm_reward'+str(i)+'.pickle', 'rb') as rdm_R_f:
                rdm_reward = pickle.load(rdm_R_f)
                rdm_reward_data.append(rdm_reward)

    if(PRps):
        PR_reward_ps_data = []
        for i in range(loop):
            with open('reward/PR_reward'+str(i)+'per_step'+'.pickle', 'rb') as PR_R_f:
                PR_reward_ps = pickle.load(PR_R_f)
                PR_reward_ps_data.append(PR_reward_ps)
    
                
    if(PR):            
        PRmean_list,PRsd_list = sd_cal(PR_reward_data,loop,T)
        PRsd_max,PRsd_min = sd_max_min(PRmean_list,PRsd_list,T)
        
    if(rdm):
        rdm_mean_list,rdm_sd_list = sd_cal(rdm_reward_data,loop,T)
        rdm_sd_max,rdm_sd_min = sd_max_min(rdm_mean_list,rdm_sd_list,T)
        
    if(PRps):
        PRps_mean_list,PRpssd_list = sd_cal(PR_reward_ps_data,loop,T)
        PRps_sd_max,PRps_sd_min = sd_max_min(PRps_mean_list,PRpssd_list,T)
    
    
    Episode_list = list( range(T) )
    if(PR):
        plt.plot(Episode_list ,PRmean_list ,label='PrioritizedReplay')
        plt.fill_between(Episode_list,PRsd_max,PRsd_min,facecolor = "blue", alpha= 0.3)
    
    if(rdm):
        plt.plot(Episode_list ,rdm_mean_list ,label='Random')
        plt.fill_between(Episode_list,rdm_sd_max,rdm_sd_min,facecolor = "red", alpha= 0.3)
    
    if(PRps):
        plt.plot(Episode_list ,PRps_mean_list ,label='PrioritizedReplay')
        plt.fill_between(Episode_list,PRps_sd_max,PRps_sd_min,facecolor = "blue", alpha= 0.3)
    
    
    plt.xlabel('Episode')
    
    str_y = "Total Reward"
    if(PRps):
        str_y = "Reward per step"
    plt.ylabel(str_y)
    
    plt.legend(loc='lower right')
    plt.show()
    
    
def save_reward_data(loop_num,PR_reward_list,rdm_reward_list,PR_reward_ps_list):
    path = "reward/"
    if not os.path.isdir(path):
        os.mkdir(path)
    
    if(PR_reward_list != False):
        with open('reward/PR_reward'+str(loop_num)+'.pickle', 'wb') as f0:
            pickle.dump(PR_reward_list,f0)

    if(rdm_reward_list != False):            
        with open('reward/rdm_reward'+str(loop_num)+'.pickle', 'wb') as f2:
            pickle.dump(rdm_reward_list,f2)
        
    if(PR_reward_ps_list != False):        
        with open('reward/PR_reward'+str(loop_num)+'per_step'+'.pickle', 'wb') as f1:
            pickle.dump(PR_reward_ps_list,f1)

if __name__ == "__main__":
    
    T = 100
    loop = 10
    for i in range(loop):
        print("loop:",i)
        MEMORY_SIZE = 4000
        env     = gym.make("MountainCar-v0")
        env     = env.unwrapped
        RL = Prioritized_Replay(env,MEMORY_SIZE)
        T = 100
        learning_step = 500
        
        total_steps = 0
        PR_reward_list = []
        PR_reward_ps_list = []
        for t in range(T):
            cur_state = RL.env.reset().reshape(1,2)
            step = 0
            total_R = 0
            while(True):
                done = False
                action = RL.act(cur_state)
                new_state, reward, done, _ = RL.env.step(action)
                RL.env.render()  
                
                pos = (new_state[0] + 1.2) / 0.9 - 1 
                vel = abs(new_state[1]) / 0.035 -1 
                reward = pos + vel
                if done:
                    reward = 1
               
                total_R += reward
                new_state = new_state.reshape(1,2)                  
                RL.remember(cur_state[0][0],cur_state[0][1], action, reward, new_state[0][0],new_state[0][1], done)
                
                if total_steps > MEMORY_SIZE:
                    
                    RL.start(cur_state[0][0],cur_state[0][1], action, reward, new_state[0][0],new_state[0][1], done)
                
                cur_state = new_state
                total_steps += 1
                step += 1 
                
                if (step == learning_step):
                    done = 1
                
                if done:
                    print("t:",t,"steps:",step,"total_R:",total_R,"R_per_step:",total_R/step)
                    PR_reward_list.append(total_R)
                    PR_reward_ps_list.append(total_R/step)
                    step = 0
                    break
    #    RL.env.close()
    #
    #    env_2     = gym.make("MountainCar-v0")
    #    env_2     = env_2.unwrapped
    #    total_steps = 0
    #    rdm_reward_list = []
    #    for t in range(T):
    #        cur_state = env_2.reset().reshape(1,2)
    #        step = 0
    #        total_R = 0
    #        while(True):
    #            env_2.render()  
    #            done = False
    #            action = env_2.action_space.sample()             
    #            new_state, reward, done, _ = env_2.step(action)       
    #            env_2.render()    
    #             
    #            pos = (new_state[0] + 1.2) / 0.9 - 1 
    #            vel = abs(new_state[1]) / 0.035 -1 
    #            
    #            reward = pos + vel
    #            if done:
    #                reward = 1
    #            
    #            total_R += reward
    #            cur_state = new_state
    #            total_steps += 1
    #            step += 1 
    #            
    #            if (step == learning_step):
    #                done = 1
    #            
    #            if done:
    #                print("t:",t,"steps:",step,"total_R:",total_R)
    #                rdm_reward_list.append(total_R/step)
    #                step = 0
    #                break
    #    env_2.close()        
        
        save_reward_data(i,PR_reward_list,False,False)
        
    plot_show(loop,T,True,False,False)
   