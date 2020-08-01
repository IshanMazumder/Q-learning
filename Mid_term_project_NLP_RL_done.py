#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import re


# In[8]:


f = open(r"D:\sms.txt","r")


# In[9]:


txt = f.read()
txt = txt.lower()


# In[10]:


txt = re.sub('dwarka sector 23','xx',txt)
txt = re.sub('dwarka sector 21','yy',txt)
txt = re.sub('hauz khaas','zz',txt)
txt = re.sub('airport','ww',txt)


# In[11]:


sms = re.split('\n',txt)


# In[12]:


sms_data = pd.DataFrame(columns=['PICK','DROP','TIME'])


# In[13]:


place = ['xx','yy','zz','ww']


# In[25]:


sms = sms_data
type(sms["PICK"])


# In[ ]:


c = 0
for i in range(0,1000):
    ss = re.split(' ',sms[i])
    t = ''
    try:
        time =  (ss.index('am')-1)
        t = ss[time] + ' ' + 'am'
    except:
        time = (ss.index('pm')-1)
        t = ss[time] + ' ' + 'pm'
    
    try:
        pick = (ss.index('from')+1)
    except:
        pick = (ss.index('to')-1)
        drop = (ss.index('to')+1)
        sms_data.loc[c] = [ss[pick]] + [ss[drop]] + [str(t)]
    drop1 = [i+1 for i,d in enumerate(ss) if d == 'to']
    d1 = [ i for i in drop1 if ss[i] in place] 
    if not len(d1) == 0:
        drop1 = d1[0]
        sms_data.loc[c] = [ss[pick]] + [ss[drop1]] + [t]
    else:
        pass
        # it means didnt get the data right
    drop2 = [i+1 for i,d in enumerate(ss) if d == 'for']
    d2 = [ i for i in drop2 if ss[i] in place]
    if not len(d2) == 0:
        drop2 = d2[0]
        sms_data.loc[c] = [ss[pick]] + [ss[drop2]] + [t]
    else:
        pass
        #it means didnt get the data right
    
    print(c)
    print(sms_data.loc[c])
    c +=1


# In[26]:



def con (s):
    if s == 'xx':
        return 'dwarka sector 23'
    if s == 'yy':
        return 'dwarka sector 21'
    if s == 'zz':
        return 'hauz khaas'
    if s == 'ww':
        return 'airport'
    
sms['PICK'] = sms_data['PICK'].apply(con)
sms['DROP'] = sms_data['DROP'].apply(con)
                         


# In[ ]:


d = pd.read_csv('D:\org_df.csv')


# In[36]:



# check if all data extracted properly or not
c = 0
for i in range(0,1000):
    if d['origin'].loc[i] == sms['PICK'].loc[i] and d['dest'].loc[i] == sms['DROP'].loc[i]:
        c +=1


# In[37]:


c # all values matched :)


# In[172]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[173]:


sms_data['PICK'] = le.fit_transform(sms_data['PICK'])
sms_data['DROP'] = le.fit_transform(sms_data['DROP'])


# In[174]:


pick_list = sms_data['PICK'].tolist()
drop_list = sms_data['DROP'].tolist()


# In[ ]:


import gym
# Importing libraries
import numpy as np
import random
import math
from collections import deque
import collections
import pickle

#for text processing
import spacy
import re
import pandas as pd
env = gym.make("Taxi-v2").env


# In[ ]:


#----------Training the BOT--------------------
import random
from IPython.display import clear_output
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 10000):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")


# In[ ]:


#-----------------
"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 1000

for e in range(episodes):
    state = env.encode(random.randrange(0, 5, 1),random.randrange(0, 5, 1),pick_list[e],drop_list[e]) # state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

