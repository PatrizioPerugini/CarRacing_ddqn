import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque , namedtuple
from torchvision import transforms
import math
import warnings
import operator


class PrioritizedReplayBuffer:

    def __init__(self, 
                buffer_size=100000, 
                batch_size=8,
                compute_weights=True):
        
        #keep track of the insertions made 
        # so to know the index of the next       
        self.experience_count=0
        self.buffer_size=buffer_size
        self.batch_size = batch_size
        
        #try also to increase this
        self.pre_fill=2000

        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        #self.seed = random.seed(seed) I THINK I WON'T NEED IT
        self.compute_weights = compute_weights
        #here we store the experience
        self.experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", "done","next_state"])
        #here we store the datas associated with the experience
        self.data = namedtuple("Data", 
                        field_names=["priority", "probability",  "weight","index"])
                                #td_error     #P(i)=p^alpha/sum   #Wi=(1/N*P(i))^b
        indexes = []
        datas=[]
        for i in range(buffer_size):
            indexes.append(i)
            d = self.data(0,0,0,i)
            datas.append(d)
        
        self.memory = {key: self.experience for key in indexes}
        self.memory_data = {key: data for key,data in zip(indexes, datas)}
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    # update the priorities based on the td errors and indices
    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority**self.alpha - old_priority**self.alpha
            updated_probability = td[0]**self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index) 
            self.memory_data[index] = data

    #update of all the parameters
    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority**self.alpha
        sum_prob_after = 0
        cnt=0
        
        for element in self.memory_data.values():
            if cnt < self.experience_count:
                probability = element.priority**self.alpha / self.priorities_sum_alpha
                sum_prob_after += probability
                weight = 1
                if self.compute_weights:
                    if element.probability!=0:
                        weight = ((N *  element.probability)**(-self.beta))/self.weights_max
                    d = self.data(element.priority, probability, weight, element.index)
                    self.memory_data[element.index] = d
            cnt+=1
 
    
    def add(self, state, action, reward, done,next_state):
        #we are adding a tuple
        self.experience_count+=1
        #it's approached as a circular buffer, retrieve the index
        index = self.experience_count % self.buffer_size
        #we have to take into account when the buffer is full
        #in that case we keep track of the removed probability
        if self.experience_count > self.buffer_size:
            if index>self.buffer_size:
                index=self.buffer_size-1
            temp = self.memory_data[index]
            #remove from the sum
            self.priorities_sum_alpha -= temp.priority**self.alpha
            #if we removed the max we also need to update everithing
            if temp.priority == self.priorities_max:
                self.memory_data[index]=self.memory_data[index]._replace(priority=0)
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1))[1].priority

            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[index] =  self.memory_data[index]._replace(weight=0)
                    self.weights_max = max(self.memory_data.items(), key=operator.itemgetter(0))[1].weight
        
        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward, done,next_state)
        self.memory[index] = e
        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d
    
    def sample_batch(self):
        #of course this sampling needs to be done accordingly to probabilities
        #and we need to give only numbers of entries which are populated
        values = list(self.memory_data.values())
        
        #sample based on the probabilities
        tuples = random.choices(self.memory_data,[data.probability for data in values],k=self.batch_size)
        #tuples = random.choices(self.memory_data,k=self.batch_size)
        
        #get the indices
        indices_l=[i.index for i in tuples]
        #use a tuple
        indices=tuple(indices_l)
        #get the weights
        weights_l=[w.weight for w in tuples]
        weights=tuple(weights_l)
        states=[]
        actions=[]
        dones=[]
        rewards=[]
        next_states=[]
        #retrieve the tuples related to the indices
        for i in indices:
            e=self.memory.get(i)
            states.append(e.state)
            actions.append(e.action)
            dones.append(e.done)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            
        #return the tuple and als the indices and weights
        return tuple(states),tuple(actions),tuple(rewards),tuple(dones),tuple(next_states),indices,weights


class DQN(nn.Module):
    def __init__(self,input_shape, num_actions,env,learning_rate=1e-4):
        super().__init__()
        #input ->[batch_size,4,84,84]
        #the state is passed already normalized 
        self.input_shape = input_shape
        self.num_actions=num_actions
        self.env = env
        self.learning_rate=learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = 16
        self.conv1 = nn.Conv2d(input_shape, self.hidden_dim, kernel_size=5, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(self.hidden_dim, 32, kernel_size=5, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)

        self.linear1 = nn.Linear(32*7*7, 256)
        self.linear2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
       
        
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(conv2_out)))

        flattened = torch.flatten(conv3_out, start_dim=1)
        linear1_out = self.linear1(flattened)
        q_value = self.linear2(linear1_out)
        return q_value
        
    
    def greedy_action(self,states):
 
        qvals = self.forward(states)
    
        #just get the max of the output and return the best action
        greedy_a = torch.max(qvals, dim=-1)[1].item()
        return greedy_a
    


class Policy(nn.Module):
    continuous = False # you can change this

    def __init__(self,learning_rate=1e-4, initial_epsilon=0.6, batch_size= 8):
        super(Policy, self).__init__()
        
        self.env = gym.make('CarRacing-v2',continuous=False)#,render_mode='human')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape=4
        self.epsilon=initial_epsilon
        #used for soft update
        self.tau=0.001
        #initialize the variable of all the states so to be able to stack them 
        self.n_frames=4
        self.states = deque(maxlen=self.n_frames)
        self.next_states =deque(maxlen=self.n_frames)
        self.reset_env()
        self.initialize()
        self.initial_epsilon=initial_epsilon
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.grassed=0
        
        #instantiate the 2 networks
        #1
        self.current_q_net = DQN(self.input_shape, self.env.action_space.n,self.env) #5 actions
        self.current_q_net.to(self.device)
        #2
        self.target_q_net = DQN(self.input_shape, self.env.action_space.n,self.env)
        self.target_q_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.current_q_net.parameters(),
                                          lr=learning_rate)

        #instantiate experience buffer
        self.buffer = PrioritizedReplayBuffer()    
        
        #transformations
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))


    #SKIP THIS FUNCTION
    #brownian motion... I've just used it for some experiments, 
    #def sample_continuous_policy(action_space, seq_len, dt):
    #    actions = [action_space.sample()]
    #    for _ in range(seq_len):
    #        daction_dt = np.random.randn(*actions[-1].shape)
    #        actions.append(
    #            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
    #                    action_space.low, action_space.high))
    #    return actions
    


    #do some steps so that we are able to explore more states
    def noisy_start(self):
        for i in range(100):
            if i<40:
                s_1, r, done, _, _ = self.env.step(0)
            else:
                s_1, r, done, _, _ = self.env.step(3)
                self.states.append(s_1)
                self.next_states.append(s_1)
                self.s_0=s_1
    
    #just an utility function to set states and next_states
    def reset_env(self):
        self.s_0, _ = self.env.reset()
        for i in range(self.n_frames):
            self.states.append(self.s_0)
            self.next_states.append(self.s_0)
              
    #a new parameter is added so that also the exploration procedure can be done
    def act(self, state ,mode='exploit'):

        self.states.append(state) 
        if mode == 'explore':
            action = self.env.action_space.sample() # either [0,1,2,3,4]
        else:
            #get the action from the network. Note that here we have to
            #pass states which are preprocessed, and this is done now
            action = self.current_q_net.greedy_action(self.preproc_state(torch.from_numpy(np.array(self.states))).unsqueeze(0).to(self.device))
        
        return action
    
    def make_step(self,action):
        #simulate action
        s_1, r, done, _, _ = self.env.step(action)
        #append the state to next states
        self.next_states.append(s_1)

        cs = torch.from_numpy(s_1)
        #this is the pixel in front of the car
        pixel=cs[64][48][1].item()
        #we are on the grass
        if pixel>150:
            r-=5
            self.grassed+=1
        else:
            #reset if we are back on track
            self.grassed=0
        #some artificial rewards
        if action==3:
            r+=0.7
        if action==0:
            r-=3

        #big penalty if we are on the grass for too long and episode terminates
        #so that our buffer is not filled with useless states
        if self.grassed>40:
            done=True
            #big penalty
            r-=5
            self.grassed=0
        

        #put experience in the buffer
        self.buffer.add(self.states, action, r, done, self.next_states)
        
        self.rewards += r
        #update the current state
        self.s_0 = s_1.copy()                
        self.step_count += 1
        #if we are going too much on the
        # grass we start over 

        if done:
            self.reset_env()
            print("done")
                        
        return done

    def train(self,gamma=0.9, max_episodes=700,
              network_update_frequency=30,
              network_sync_frequency=700):
        
        #self.load()
        
        self.gamma=gamma
        self.loss_function = nn.MSELoss()
        self.s_0, _ = self.env.reset()
        cnt=0
        #self.update_target_q_net()
        #here we perform the soft update
        self.soft_update(self.current_q_net,self.target_q_net,self.tau)
        #at most these steps are performed
        max_dones=2000
        #populate the buffer with random actions
        while cnt<self.buffer.pre_fill:
            action=self.act(self.s_0,mode='explore')
            self.make_step(action)
            cnt+=1
        print("have been inserted")
        print(self.buffer.experience_count)
        #buffer is populated
        print("starting training")
        for ep in range(max_episodes):
            #self.reset_env()
            self.noisy_start()
            done = False
            cc=0
            while not done and cc < max_dones:
                cc+=1
                prob = np.random.random()
                #exploration vs exploitation
                if prob < self.initial_epsilon:
                    action=self.act(self.s_0,mode='explore')
                    done=self.make_step(action)
                else:
                    action=self.act(self.s_0)
                    done=self.make_step(action)
                #update the network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                #sync the network
                if self.step_count % network_sync_frequency == 0:
                    #self.update_target_q_net()
                    #perform soft update of the weights
                    self.soft_update(self.current_q_net,self.target_q_net,self.tau)
                if done or cc >= max_dones :
                    #decrease epsilon
                    if self.epsilon >= 0.1:
                        self.epsilon = self.epsilon * 0.993
                        #self.epsilon=self.epsilon_min + (self.epsilon - self.epsilon_min) * 
                        # np.exp(-self.epsilon_decay * decay_step)
                    if len(self.update_loss) == 0:
                        self.training_loss.append(0)
                    else:
                        self.training_loss.append(np.mean(self.update_loss))
                    #print some stuff
                    print("episode number",ep)
                    print("current epsilon is ",self.epsilon)
                    print("episode reward is",self.rewards)
                    print("loss is ",self.training_loss[0])
                    #self.rewards=0
                    self.training_loss = []
                    self.update_loss=[]
                    #save the parameters
                    self.save()
                    self.rewards=0
            #update network parameters
            self.buffer.update_parameters()


        return
    def update(self):
        self.optimizer.zero_grad()
        #get the batch
        batch=self.buffer.sample_batch()

        #now I have 4 states and the action reward that lead me to next state 
        #note that we have to work in this way to give a context to the network

        #batch_size*[ 4states,action,reaward,done,4next_states] indices and weights

        states, actions, rewards, dones, next_states, indices,weights= list(batch)
        
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(self.device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(self.device)

        #we need to manipulate the states in order to be able to feed them to the model 
        states = self.from_tuple_to_tensor(states)
        next_states = self.from_tuple_to_tensor(next_states)
    
        #get the q-values for the current network
        #we'll see below that states and next states are 
        #normalized so that they can directly be given to the network
        qvals = self.current_q_net(states)
        print(actions)    
        print(qvals.shape)
        input()
        qvals = torch.gather(qvals, 1, actions)
        print(qvals.shape)
        input()
        #get the q-values for the next state using the target network
        
        next_qvals = self.target_q_net(next_states)   
        next_qvals_max = torch.max(next_qvals, dim=-1)[0].reshape(-1, 1) 
        target_qvals = rewards + (1 - dones)*self.gamma*next_qvals_max

        loss = self.loss_function(qvals, target_qvals)
                                #output  #expected value
        #note that the core of double DQN is in the lines of code above and 
        # in the particular interaction between the two networks

        if self.buffer.compute_weights:
            with torch.no_grad():
                #multiply the weights as indicated in the algorithm
                #this should act as a correction factor for the bias 
                #we introduced with importance sampling
                weight=sum(np.multiply(weights,loss.data.cpu().numpy()))
            loss*=weight

        #clamp - this actually never happens but since in the first epochs 
        # training is unstable is just a precaution..
        loss=torch.clamp(loss,min=0,max=700)

        #backward and step
        loss.backward()
        self.optimizer.step()
        self.update_loss.append(loss.item())
        
        #update priorities based on the td error 
        td_error = abs(target_qvals.detach() - qvals.detach()).cpu().numpy()
        #update our priorities, we need the indices to keep track 
        # of the updates that we are doing
        self.buffer.update_priorities(td_error,indices)
                

    def save(self,model="model.pt"):
        torch.save(self.state_dict(), model)

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
   

    ###SOFT UPDATE OF THE WEIGHTS
    def soft_update(self, local_model, target_model, tau):
       
        #θ_target = τ*θ_local + (1 - τ)*θ_target

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    #alternatively just copy the model 
    def update_target_q_net(self):
        self.target_q_net.load_state_dict(self.current_q_net.state_dict())
    
    def initialize(self):
        
        self.training_loss = []
        self.update_loss = []
        self.rewards = 0
        self.step_count = 0
    
    def preproc_state(self, state):
            #state is [4,96,96,3] and this should return [4,84,84]
            # State Preprocessing
             #self.rs = transforms.Resize((84,84))
            #[batch,num_frames=4,img_H=84,img_W=84]
            state = state.permute(0,3,1,2) #Torch wants images in format (channels, height, width)
            transform=transforms.ToPILImage()
            out = torch.zeros((self.n_frames,84,84))
            for i in range(self.n_frames):
                out[i,:]=torch.from_numpy(np.array(self.rs(self.gs(transform(state[i,:])))))     
                
            return out/255 # normalize

    def from_tuple_to_tensor(self,tuple_of_np):
         #[32, 4, 96, 96, 3 ])
        a=tuple_of_np[0][0].shape[0]
        b=tuple_of_np[0][0].shape[1]
        c=tuple_of_np[0][0].shape[2]
        tensor = torch.zeros(len(tuple_of_np),len(tuple_of_np[0]),a,b,c)
        for j in range(len(tuple_of_np)):
            for i, x in enumerate(tuple_of_np[j]):
                tensor[j,i,:] = torch.FloatTensor(x)
        #preprocess the states
        norm_t=torch.stack([self.preproc_state(state).float() for state in tensor]).to(self.device)
        #norm_t.shape=[32,4,84,84]
        
        return norm_t
