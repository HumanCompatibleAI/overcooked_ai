import sys
sys.path.append("/home/biswas/overcooked_ai/src/")
from arguments import get_arguments
from overcooked_dataset import Subtasks
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_dataset import OvercookedDataset, Subtasks
from state_encodings import ENCODING_SCHEMES
from sklearn.metrics import accuracy_score

from copy import deepcopy
import numpy as np
from pathlib import Path
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
from tqdm import tqdm
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import wandb
import tensorflow as tf

def get_output_shape(model, image_dim):
    return model(th.rand(*(image_dim))).data.shape[1:]

def weights_init_(m):
    if hasattr(m, 'weight') and m.weight is not None and len(m.weight.shape) > 2:
        th.nn.init.xavier_uniform_(m.weight, gain=1)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, th.Tensor):
        th.nn.init.constant_(m.bias, 0)

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

class SubtaskPredictor(nn.Module):
    def __init__(self, device, visual_obs_shape, agent_obs_shape, depth=16, act=nn.ReLU, hidden_dim1=512, hidden_dim2=256):
        super(SubtaskPredictor, self).__init__()
        self.device = device
        self.act = act
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.use_visual_obs = np.prod(visual_obs_shape) > 0
        assert len(agent_obs_shape) == 1
        self.use_agent_obs = np.prod(agent_obs_shape) > 0
        self.subtasks_obs = Subtasks.NUM_SUBTASKS 
    
        # Define CNN for grid-like observations
        if self.use_visual_obs:
            self.kernels = (4, 4, 4, 4) if max(visual_obs_shape) > 64 else (3, 3, 3, 3)
            self.strides = (2, 2, 2, 2) if max(visual_obs_shape) > 64 else (1, 1, 1, 1)
            self.padding = (1, 1)
            layers = []
            current_channels = visual_obs_shape[0]
            for i, (k, s) in enumerate(zip(self.kernels, self.strides)):
                layers.append(nn.Conv2d(current_channels, depth, k, stride=s, padding=self.padding))
                layers.append(nn.GroupNorm(1, depth))
                layers.append(self.act())
                current_channels = depth
                depth *= 2

            layers.append(nn.Flatten())
            self.cnn = nn.Sequential(*layers)
            self.cnn_output_shape = get_output_shape(self.cnn, [1, *visual_obs_shape])[0]
            self.pre_flatten_shape = get_output_shape(self.cnn[:-1], [1, *visual_obs_shape])
        else:
            self.cnn_output_shape = 0

        print(self.cnn_output_shape + agent_obs_shape[0] +self.subtasks_obs+ 10*self.subtasks_obs+ Subtasks.NUM_SUBTASKS)
        # Define layers
        self.rnn = nn.Sequential(
            nn.Linear(self.cnn_output_shape + agent_obs_shape[0] +self.subtasks_obs+ 10*self.subtasks_obs+ Subtasks.NUM_SUBTASKS, self.hidden_dim1),
            # nn.Dropout(),
            act(),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.Dropout(),
            act(),
        )
        #fully connected layer for prediction output
        self.fc =  nn.Linear(self.hidden_dim2, Subtasks.NUM_SUBTASKS)
        #initialize weights
        self.apply(weights_init_)
        self.to(self.device)
    
    def forward(self, obs, past_obs, hidden):
        visual_obs, agent_obs, subtask= obs
        latent_state = []
        if self.use_visual_obs:
            # Convert all grid-like observations to features using CNN
            latent_state.append(self.cnn(visual_obs))
        if self.use_agent_obs:
            latent_state.append(agent_obs)
        # add current subtask observation
        latent_state.append(subtask)
        # add past history of observations
        latent_state.append(past_obs)
        # add last hidden state
        latent_state.append(hidden)
        inp = th.cat(latent_state, dim=-1)
        # Passing in the input added to hidden state into rnn and obtaining outputs
        hidden = self.rnn(inp)
        
        # pass through fc to get logits
        out = self.fc(hidden)

        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = th.zeros(batch_size, Subtasks.NUM_SUBTASKS)
        hidden[:,11]  = 1
        return hidden
    
class SP_trainer():
    def __init__(self, env, encoding_fn, dataset, args):
        self.env = env
        self.encode_state_fn = encoding_fn
        self.remove_list = []
        self.dataset = dataset
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.num_players = 2
        self.env = env
        self.encode_state_fn = encoding_fn
        self.args = args
        visual_obs, agent_obss = self.encode_state_fn(self.env.mdp, self.env.state, self.args.horizon)
        visual_obs_shape = visual_obs[0].shape
        agent_obs_shape = agent_obss[0].shape
        self.players = (
            SubtaskPredictor(self.device, visual_obs_shape, agent_obs_shape),
            SubtaskPredictor(self.device, visual_obs_shape, agent_obs_shape)
        )
        self.hidden = [self.players[0].init_hidden(args.batch_size), self.players[1].init_hidden(args.batch_size)]
        #define optimizers and loss function
        if dataset is not None:
            self.optimizers = tuple([th.optim.Adam(player.parameters(), lr=args.lr) for player in self.players])
            self.subtask_criterion = nn.CrossEntropyLoss(weight=th.tensor(dataset.get_subtask_weights(), dtype=th.float32, device=self.device),
                                                             reduction='none')
  
    
    def train_on_batch(self, batch):
        """
        Train BC agent on a batch of data
        """
        batch = {k: v.to(self.device) for k,v in batch.items()}
        vo, ao, action, subtasks = batch['visual_obs'].float(), batch['agent_obs'].float(), \
                                   batch['joint_action'].long(), batch['subtasks'].long()

        curr_subtask, next_subtask = subtasks[:, 0], subtasks[:, 1]
        metrics = {}
        
        
        for i in range(self.num_players):
            self.optimizers[i].zero_grad()
            cs_i = F.one_hot(curr_subtask[:,i], num_classes=Subtasks.NUM_SUBTASKS)
            sub_history = []
            # create subtask history of past 10 subtasks
            for idx, sample in enumerate(cs_i):
                past_history = []
                if idx>10:
                    # append past 10 subtasks 
                    for ids in range(idx-10,idx):
                        past_history.append(cs_i[ids])
                else:
                    for ids in range(0,10):
                        past_history.append( [0]*12)
                        past_history[ids][11]=1
                #flatten
                past_history = [item for sublist in past_history for item in sublist]
                
                past_history = th.FloatTensor(past_history)
                sub_history.append(past_history)
            his = th.empty((batch['agent_obs'].size()[0],120))
            for idx,sample in enumerate(sub_history):
                his[idx]= sample
            
            # Train on subtask prediction task
            pred_subtask = self.players[i].forward( (vo[:,i], ao[:,i], cs_i), his, self.hidden[i] )
            tot_loss = 0
            # update hidden state for this player
            self.hidden[i] = pred_subtask
            subtask_loss = self.subtask_criterion(pred_subtask, next_subtask[:, i])
            loss_mask = th.logical_or(action[:,i] == Action.ACTION_TO_INDEX[Action.INTERACT],
                                        th.rand_like(subtask_loss, device=self.device) > 0.9)
            subtask_loss = th.mean(subtask_loss*loss_mask)
            # calculate total loss
            tot_loss += th.mean(subtask_loss)
            metrics[f'p{i}_subtask_loss'] = subtask_loss.item()
            # get best action from softmax output
            pred_subtask_indices = th.softmax(pred_subtask, dim=1)
            pred_subtask_indices = th.argmax(pred_subtask_indices, dim=1)
            metrics[f'p{i}_subtask_acc'] = accuracy_score(next_subtask[:,i], pred_subtask_indices)
            tot_loss.backward()
            self.optimizers[i].step()
        return metrics

    def evaluate_on_batch(self, batch):
        """
        Evaluate agent on 
        :param batch: The dataset we want to test on
        :return: fraction of correctly classified samples out of the whole batch
        """
        batch = {k: v.to(self.device) for k,v in batch.items()}
        vo, ao, action, subtasks = batch['visual_obs'].float(), batch['agent_obs'].float(), \
                                   batch['joint_action'].long(), batch['subtasks'].long()

        curr_subtask, next_subtask = subtasks[:, 0], subtasks[:, 1]
        metrics = {}
        for i in range(self.num_players):
            cs_i = F.one_hot(curr_subtask[:,i], num_classes=Subtasks.NUM_SUBTASKS)
            sub_history = []
            # create subtask history of past 10 subtasks
            for idx, sample in enumerate(cs_i):
                past_history = []
                if idx>10:
                    # append past 10 subtasks 
                    for ids in range(idx-10,idx):
                        past_history.append(cs_i[ids])
                else:
                    for ids in range(0,10):
                        past_history.append( [0]*12)
                #flatten
                past_history = [item for sublist in past_history for item in sublist]
                
                past_history = th.FloatTensor(past_history)
                sub_history.append(past_history)
            his = th.empty((batch['agent_obs'].size()[0],120))
            for idx,sample in enumerate(sub_history):
                his[idx]= sample
            pred_subtask = self.players[i].forward( (vo[:,i], ao[:,i], cs_i), his, self.hidden[i] )
            #flatten
            self.hidden[i] = pred_subtask
            # get best action from softmax output
            pred_subtask_indices = th.softmax(pred_subtask, dim=1)
            pred_subtask_indices = th.argmax(pred_subtask_indices, dim=1)
            metrics[f'p{i}_subtask_acc_evaluation'] = accuracy_score(next_subtask[:,i], pred_subtask_indices)
        return metrics

    def evaluate(self, num_trials):
        dataloader = DataLoader(self.dataset, batch_size=num_trials, shuffle=True, num_workers=4)
        batch = next(iter(dataloader))
        eval_acc = self.evaluate_on_batch(batch)
        return eval_acc

    def train_epoch(self):
        metrics = {}
        for i in range(2):
            self.players[i].train()
        dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for batch in tqdm(dataloader):
            # update hidden state at beginning of batch for each player
            for i in range(self.num_players):
                self.hidden[i] = self.players[i].init_hidden( batch['agent_obs'].size()[0])
            # log losses for each player for this epoch 
            new_losses = self.train_on_batch(batch)
            metrics = {k: [new_losses[k]] + metrics.get(k, []) for k in new_losses}
            length = batch['agent_obs'].size()[0]
        # calculate evaluation accuracy 
        eval_acc = self.evaluate(length)
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        metrics['total_loss'] = sum([v for k, v in metrics.items() if 'loss' in k])
        metrics = merge_two_dicts(metrics, eval_acc)
        return metrics

   
        
    def training(self, num_epochs=500):
        """ Training routine """
        wandb.init(project="overcooked_ai_subtaskpred", entity="ubiswas", dir=str(args.base_dir / 'wandb'))#, mode='disabled')
        best_loss = float('inf')
        best_reward = 0
        for epoch in range(num_epochs):
            # log losses and accuracies for each epoch 
            metrics = self.train_epoch()
            wandb.log({'epoch': epoch, **metrics})
            if metrics['total_loss'] < best_loss:
                print(f'Best loss achieved on epoch {epoch}, saving models')
                self.save(tag='best_loss')
                best_loss = metrics['total_loss']
            


    def save(self, tag=''):
        save_path = self.args.base_dir / 'saved_models' / f'{args.exp_name}_{tag}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            th.save(self.players[i].state_dict(), save_path / f'player{i}')

    def load(self, load_name='default_exp_225'):
        load_path = self.args.base_dir / 'saved_models' / load_name
        for i in range(2):
            self.players[i].load_state_dict(th.load(load_path / f'player{i}', map_location=self.device))


if __name__ == '__main__':
    args = get_arguments()
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.layout), horizon=args.horizon)
    dataset = OvercookedDataset(env, encoding_fn, args)
    spt = SP_trainer(env, encoding_fn, dataset, args)
    spt.training()



