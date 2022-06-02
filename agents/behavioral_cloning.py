from arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_dataset import OvercookedDataset
from state_encodings import ENCODING_SCHEMES

import numpy as np
from pathlib import Path
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
from tqdm import tqdm
import time
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

NUM_ACTIONS = 6 # UP, DOWN, LEFT, RIGHT, INTERACT, NOOP

def get_output_shape(model, image_dim):
    return model(th.rand(*(image_dim))).data.shape[1:]

def weights_init_(m):
    if hasattr(m, 'weight') and m.weight is not None and len(m.weight.shape) > 2:
        th.nn.init.xavier_uniform_(m.weight, gain=1)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, th.Tensor):
        th.nn.init.constant_(m.bias, 0)

class BehaviouralCloning(nn.Module):
    def __init__(self, visual_obs_shape, agent_obs_shape, depth=16, act=nn.ReLU, hidden_dim=256):
        '''
        :param depth: depth of cnn (i.e. num output channels
        :param act: activation fn to use
        :param kernels: kernel sizes for cnn
        :param stride: stride of kernels - should be >= 2 for larger images, 1 for small images
        :param image_dims: dimensions of images
        '''
        super(BehaviouralCloning, self).__init__()
        self.act = act
        self.hidden_dim = hidden_dim
        self.use_visual_obs = np.prod(visual_obs_shape) > 0
        assert len(agent_obs_shape) == 1
        self.use_agent_obs = np.prod(agent_obs_shape) > 0
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

        self.mlp = nn.Sequential(
            nn.Linear(self.cnn_output_shape + agent_obs_shape[0], self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, NUM_ACTIONS),
            # nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            # nn.LayerNorm(self.cnn_output_shape),
        )
        self.apply(weights_init_)

    def forward(self, obs):
        visual_obs, agent_obs = obs
        latent_state = []
        visual_obs, agent_obs = visual_obs, agent_obs
        if self.use_visual_obs:
            latent_state.append(self.cnn(visual_obs))
        if self.use_agent_obs:
            latent_state.append(agent_obs)
        logits = self.mlp(th.cat(latent_state, dim=-1))
        return logits

class BC_trainer():
    def __init__(self, env, encoding_fn, dataset, args, vis_eval=False):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.num_players = 2
        self.env = env
        self.encode_state_fn = encoding_fn
        self.dataset = dataset
        self.args = args
        self.visualize_evaluation = vis_eval
        visual_obs, agent_obss = self.encode_state_fn(self.env.mdp, self.env.state, self.args.horizon)
        visual_obs_shape = visual_obs[0].shape
        agent_obs_shape = agent_obss[0].shape
        self.players = (
            BehaviouralCloning(visual_obs_shape, agent_obs_shape).to(self.device), 
            BehaviouralCloning(visual_obs_shape, agent_obs_shape).to(self.device)
        )
        self.optimizers = tuple([th.optim.Adam(player.parameters(), lr=args.lr) for player in self.players])
        a = th.tensor(dataset.get_class_weights(), dtype=th.float32)
        self.criterion = nn.CrossEntropyLoss(weight=th.tensor(dataset.get_class_weights(), dtype=th.float32))
        if self.visualize_evaluation:
            pygame.init()
            surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx)
            self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
            self.window.blit(surface, (0, 0))
            pygame.display.flip()


    def train_on_batch(self, batch):
        metrics = {}
        batch = {k: v.to(self.device) for k,v in batch.items()}
        vo, ao, action = batch['visual_obs'].float(), batch['agent_obs'].float(), batch['joint_action'].long()
        for i in range(self.num_players):
            self.optimizers[i].zero_grad()
            pred_action = self.players[i].forward( (vo[:,i], ao[:,i]) )
            loss = self.criterion(pred_action, action[:,i])
            loss.backward()
            self.optimizers[i].step()
            metrics[f'player{i}_loss'] = loss.item()
        metrics['total_loss'] = sum(metrics.values())
        wandb.log(metrics)
        return metrics

    def evaluate(self, num_trials=1):
        STATIC_LIMIT = 3 # number of timesteps an agent can be static before they are force to pick a different action
        average_reward = []
        metrics = {'onions_pickedup': 0, 'soups_made': 0, 'dishes_pickedup': 0}
        for trial in range(num_trials):
            self.env.reset()
            obs = self.encode_state_fn(self.env.mdp, self.env.state, self.args.horizon)
            vis_obs, agent_obs = (th.tensor(o, device=self.device, dtype=th.float32) for o in obs)
            trial_reward = 0
            done = False
            prev_pos = (np.zeros( (STATIC_LIMIT, 2) ), np.zeros( (STATIC_LIMIT, 2) ))
            timestep = 0
            while not done:
                if self.visualize_evaluation:
                    self.render()
                    time.sleep(0.01)

                def select_action(idx, vo, ao, sample=False):
                    logits = self.players[idx].forward((vo.unsqueeze(dim=0), ao.unsqueeze(dim=0))).squeeze()
                    if sample:
                        return th.distributions.categorical.Categorical(logits=logits).sample()
                    action_ranking = th.argsort(logits, dim=-1)
                    pos_hist = prev_pos[idx]
                    if np.all(np.array([pos_hist[0] == pos_hist[i] for i in range(1, len(pos_hist))])):
                        # Agent has not moved in 3 turns, choose 2nd best action instead of best action
                        action = action_ranking[1]
                    else:
                        action = action_ranking[0]
                    prev_pos[idx][timestep % STATIC_LIMIT] = self.env.state.players[idx].position
                    return Action.INDEX_TO_ACTION[action]

                joint_action = tuple(select_action(i, vis_obs[i], agent_obs[i]) for i in range(2))
                next_state, reward, done, info = self.env.step(joint_action)
                print('-->', info['shaped_r_by_agent'])
                trial_reward += reward
                timestep += 1
                
                obs = self.encode_state_fn(self.env.mdp, self.env.state, self.args.horizon)
                vis_obs, agent_obs = (th.tensor(o, device=self.device, dtype=th.float32) for o in obs)
            average_reward.append(trial_reward)
        return np.mean(average_reward)

    def render(self):
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

    def train_epoch(self):
        dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for batch in tqdm(dataloader):
            self.train_on_batch(batch)

    def training(self, num_epochs=250):
        base_path = '.'
        wandb.init(project="overcooked_ai_test", entity="stephaneao", dir=base_path, mode='disabled')
        for epoch in range(num_epochs):
            mean_reward = self.evaluate()
            self.train_epoch()
            wandb.log({'evaluation_reward': mean_reward, 'epoch': epoch})

    def save(self):
        save_path = self.args.base_path / 'saved_models' / args.exp_name
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            torch.save(self.players[i].state_dict(), save_path / f'player{i}')

    def load(self, load_name):
        load_path = self.args.base_path / 'saved_models' / args.load_name
        for i in range(2):
            self.players[i].load_state_dict(torch.load(load_path / f'player{i}'))
        model.eval()


if __name__ == '__main__':
    args = get_arguments()
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.layout), horizon=args.horizon)
    dataset = OvercookedDataset(env, encoding_fn, args)
    bct = BC_trainer(env, encoding_fn, dataset, args, vis_eval=False)
    bct.training()


