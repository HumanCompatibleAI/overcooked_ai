from arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


import numpy as np
from tqdm import tqdm
import torch as th
import torch.nn as nn

NUM_ACTIONS = 6 # UP, DOWN, LEFT, RIGHT, INTERACT, NOOP

def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape[1:]

def weights_init_(m):
    if hasattr(m, 'weight') and m.weight is not None and len(m.weight.shape) > 2:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
    if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
        torch.nn.init.constant_(m.bias, 0)

class BehaviouralCloning(nn.Module):
    def __init__(self, robot_obs_size=9, visual_obs_size=(8, 8), depth=16, act=nn.ReLU, num_channels=3,
                 hidden_dim=1024):
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
        self.kernels = (4, 4, 4, 4) if max(visual_obs_size) > 64 else (3, 3, 3, 3)
        self.strides = (2, 2, 2, 2) if max(visual_obs_size) > 64 else (1, 1, 1, 1)
        self.padding = (1, 1)
        layers = []
        current_channels = num_channels
        for i, (k, s) in enumerate(zip(self.kernels, self.strides)):
            layers.append(nn.Conv2d(current_channels, depth, k, stride=s, padding=self.padding))
            layers.append(nn.GroupNorm(1, depth))
            layers.append(self.act())
            current_channels = depth
            depth *= 2

        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        self.output_shape = get_output_shape(self.cnn, [1, 3, *visual_obs_size])[0]
        self.pre_flatten_shape = get_output_shape(self.cnn[:-1], [1, 3, *visual_obs_size])

        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape + robot_obs_size, self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act(),
            nn.Linear(self.hidden_dim, NUM_ACTIONS),
            # nn.utils.spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim)),
            # nn.LayerNorm(self.output_shape),
        )
        self.apply(weights_init_)

    def forward(self, obs):
        visual_obs, robot_obs = obs
        latent_state = [self.cnn(visual_obs), robot_obs]
        logits = self.mlp(torch.cat(latent_state, dim=-1))
        return logits

    def get_action(self, obs):
        logits = self.forward(obs)
        values, action_idx = torch.max(logits, dim=-1)[1]
        return action_idx


class BC_trainer():
    def __init__(self, env, dataset, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.num_players = 2
        self.env = env
        obs = self.featurize_state(self.env.state)
        visual_obs_size = obs['visual_obs'].shape
        robot_obs_size = obs['robot_obs'].shape
        self.players = (
            BehaviouralCloning(visual_obs_size=visual_obs_size, robot_obs_size=robot_obs_size),
            BehaviouralCloning(visual_obs_size=visual_obs_size, robot_obs_size=robot_obs_size)
        )
        self.optimizers = tuple([th.optim.Adam(player.params(), lr=args.lr) for player in self.players])
        self.criterion = nn.CrossEntropyLoss()

    def train_on_batch(self, batch):
        metrics = {}
        state, action = batch['state'].to(self.device), batch['joint_action'].to(self.device)
        obs = self.featurize_state(state)
        for i in range(self.num_players):
            self.optimizers[i].zero_grad()
            pred_action = self.players[i].forward(obs[i])
            loss = self.criterion(pred_action, action[i])
            loss.backward()
            self.optimizers[i].step()
            metrics[f'player{i}_loss'] = loss.item()
        metrics['total_loss'] = sum(metrics.values())
        wandb.log(metrics)
        return metrics

    def featurize_state(self, state):
        # TODO Remember to include agent idx
        pass

    def evaluate(self, num_trials=10):
        average_reward = []
        for trial in range(num_trials):
            self.env.reset()
            obs_p1, obs_p2 = self.featurize_state(self.env.state)
            trial_reward = 0
            done = False
            while not done:
                joint_action = self.players[0].get_action(obs_p1), self.players[1].get_action(obs_p2)
                next_state, reward, done, info = self.env.step(joint_action)
                trial_reward += reward
                obs_p1, obs_p2 = self.featurize_state(next_state)
            average_reward.append(trial)
        return np.mean(average_reward)

    def train_epoch(self, dataset):
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
        for batch in tqdm(dataloader):
            self.train_on_batch(batch)

    def training(self, num_epochs=100):
        base_path = '.'
        wandb.init(project="overcooked_ai_test", entity="stephaneao", dir=base_path)
        dataset = None # TODO
        for epoch in range(num_epochs):
            self.train_epoch(dataset)
            mean_reward = self.evaluate()
            wandb.log({'evaluation_reward': mean_reward})


if __name__ == '__main__':
    args = get_arguments()
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.env_name), horizon=400)
    dataset = None # TODO

    bct = BC_trainer(env, dataset, args)


