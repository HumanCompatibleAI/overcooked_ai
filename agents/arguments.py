import argparse
from pathlib import Path


def get_arguments():
    """
    Arguments for training agents
    :return:
    """
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--layout', default='asymmetric_advantages',  help='Overcooked map to use')
    parser.add_argument('--horizon', type=int, default=400, help='Max timesteps in a rollout')
    parser.add_argument('--encoding-fn', type=str, default='dense_lossless',
                        help='Encoding scheme to use. Options: "dense_lossless", "OAI_lossless", "OAI_feats"')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='learning rate')
    parser.add_argument('--exp-name', type=str, default='default_exp',
                        help='Name of experiment. Used to name save files.')
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save all models, data, tensorboard metrics.')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path from base_dir to where the expert data is stored')
    parser.add_argument('--dataset', type=str, default='2019_hh_trials_all.pickle',
                        help='Which set of expert data to use. '
                             'See https://github.com/HumanCompatibleAI/human_aware_rl/tree/master/human_aware_rl/static/human_data for options')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers for pytorch train_dataloader (default: 4)')

    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    return args