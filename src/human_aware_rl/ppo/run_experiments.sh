#!/usr/bin/env bash
# This file contains the script to generate the baseline ppo self-play agents for the 5 classic layouts

# Please check if your computer has enough power for 16x parallelization, otherwise change the num_workers parameter
python ppo_rllib_client.py with  seeds=[11] layout_name="cramped_room" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[21] layout_name="cramped_room" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[31] layout_name="cramped_room" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[41] layout_name="cramped_room" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3


python ppo_rllib_client.py with  seeds=[11] layout_name="asymmetric_advantages" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[21] layout_name="asymmetric_advantages" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[31] layout_name="asymmetric_advantages" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[41] layout_name="asymmetric_advantages" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3

python ppo_rllib_client.py with  seeds=[11] layout_name="coordination_ring" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[21] layout_name="coordination_ring" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[31] layout_name="coordination_ring" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[41] layout_name="coordination_ring" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3

python ppo_rllib_client.py with  seeds=[11] layout_name="forced_coordination" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[21] layout_name="forced_coordination" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[31] layout_name="forced_coordination" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[41] layout_name="forced_coordination" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3

python ppo_rllib_client.py with  seeds=[11] layout_name="counter_circuit_o_1order" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[21] layout_name="counter_circuit_o_1order" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[31] layout_name="counter_circuit_o_1order" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3
python ppo_rllib_client.py with  seeds=[41] layout_name="counter_circuit_o_1order" num_workers=16 train_batch_size=12800 sgd_minibatch_size=8000 num_training_iters=500 evaluation_interval=100 use_phi=False entropy_coeff_start=0.2 entropy_coeff_end=0.0005 num_sgd_iter=8 lr=1e-3

python plot_example_experiments.py