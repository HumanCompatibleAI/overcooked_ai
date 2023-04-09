import os

from human_aware_rl.imitation.behavior_cloning_tf2 import (
    get_bc_params,
    train_bc_model,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_TRAIN,
)

if __name__ == "__main__":
    # random 3 is counter_circuit
    # random 0 is forced coordination
    # the reason why we use these as the layouts name here is that in the cleaned pickled file of human trajectories, the df has layout named random3 and random0
    # So in order to extract the right data from the df, we need to use these names
    # however when loading layouts there are no random0/3
    # The same parameter is used in both setting up the layout for training and loading the corresponding trajectories
    # so without modifying the dataframes, I have to create new layouts
    for layout in [
        "random3",
        "coordination_ring",
        "cramped_room",
        "random0",
        "asymmetric_advantages",
    ]:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # this is where
        bc_dir = os.path.join(current_file_dir, "bc_runs", "train", layout)
        if os.path.isdir(bc_dir):
            # if this bc agent has been created, we continue to the next layout
            continue
        params_to_override = {
            "layouts": [layout],
            "layout_name": layout,
            "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
            "epochs": 100,
            "old_dynamics": True,
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(bc_dir, bc_params, True)
