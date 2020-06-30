from overcooked_ai_py.mdp.layout_generator import LayoutGenerator, mdp_fn_random_choice



def test_from_name(name):
    mdp_params_lst = [{"layout_name": name}]
    mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params_lst[0])
    print("mdp_fn", mdp_fn)
    mdp = mdp_fn()
    print("mdp", mdp)
    print("mdp terrain")
    for l in mdp.terrain_mtx:
        print(l)
    print("start positions", mdp.start_player_positions)
    print("----")

    lg = LayoutGenerator((10, 10), mdp_params_lst[0])
    mdp_fn_2 = lg.generate_padded_mdp()
    print("mdp_fn_2", mdp_fn_2)
    mdp_2 = mdp_fn_2()
    print("mdp_2", mdp_2)
    print("mdp_2 terrain")
    for l in mdp_2.terrain_mtx:
        print(l)
    print("success test_from_name")


def test_from_params(inner_shape=(5, 4), prop_empty=0.6, prop_feats=0.1, display=False):
    mdp_params = {
        "inner_shape": inner_shape,
        "prop_empty": prop_empty,
        "prop_feats": prop_feats,
        "display": display
    }

    lg = LayoutGenerator((10, 10), mdp_params)
    mdp_fn_2 = lg.generate_padded_mdp()
    print("mdp_fn_2", mdp_fn_2)


    for i in range(5):
        mdp_2 = mdp_fn_2()
        print("mdp_2 terrain", i)
        for l in mdp_2.terrain_mtx:
            print(l)

        print("start player position", mdp_2.start_player_positions)
        print("=======================")
    print("success test_from_params")

def test_from_name_lst(name_lst):

    mdp_params_lst = [{"layout_name": name} for name in name_lst]
    mdp_lst = [LayoutGenerator.mdp_gen_fn_from_dict(mdp_params) for mdp_params in mdp_params_lst]
    print("success test_from_name_lst")






test_from_name("cramped_room")

#test_from_name_lst(["cramped_room", "cramped_room_2"])
# test_from_params()