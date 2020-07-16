from overcooked_ai_py.mdp.layout_generator import MDPParamsGenerator, LayoutGenerator, DEFAILT_PARAMS_SCHEDULE_FN



def test_from_name(name):
    mdp_params = {"layout_name": name}
    mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params)
    print("mdp_fn", mdp_fn)
    mdp = mdp_fn()
    print("mdp", mdp)
    print("mdp terrain")
    for l in mdp.terrain_mtx:
        print(l)
    print("start positions", mdp.start_player_positions)
    print("----")


def test_from_name_padded(name):
    mdp_params = {"layout_name": name}
    name_mpg = MDPParamsGenerator(mdp_params_always=mdp_params)
    lg = LayoutGenerator(name_mpg, (10, 10))
    mdp_fn_2 = lg.generate_padded_mdp
    print("mdp_fn_2", mdp_fn_2)
    mdp_2 = mdp_fn_2()
    print("mdp_2", mdp_2)
    print("mdp_2 terrain")
    for l in mdp_2.terrain_mtx:
        print(l)
    print("success test_from_name")


def test_from_params():

    variable_mpg = MDPParamsGenerator(params_schedule_fn=DEFAILT_PARAMS_SCHEDULE_FN)
    lg = LayoutGenerator(variable_mpg, (5, 4))
    mdp_fn_2 = lg.generate_padded_mdp
    print("mdp_fn_2", mdp_fn_2)
    for i in range(1):
        mdp_2 = mdp_fn_2({'yoyuogsfrg': 300})
        print("mdp_2 terrain", i)
        for l in mdp_2.terrain_mtx:
            print(l)

        print("start player position", mdp_2.start_player_positions)
        print("=======================")
    print("success test_from_params")






test_from_name("cramped_room")
test_from_params()