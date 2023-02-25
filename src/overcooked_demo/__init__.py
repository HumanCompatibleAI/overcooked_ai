import os
import subprocess


def start_server():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    subprocess.call("./up.sh")


def move_agent():
    from overcooked_demo.server.move_agents import main

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(dir_path, "server"))
    main()
