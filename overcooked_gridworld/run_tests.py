import unittest
import argparse

def load_all_environment_tests(skip_planning):
    # Basics
    from overcooked_gridworld.mdp.overcooked_test import TestDirection, TestGridworld

    # Evaluation
    # TODO: should add more evaluation, benchmarking class for example
    from overcooked_gridworld.mdp.overcooked_interactive_test import TestInteractiveSetup

    if not skip_planning:
        from overcooked_gridworld.planning.planners_test import TestMotionPlanner, TestMediumLevelPlanner, TestJointMotionPlanner, TestHighLevelPlanner
        from overcooked_gridworld.agents.agent_test import TestAgents, TestScenarios

def run_tests():
    unittest.main()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='fast testing', action="store_true")
    args = parser.parse_args()
    load_all_environment_tests(args.f)
    run_tests()