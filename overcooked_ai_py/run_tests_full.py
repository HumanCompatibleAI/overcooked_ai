import unittest
import argparse
from overcooked_ai_py.mdp.overcooked_test import TestDirection, TestGridworld, TestOvercookedEnvironment, TestGymEnvironment
from overcooked_ai_py.planning.planners_test import TestMotionPlanner, TestMediumLevelPlanner, TestJointMotionPlanner, TestHighLevelPlanner
from overcooked_ai_py.agents.agent_test import TestAgents, TestScenarios

if __name__ == '__main__':
    unittest.main()