import unittest
import argparse
from overcooked_ai_py.mdp.overcooked_test import *
from overcooked_ai_py.planning.planners_test import TestMotionPlanner, TestMediumLevelPlanner, TestJointMotionPlanner, TestHighLevelPlanner
from overcooked_ai_py.agents.agent_test import TestAgentEvaluator, TestBasicAgents
from overcooked_ai_py.visualization.visualization_test import TestVisualizations

if __name__ == '__main__':
    unittest.main()