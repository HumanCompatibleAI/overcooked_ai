import unittest, uuid, json
import os
from overcooked_ai_py.visualization.extract_events import extract_events
from overcooked_ai_py.visualization.visualization_utils import create_chart_html
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import load_from_json, NumpyArrayEncoder
from utils import TESTING_DATA_DIR
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

def jsonify(obj):
    return json.loads(json.dumps(obj, cls=NumpyArrayEncoder))

class TestVisualizations(unittest.TestCase):
    def setUp(self):
        Recipe.configure({})
        trajectory_path = os.path.join(TESTING_DATA_DIR, "test_visualizations", "trajectory.json")
        events_path = os.path.join(TESTING_DATA_DIR, "test_visualizations", "expected_extracted_events.json")
        self.trajectory1 = AgentEvaluator.load_traj_from_json(trajectory_path)
        self.extracted_events1 = load_from_json(events_path)

    def test_event_extraction(self):
        # jsonify to not care about object types (i.e. list vs tuple), but its contents
        self.assertEqual(jsonify(extract_events(self.trajectory1)), jsonify(self.extracted_events1))

    def test_create_chart_html(self):
        box_id = "graph-div-" + str(uuid.uuid1())
        html = create_chart_html(self.extracted_events1, box_id=box_id)

        self.assertTrue("it is chart.html file" in html)
        self.assertTrue("it is event_chart.js file" in html)
        self.assertTrue("it is style.css file" in html)

        self.assertTrue("#"+box_id in html)

        list_of_replaced_variable_names = ["data", "box_id", "settings", "css_text", "js_text"]
        for name in list_of_replaced_variable_names:
            self.assertFalse("$"+name in html)

        self.assertTrue(json.dumps(self.extracted_events1) in html)

if __name__ == '__main__':
    unittest.main()