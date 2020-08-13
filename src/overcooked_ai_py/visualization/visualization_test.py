import unittest, uuid, json
from overcooked_ai_py.visualization.extract_events import extract_events
from overcooked_ai_py.visualization.visualization_utils import create_chart_html, load_visualization_file_as_str
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.utils import load_from_json, NumpyArrayEncoder

def jsonify(obj):
    return json.loads(json.dumps(obj, cls=NumpyArrayEncoder))
    
class TestVisualizations(unittest.TestCase):
    def setUp(self):
        trajectory_json = json.loads(load_visualization_file_as_str("test_trajectory1.json"))
        self.trajectory1 = AgentEvaluator.load_traj_from_json_obj(trajectory_json)
        self.extracted_events1 = json.loads(load_visualization_file_as_str("extracted_events1.json"))

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
        self.assertFalse("$" in html)
        self.assertTrue(json.dumps(self.extracted_events1) in html)
