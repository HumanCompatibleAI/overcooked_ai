import tempfile, webbrowser, json, uuid, os
from IPython.display import display, HTML
from string import Template
from overcooked_ai_py.static import VISUALIZATION_DIR

def load_visualization_file_as_str(filename):
    with open(os.path.join(VISUALIZATION_DIR, filename), 'r') as f:
        lines = f.readlines()
        return "\n".join(lines)


def run_html_in_web(html, prefix="overcooked_ai_visualization_"):
    with tempfile.NamedTemporaryFile('w', delete=False, prefix=prefix, 
                                     suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)


def run_html_in_ipython(html):
    display(HTML(html))


def create_chart_html(events_data, chart_settings=None, box_id=None):
    # all numerical values are pixels
    # not explicity stated below margins are not implemented
    default_chart_settings = {'height': 250, # height of whole chart container
        'width': 720,  # width of whole chart container
        'margin': {'top': 20, 'right': 60, 'bottom': 180, 'left': 40}, # margins of chart (not container) without legends
        'hold_line_width': 3, # hold line is line between pickup and drop event that symbolizes holding object by the player
        'highlighted_hold_line_width': 6, # highlighting element is by hovering mouse over something associated with described object
        'object_line_width': 0, 'highlighted_object_line_width':3,
        'object_event_height': 10, 'object_event_width':16, # object event is triangle
        'label_text_shift': 30, # how far from axis is axis label
        'add_cumulative_data': True, # cumulative data is data about cumulative events
        'cumulative_data_ticks': 4,
        'show_legends': True, # when setting show_legends to False it is recommended to also lower height and margin.bottom
        'legend_title_size': 10, 'legend_points_height': 10, 'legend_points_width': 16, 
        'legend_points_margin': {'bottom': 5, 'right': 5}, # margin after legend point and next point or legend text
        'legend_margin': {'right':  5} # margin between legend columns
        }

    settings = default_chart_settings.copy()
    settings.update(chart_settings or {})

    box_id = box_id or "graph-div-" + str(uuid.uuid1())

    js_text_template = Template(load_visualization_file_as_str("event_chart.js"))
    js_text = js_text_template.substitute({'data': json.dumps(events_data), 'box_id': "#"+box_id, 
        'settings': json.dumps(settings)})

    html_template = Template(load_visualization_file_as_str("chart.html"))
    css_text = load_visualization_file_as_str("style.css")

    return html_template.substitute({'css_text': css_text, 'js_text': js_text, 'box_id':box_id})
