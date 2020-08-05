import tempfile 
import webbrowser 
from IPython.display import display, HTML
import pkgutil
from .. import visualization
import json
from string import Template

def run_html_in_web(html, prefix="overcooked_ai_visualization_"):
    with tempfile.NamedTemporaryFile('w', delete=False, prefix=prefix, 
                                     suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)


def run_html_in_ipython(html):
    display(HTML(html))


def create_chart_html(events_data, chart_settings=None):
    # all numerical values are pixels
    # not explicity stated below margins are not implemented
    default_chart_settings = {'height': 250, # height of whole chart container
        'width': 720,  # width of whole chart container
        'margin': {'top': 20, 'right': 60, 'bottom': 180, 'left': 40}, # margins of chart (not container) without legends
        'hold_line_width': 3, # hold line is line between pickup and drop event that symbolizes holding item by the player
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

    visualization_path = visualization.__name__

    js_text_template = Template(pkgutil.get_data(visualization_path, "event_chart.js").decode())
    js_text = js_text_template.substitute({'data': json.dumps(events_data), 'box_id': '#graph-div', 
        'settings': json.dumps(settings)})

    html_template = Template(pkgutil.get_data(visualization_path, "chart.html").decode())
    css_text = pkgutil.get_data(visualization_path, "style.css").decode()

    return html_template.substitute({'css_text': css_text, 'js_text': js_text})