
import tempfile, webbrowser, json, uuid, os
from IPython.display import display, HTML, Image
from ipywidgets import interactive, IntSlider
from string import Template
from overcooked_ai_py.static import WEB_VISUALIZATION_DIR
from overcooked_ai_py.utils import load_text_file

DEFAULT_EVENT_CHART_SETTINGS = {
    # all numerical values are pixels
    "label_text_shift": 30, # how far from axis is axis label
    "object_event_height": 10, "object_event_width":16, # used both in chart and its legend
    "show_event_data": True, # chart that shows when events happened
    "use_adjectives": True, # adjectives of events like useful, catastrophic etc.
    "actions_to_show": ["pickup","drop", "delivery", "potting", "holding"], # holding is the only continous event - line between pickup and drop/potting/delivery
    "show_cumulative_data": True, # cumulative data is data about sum of events done so far in any point in time - line chart
    "cumulative_data_ticks": 4, # number of ticks on cumulative data axis
    # list of dicts where every dict is one cumulative events curve for every player + one additional for all players
    # in format: {"actions":["action1", "action2"], "adjectives":["adj1", "adj2"]}, "name": "Name of line in legend",
    #  "class": "css-class-name"} in case of lacking key all actions/adjectives are assumed; only 1 adjective needs to match between adjectives list and event adjectives
    "cumulative_events_description": [{"actions":["pickup", "drop", "potting", "delivery"], "name": "All events"}],
    "show_legends": True,
    "use_default_css_file": True, # file found in overcooked_ai_py/visualization/web/event_chart_default.css; when set false you need to rewrite most of the settings from there
    "custom_css_path": None,
    "custom_css_string": ""
}

def load_visualization_file_as_str(filename):
    return load_text_file(os.path.join(WEB_VISUALIZATION_DIR, filename))


def run_html_in_web(html, prefix="overcooked_ai_visualization_"):
    with tempfile.NamedTemporaryFile('w', delete=False, prefix=prefix, 
                                     suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html)
    webbrowser.open(url)


def run_html_in_ipython(html):
    display(HTML(html))


def create_chart_html(events_data, chart_settings=None,  chart_box_id=None, legends_box_id=None):
    
    css_strings = []
    if chart_settings.get("use_default_css_file"):
        css_strings.append(load_visualization_file_as_str("event_chart_default.css"))
        del chart_settings["use_default_css_file"]
    if chart_settings.get("custom_css_path"):
        css_strings.append(load_text_file(chart_settings["custom_css_path"]))
        del chart_settings["custom_css_path"]
    if chart_settings.get("custom_css_string"):
        css_strings.append(chart_settings["custom_css_string"])
        del chart_settings["custom_css_string"]
    css_string = "\n/* new css string */\n".join(css_strings)
    


    chart_box_id = chart_box_id or "chart-div-" + str(uuid.uuid1())
    legends_box_id = legends_box_id or "legends-div-" + str(uuid.uuid1())
    js_string = load_visualization_file_as_str("render_event_chart.js")
    js_string += Template(load_visualization_file_as_str("run_event_chart.js")).substitute(
        {'data': json.dumps(events_data), 'chart_box_id': "#"+chart_box_id,
        'legends_box_id': "#"+legends_box_id, 'settings': json.dumps(chart_settings)})

    html_template = Template(load_visualization_file_as_str("event_chart.html"))

    return html_template.substitute({'css_text': css_string, 'js_text': js_string, 'chart_box_id': chart_box_id,
     'legends_box_id': legends_box_id})

def show_image_in_ipython(data, *args, **kwargs):
    display(Image(data, *args, **kwargs))

def ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))
    return interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list)-1,step=1)})

def show_ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))
    display(interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list)-1,step=1)}))
