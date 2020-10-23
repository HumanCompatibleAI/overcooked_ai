//it is run_event_chart.js file - do not delete this line, is it used for tests
require.config({
    paths: {
        "d3": "https://d3js.org/d3.v5.min",
        "jquery": "https://code.jquery.com/jquery-3.5.1.slim.min"
        }
}
);

var data = $data ;
var chart_box_id = "$chart_box_id" ;
var legends_box_id = "$legends_box_id" ;
var settings = $settings ;
render_event_chart(data, chart_box_id, legends_box_id, settings);
