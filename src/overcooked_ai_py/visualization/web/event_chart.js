//it is event_chart.js file - do not delete this line, is it used for tests
require.config({
    paths: {
        "d3": "https://d3js.org/d3.v5.min",
        "jquery": "https://code.jquery.com/jquery-3.5.1.slim.min"
        }
}
);

require(['d3', "jquery"], function(d3, jQuery) {   

    function check_mark_points(center_x, center_y, width, height){
        // points are rescaled to be between -0.5 and 0.5 for easier math
        var points = [[-0.5, 0.0], [-0.115, 0.5], [0.5, -0.28], [0.32, -0.5], [-0.115, 0.084], [-0.345, -0.2]]
        return points.map(point => [(point[0]*width)+center_x, (point[1]*height)+center_y])
    }

    function triangle_points(center_x, center_y, triangle_width, triangle_height, upward) {
        if (upward == true) {
            return [[center_x-triangle_width/2, center_y+triangle_height/2],
                    [center_x+triangle_width/2, center_y+triangle_height/2],
                    [center_x, center_y - triangle_height*0.5]];}
        else {
            return [[center_x-triangle_width/2, center_y-triangle_height/2],
                    [center_x+triangle_width/2, center_y-triangle_height/2],
                    [center_x, center_y + triangle_height*0.5]];}
    }

    function points_attr_from_data(d, xScale, yScale, object_width, object_height) {
        var center_x = xScale(d.timestep)
        var center_y = yScale(d.player)
        if (d.action == "delivery") {
            Math.min()
            var points = check_mark_points(center_x, center_y, object_width, object_height);
        }
        else {
            var points = triangle_points(center_x, center_y, object_width, object_height, upward = d.action == "pickup");
        }
        return points_to_attr(points);
    }

    function points_to_attr(points){
        return points.map(x => x.join(",")).join(" ");
    }

    function create_id_attribute(d) {
        return "object-id"+d.object.object_id;
    }
    
    function player_name(d) {
        return typeof(d) == "undefined" ? "all" : d;
    }

    function adjective_name_to_class(adj_name) {
        return "adjective-"+ (typeof(adj_name) == "undefined" ? "none" : adj_name);
    }

    function adjectives_to_classes(d) {
        return d.adjectives.map(adjective_name_to_class).join(' ');
    }

    function data_point_classes(d, use_adjectives) {
        var result = "data-point object-type-" + d.object.name + " action-" + d.action;
        if (use_adjectives) {
            result += " " + adjectives_to_classes(d);
        }
        return result;
    }

    function cumulative_line_classes(d) {
        return "cumulative-data line player-" + player_name(d[0].player);
    }

    function is_cumulative_data_matching(d, desc){
        function is_empty(param){
            return !Array.isArray(param) || param.length == 0
        }

        var are_actions_matching = array_equals(d.actions, desc.actions) || (is_empty(d.actions) && is_empty(desc.actions));
        var are_adjectives_matching = array_equals(d.adjectives, desc.adjectives) || (is_empty(d.adjectives) && is_empty(desc.adjectives));
        return are_actions_matching && are_adjectives_matching
    }

    // taken from https://stackoverflow.com/questions/50813950/how-do-i-make-an-svg-size-to-fit-its-content
    function resize_svg(svg) {
        // Get the bounds of the SVG content
        var bbox = svg.getBBox();
        // Update the width and height using the size of the contents
        svg.setAttribute("width", bbox.x + bbox.width + bbox.x);
        svg.setAttribute("height", bbox.y + bbox.height + bbox.y);
        }
    
    // taken from https://masteringjs.io/tutorials/fundamentals/compare-arrays
    function array_equals(a, b) {
        return Array.isArray(a) &&
          Array.isArray(b) &&
          a.length === b.length &&
          a.every((val, index) => val === b[index]);
      }

    // taken from https://stackoverflow.com/questions/12503146/create-an-array-with-same-element-repeated-multiple-times
    function repeated_element_array(size, element) {
        return [...Array(size)].map((_, i) => element)
    }

    function replace_array_element(arr, replaced_value, replacing_value)
    {
        for (var i = 0; i < arr.length; i++)
        {
            if (arr[i] === replaced_value)
            arr[i] = replacing_value;
        }
    }
    function indexes_satisfying_condition(arr, condition) {
        return arr.reduce(function(arr, e, i) {
            if (condition(e)) arr.push(i);
            return arr;
          }, [])
    }
    
    function get_elements_by_indexes(arr, indexes) {
        return indexes.map(i => arr[i])
    }

    function render_event_chart(data, chart_box_id="#graph-div", legends_box_id="#legends-div", settings = 
        {label_text_shift: 30,
        object_event_height:10, object_event_width:16, 
        show_event_data: true, use_adjectives: true, actions_to_show: ["pickup","drop", "delivery", "potting", "holding"],
        show_cumulative_data:true, cumulative_data_ticks:4,
        cumulative_events_description: [{actions: ["pickup", "drop", "potting", "delivery"], name: "All events"}],
        show_legends:true})
    {
        var svg = d3.select(chart_box_id)
            .append("svg")
                .attr("class", "chart-box")
             .append("g")
                .attr("class", "chart-area");

        var inner_width = parseInt(jQuery(".chart-area").css("width"), 10);
        var inner_height = parseInt(jQuery(".chart-area").css("height"), 10);

        var max_end_timestep = d3.max(data, d => d.end_timestep) || 0;
        var max_timestep = d3.max(data, d => d.timestep) || 0;
        var xScaleTime = d3.scaleLinear()
            .range([0, inner_width])
            .domain([0, Math.max(max_end_timestep, max_timestep)]).nice();
        var xAxisTime = d3.axisBottom(xScaleTime);
        
        function add_bottom_axis(svg, xAxis, text) {
            svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," +  inner_height + ")")
                    .call(xAxis)
                .append("text")
                    .attr("class", "label")
                    .attr("x", inner_width/2)
                    .attr("y", settings.label_text_shift)
                    .style("text-anchor", "end")
                    .text(text);
        }
        add_bottom_axis(svg, xAxisTime, "Timesteps");

        if (settings.actions_to_show && settings.show_event_data){
            var yScalePlayers = d3.scaleOrdinal()
                .range([inner_height, 0])
                .domain(d3.extent(data, d => d.player));
            var yAxisPlayers = d3.axisLeft(yScalePlayers);

            function add_left_axis(svg, yAxis, text) {
                svg.append("g")
                        .attr("class", "y axis")
                        .call(yAxis)
                    .append("text")
                        .attr("class", "label")
                        .attr("y", -settings.label_text_shift)
                        .text(text);
            }

            add_left_axis(svg, yAxisPlayers, "Players");

            function highlight_object(svg, d)
            {
                svg.selectAll("#"+create_id_attribute(d))
                .filter(".data-point")
                    .classed("highlighted", true)
    
                svg.selectAll(".line-holding")
                .filter("#"+create_id_attribute(d))
                    .classed("highlighted", true)
            }
            
            function unghlight_object(svg, d)
                {
                    svg.selectAll("#"+create_id_attribute(d))
                    .filter(".data-point")
                        .classed("highlighted", false)
                
                    svg.selectAll(".line-holding")
                    .filter("#"+create_id_attribute(d))
                        .classed("highlighted", false)
                }
        
            svg.selectAll(".line-holding")
            .data(data.filter(d => settings.actions_to_show.includes(d.action)))
            .enter().append("line")
                .attr("class", d => "line-holding object-type-" + d.object.name)
                .attr('id', create_id_attribute)
                .attr("x1", d => xScaleTime(d.start_timestep))
                .attr("x2", d => xScaleTime(d.end_timestep))
                .attr("y1", d => yScalePlayers(d.player))
                .attr("y2", d => yScalePlayers(d.player))
                .on("mouseover", d => highlight_object(svg, d))
                .on("mouseout", d => unghlight_object(svg, d));
            
            svg.selectAll(".data-point polygon")
            .data(data.filter(d => ["drop", "pickup"].includes(d.action) && settings.actions_to_show.includes(d.action)))
            .enter().append("polygon")
                .attr("points", d => points_attr_from_data(d, xScaleTime, yScalePlayers, 
                    settings.object_event_width, settings.object_event_height))
                .attr("id", create_id_attribute)
                .attr("class", d => data_point_classes(d, settings.use_adjectives))
                .on("mouseover", d => highlight_object(svg, d))
                .on("mouseout", d => unghlight_object(svg, d));

            svg.selectAll(".data-point ellipse")
                .data(data.filter(d => d.action == "potting" && settings.actions_to_show.includes(d.action)))
                .enter().append("ellipse")
                    .attr("cx", d => xScaleTime(d.timestep))
                    .attr("cy", d => yScalePlayers(d.player))
                    .attr("rx", 0.5*settings.object_event_width)
                    .attr("ry", 0.5*settings.object_event_height)
                    .attr("id", create_id_attribute)
                    .attr("class", d => data_point_classes(d, settings.use_adjectives))
                    .on("mouseover", d => highlight_object(svg, d))
                    .on("mouseout", d => unghlight_object(svg, d));
            
            // create delivery marks at the end to make them more visible
            svg.selectAll(".data-point polygon")
                .data(data.filter(d => d.action == "delivery" && settings.actions_to_show.includes(d.action)))
                .enter().append("polygon")
                    .attr("points", d => points_attr_from_data(d, xScaleTime, yScalePlayers, 
                        settings.object_event_width, settings.object_event_height))
                    .attr("id", create_id_attribute)
                    .attr("class", d => data_point_classes(d, settings.use_adjectives))
                    .on("mouseover", d => highlight_object(svg, d))
                    .on("mouseout", d => unghlight_object(svg, d));
                }

    if (settings.show_cumulative_data) {
        var cumulative_data = data.filter(d => d.sum >= 0);
        var yScaleCumulative = d3.scaleLinear()
            .range([inner_height, 0])
            .domain(d3.extent(cumulative_data, d => d.sum)).nice();
        var yAxisCumulative = d3.axisRight(yScaleCumulative).ticks(settings.cumulative_data_ticks);
                    
        function add_right_axis(svg, yAxis, text) {
            svg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .attr("transform", "translate(" + inner_width +")")
                .append("text")
                    .attr("class", "label")
                    .attr("y", 10 + settings.label_text_shift)
                    .attr("dy", ".71em")
                    .text(text);
        }
        
        add_right_axis(svg, yAxisCumulative, "Event count")
        var cumulative_line = d3.line()
            .x(d => xScaleTime(d.timestep))
            .y(d => yScaleCumulative(d.sum));

        var players = [...new Set(cumulative_data.map(d => d.player))];
        players.sort();
        for (line_desc of settings.cumulative_events_description) {
            for (player of players){
                svg.append("path")
                .datum(cumulative_data.filter(d => d.player === player && is_cumulative_data_matching(d, line_desc)))
                    .attr("class", cumulative_line_classes)
                    .classed(line_desc.class, true)
                    .attr("d", cumulative_line);
            }
        }
    }

    if (settings.show_legends) {
        var legends_div = d3.select(legends_box_id)
            .append("div")
            .classed("legends", true);
        
        function add_legend(legends, legend_class, title, texts, data_point_types, data_point_attrs) {
            var legend = legends.append("div")
                .attr("class", legend_class);

            legend.append("text")
                .attr("class", "legend-title")
                .text(title);
            
            // taken from https://stackoverflow.com/questions/22015684/how-do-i-zip-two-arrays-in-javascript
            const zip = (arr1, arr2, arr3) => arr1.map((k, i) => [k, arr2[i], arr3[i]]);
                   
            var legend_rows = legend.selectAll(".legend-row")
                .data(zip(texts, data_point_types, data_point_attrs))
                .enter()
            .append("span")
                .classed("legend-row", true);

            // to avoid listing every used attr maybe d3-selection can be used https://github.com/d3/d3-selection-multi       
            legend_rows.append("svg")
                    .classed("legend-data-point-area", true)
                .append(function(d) { 
                    return document.createElementNS(d3.namespaces.svg, d[1]);
                    })
                    .attr("class", d => d[2].class)
                    .attr("points", d => d[2].points) 
                    .attr("cx", d => d[2].cx)                    
                    .attr("cy", d => d[2].cy)                    
                    .attr("rx", d => d[2].rx)                    
                    .attr("ry", d => d[2].ry)
                    .attr("d", d => d[2].d)
                    .attr("transform",  d => d[2].transform)
                    .attr("width", d => d[2].width)
                    .attr("height",  d => d[2].height)
                    .classed("data-point", true)
                    .classed("legend-data-point", true);

            legend_rows.append("text")
                .classed("legend-point-data-description", true)
                .text(d => d[0]);

            legend.selectAll(".legend-data-point-area")._groups[0].forEach(resize_svg)
        }
        if (settings.actions_to_show && settings.show_event_data)
        {    
            var event_types_texts = ["pickup", "drop", "potting", "delivery"];
            var event_types_data_point_types = ["polygon", "polygon", "ellipse", "polygon"];
            var upward_triangle_points = triangle_points(settings.object_event_width/2, settings.object_event_height/2, settings.object_event_width, settings.object_event_height, true);
            var event_types_attrs = [
                {"points": upward_triangle_points},
                {"points": triangle_points(settings.object_event_width/2, settings.object_event_height/2, settings.object_event_width, settings.object_event_height, false)},
                {"cx": 0.5*settings.object_event_width, "cy":0.5*settings.object_event_height, "rx":0.5*settings.object_event_width, "ry": 0.5*settings.object_event_height},
                {"points": check_mark_points(settings.object_event_width/2, settings.object_event_height/2, settings.object_event_width, settings.object_event_height)}
            ];
            for ([i, text] of event_types_texts.entries()){
                event_types_attrs[i].class = "data-point event-type-legend-data-point action-"+text;
            }
            // show only actions inside actions_to_show

            var used_event_types_indexes = indexes_satisfying_condition(event_types_texts, e => settings.actions_to_show.includes(e));
            event_types_texts = get_elements_by_indexes(event_types_texts, used_event_types_indexes);
            event_types_data_point_types = get_elements_by_indexes(event_types_data_point_types, used_event_types_indexes);
            event_types_attrs = get_elements_by_indexes(event_types_attrs, used_event_types_indexes);
            add_legend(legends_div, "legend legend-column event-type-legend", "Event types:", event_types_texts, event_types_data_point_types, event_types_attrs);
            
            var object_types_texts = ["onion", "tomato", "dish", "soup"];
            var object_types_data_point_types = repeated_element_array(object_types_texts.length, "polygon");
            var object_types_attrs = [];
            for ([i, text] of object_types_texts.entries()){
                object_types_attrs.push(
                    {"points": upward_triangle_points,
                    "class": "data-point object-type-legend-data-point object-type-"+text
                });
            }
            add_legend(legends_div, "legend legend-column object-type-legend", "Object types:", object_types_texts, object_types_data_point_types, object_types_attrs);
            
            if (settings.use_adjectives){
                var adjectives_texts = [...new Set(data.map(d => d.adjectives).flat(1))];
                replace_array_element(adjectives_texts, undefined, "none");
                adjectives_texts.sort();
                var adjectives_data_point_types = repeated_element_array(adjectives_texts.length, "polygon");
                var adjectives_attrs = [];
                for ([i, text] of adjectives_texts.entries()){
                    adjectives_attrs.push(
                        {"points": upward_triangle_points,
                        "class": "data-point object-type-legend-data-point object-type-onion "+adjective_name_to_class(text)
                    });
                }
                add_legend(legends_div, "legend legend-column adjectives-legend", "Adjectives:",  adjectives_texts, adjectives_data_point_types, adjectives_attrs);
            }
        }

        if (settings.show_cumulative_data) {

            var player_colours_texts = players.map(player_name);
            var player_colours_data_point_types = repeated_element_array(players.length, "rect");
            var player_colours_attrs = [];
            for ([i, text] of player_colours_texts.entries()){
                player_colours_attrs.push({"width": settings.object_event_width, "height":settings.object_event_height,
                "class": "data-point player-colour-legend-data-point cumulative-data player-"+text});
            }
            add_legend(legends_div, "legend legend-column player-colour-legend", "Player line colour:", player_colours_texts, player_colours_data_point_types, player_colours_attrs);

            var cumulative_line_types_classes = settings.cumulative_events_description.map(d => d.class || "");
            var cumulative_line_types_names = settings.cumulative_events_description.map(d => d.name || "");
            var cumulative_line_types_data_point_types = repeated_element_array(players.length, "rect");
            var cumulative_line_types_attrs = [];
            for ([i, cls] of cumulative_line_types_classes.entries()){
                cumulative_line_types_attrs.push({"width": settings.object_event_width, "height":settings.object_event_height, "class":"data-point cumulative-line-type-legend-data-point cumulative-data "+ cls});
            }
            add_legend(legends_div, "legend legend-column cumulative-line-type-legend cumulative-data", "Cumulative line type:", cumulative_line_types_names, cumulative_line_types_data_point_types, cumulative_line_types_attrs);
        }
    }}
    var data = $data ;
    var chart_box_id = "$chart_box_id" ;
    var legends_box_id = "$legends_box_id" ;
    var settings = $settings ;
    render_event_chart(data, chart_box_id, legends_box_id, settings);
});