require.config({
    paths: {
        d3: "https://d3js.org/d3.v5.min"
    }
});

require(['d3'], function(d3) {   

    function triangle_points(center_x, center_y, triangle_width, triangle_height) {
        if (upward == true) {
            return [[center_x-triangle_width/2, center_y+triangle_height/2],
                    [center_x+triangle_width/2, center_y+triangle_height/2],
                    [center_x, center_y - triangle_height*0.5]];}
        else {
            return [[center_x-triangle_width/2, center_y-triangle_height/2],
                    [center_x+triangle_width/2, center_y-triangle_height/2],
                    [center_x, center_y + triangle_height*0.5]];}
    }

    function triangle_points_from_data(d, xScale, yScale, triangle_width, triangle_height) {
        var center_x = xScale(d.timestep)
        var center_y = yScale(d.player)

        if (d.action == "pickup") { upward = true;}
        else if (d.action == "drop") { upward = false;}
        else {throw 'unknown action';}

        return triangle_points(center_x, center_y, triangle_width, triangle_height, upward);
    }

    function points_to_attr(points){
        return points.map(x => x.join(",")).join(" ");
    }

    function create_id_attribute(d) {
        return "item-id"+d.item_id
    }
    
    function render_event_chart(data, box_id="#graph-div", settings = 
        {height: 250, width: 720, margin: {top: 20, right: 60, bottom: 180, left: 40},
        hold_line_width:3, highlighted_hold_line_width:6, object_line_width:0, highlighted_object_line_width:3, 
        object_event_height:10, object_event_width:16, label_text_shift: 30,
        add_cumulative_data:true, cumulative_data_ticks:4,
        show_legends:true, legend_title_size: 10, legend_points_height: 30, legend_points_width:30, 
        legend_points_margin: {bottom:5, right:5}, legend_margin: {right:5}} ){
        
        inner_height = settings.height - settings.margin.top - settings.margin.bottom;
        inner_width = settings.width - settings.margin.left - settings.margin.right;

        function add_left_axis(svg, yAxis, text) {
            svg.append("g")
                .attr("class", "y axis")
                .call(yAxis)
            .append("text")
                .attr("class", "label")
                .attr("transform", "rotate(-90)")
                .attr("y", -settings.label_text_shift )
                .attr("dy", ".71em")
                .style("text-anchor", "end")
                .text(text);
        }
        
        function add_right_axis(svg, yAxis, text) {
            svg.append("g")
                    .attr("class", "y axis")
                    .call(yAxis)
                    .attr("transform", "translate(" + inner_width +")")
                .append("text")
                    .attr("class", "label")
                    .attr("transform", "rotate(-90)")
                    .attr("y", 10 + settings.label_text_shift)
                    .attr("dy", ".71em")
                    .style("text-anchor", "end")
                    .text(text);
        }
        
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
        
        var svg = d3.select(box_id).append("svg")
            .attr("width", settings.width)
            .attr("height", settings.height)
            .append("g")
            .attr("transform", "translate(" + settings.margin.left + "," + settings.margin.top + ")");

        var xScaleTime = d3.scaleLinear()
            .range([0, inner_width])
            .domain(d3.extent(data, d => d.end_timestep)).nice();
        var xAxisTime = d3.axisBottom(xScaleTime);
        add_bottom_axis(svg, xAxisTime, "Timesteps");

        var yScalePlayers = d3.scaleOrdinal()
            .range([inner_height, 0])
            .domain(d3.extent(data, d => d.player));
        yAxisPlayers = d3.axisLeft(yScalePlayers);
        add_left_axis(svg, yAxisPlayers, "Players");


        item_colours = ["silver", "red", "#F6C283", "blue"];
        item_types = ["dish", "tomato", "onion", "soup"];

        colorScaleItem = d3.scaleOrdinal()
            .range(item_colours)
            .domain(item_types);

        function highlight_item(svg, d)
        {
            
            svg.selectAll("#"+create_id_attribute(d))
            .filter(".data-point")
                .style("stroke-width", settings.highlighted_object_line_width);
                
            svg.selectAll(".line-holding")
            .filter("#"+create_id_attribute(d))
                .style("stroke-width", settings.highlighted_hold_line_width);
        }
        
        function unghlight_item(svg, d)
            {
            
                svg.selectAll(".data-point")
                .filter("#"+create_id_attribute(d))
                    .style("stroke-width", settings.object_line_width);
                svg.selectAll(".line-holding")
                .filter("#"+create_id_attribute(d))
                    .style("stroke-width", settings.hold_line_width);
            }

        svg.selectAll(".line-holding")
        .data(data.filter(d => ["holding"].includes(d.action)))
        .enter().append("line")
            .attr("class", "line-holding")
            .attr('id', create_id_attribute)
            .attr("x1", d => xScaleTime(d.start_timestep))
        .attr("x2", d => xScaleTime(d.end_timestep))
            .attr("y1", d => yScalePlayers(d.player))
        .attr("y2", d => yScalePlayers(d.player))
        .style("stroke", d => colorScaleItem(d.item_name))
        .style("stroke-width", settings.hold_line_width)
            .on("mouseover", d => highlight_item(svg, d))
            .on("mouseout", d => unghlight_item(svg, d));

        svg.selectAll(".data-point")
        .data(data.filter(d => ["drop", "pickup"].includes(d.action)))
        .enter().append("polygon")
            .attr('id', create_id_attribute)
            .attr("class", "data-point")
            .attr("points", d => points_to_attr(triangle_points_from_data(d, 
                xScaleTime, yScalePlayers, settings.object_event_width, settings.object_event_height)))
            .attr("fill", d => colorScaleItem(d.item_name))
            .style("stroke", d => colorScaleItem(d.item_name))
            .style("stroke-width", settings.object_line_width)
            .on("mouseover", d => highlight_item(svg, d))
            .on("mouseout", d => unghlight_item(svg, d));

    if (settings.add_cumulative_data) {

        cumulative_data = data.filter(d => d.sum >= 0);
        var yScaleCumulative = d3.scaleLinear()
            .range([inner_height, 0])
            .domain(d3.extent(cumulative_data, d => d.sum)).nice();

        var yAxisCumulative = d3.axisRight(yScaleCumulative).ticks(settings.cumulative_data_ticks);
            
        add_right_axis(svg, yAxisCumulative, "Event count")
        cumulative_line = d3.line()
            .x(d => xScaleTime(d.timestep))
            .y(d => yScaleCumulative(d.sum));

        players = [...new Set(cumulative_data.map(d => d.player))];
        players.sort();

        colorScalePlayers = d3.scaleOrdinal()
            .range(["#A33E3E","#4E8C2C","#55BDB3","#6F4A74","#6F2929"])
            .domain(players);
        
        for (player of players){
            svg.append("path")
            .datum(cumulative_data.filter(d => d.player === player))
                .attr("fill", "none")
                .attr("stroke", colorScalePlayers(player))
                .attr("stroke-width", 1.5)
                .attr("stroke-linejoin", "round")
                .attr("stroke-linecap", "round")
                .attr("d", cumulative_line);
                }
        }

    if (settings.show_legends) {
        legend_header_y = settings.margin.top + inner_height + settings.label_text_shift;
        legend_points_y = legend_header_y + settings.legend_title_size + settings.legend_points_margin.bottom*2;
        legend_point_height_with_margin = settings.legend_points_height + settings.legend_points_margin.bottom;
        legend_text_offset_y = settings.legend_points_width + settings.legend_points_margin.right;
        first_polygon_center_y = legend_points_y + settings.legend_points_height/2;
        legends_width_so_far = 0;
        polygon_center_x = legends_width_so_far + settings.legend_points_width/2;

        legends = svg.append("g");
        items_colour_legend = legends.append("g")
            .attr("class", "legend");

        items_colour_legend.append("text")
            .attr("class", "legend-title")
            .attr("x", legends_width_so_far)
            .attr("y", legend_header_y + settings.legend_title_size)
            .text("Item colour")
            .attr("text-anchor", "left")
            .style("alignment-baseline", "middle");

        items_colour_legend.selectAll(".event-colour-legend-point")
            .data(item_types)
            .enter()
            .append("polygon")
                .attr("points", (d,i) => points_to_attr(triangle_points(polygon_center_x, 
                    first_polygon_center_y + i * legend_point_height_with_margin,
                    settings.legend_points_width, settings.legend_points_height, upward=true)))
                .style("fill", d => colorScaleItem(d));

        items_colour_legend.selectAll(".event-colour-legend-text")
            .data(item_types)
            .enter()
            .append("text")
                .attr("x", legends_width_so_far + legend_text_offset_y)
                .attr("y", (d,i) => legend_points_y + i*legend_point_height_with_margin + settings.legend_points_height/2)
                .style("fill", d => colorScaleItem(d))
                .text(d => d)
                .attr("text-anchor", "left")
                .style("alignment-baseline", "middle");
                
        legends_width_so_far = legends.node().getBoundingClientRect().width + settings.legend_margin.right;
        polygon_center_x = legends_width_so_far + settings.legend_points_width/2;

        event_type_legend = legends.append("g")
            .attr("class", "legend");
        
        event_type_legend.append("text")
            .attr("class", "legend-title")
            .attr("x", legends_width_so_far)
            .attr("y", legend_header_y + settings.legend_title_size)
            .text("Event type")
            .attr("text-anchor", "left")
            .style("alignment-baseline", "middle");
        // pickup polygon
        event_type_legend.append("polygon")
            .attr("class", "event-shape-legend-point")
            .attr("points", points_to_attr(triangle_points(polygon_center_x, 
                first_polygon_center_y + 0 * legend_point_height_with_margin,
                settings.legend_points_width, settings.legend_points_height, upward=true)));
        // drop polygon
        event_type_legend.append("polygon")
            .attr("class", "event-shape-legend-point")
            .attr("points", points_to_attr(triangle_points(polygon_center_x, 
                first_polygon_center_y + 1 * legend_point_height_with_margin,
                settings.legend_points_width, settings.legend_points_height, upward=false)));
        // swtich icon made from 2 polygons
        event_type_legend.append("polygon")
            .attr("class", "event-shape-legend-point")
            .attr("points", points_to_attr(triangle_points(polygon_center_x, 
                first_polygon_center_y + 2 * legend_point_height_with_margin,
                settings.legend_points_width, settings.legend_points_height, upward=true)));
        event_type_legend.append("polygon")
            .attr("class", "event-shape-legend-point")
            .attr("points", points_to_attr(triangle_points(polygon_center_x, 
                first_polygon_center_y + 2 * legend_point_height_with_margin,
                settings.legend_points_width, settings.legend_points_height, upward=false)));

        event_types = ["pickup", "drop", "change item"];
        event_type_legend.selectAll(".event-shape-legend-text")
                .data(event_types)
                .enter()
                .append("text")
                    .attr("x", legends_width_so_far + legend_text_offset_y)
                    .attr("y", (d,i) => legend_points_y + i*legend_point_height_with_margin + settings.legend_points_height/2)
                    .text(d => d)
                    .attr("text-anchor", "left")
                    .style("alignment-baseline", "middle");
        
        if (settings.add_cumulative_data) {
            player_colour_legend = legends.append("g").attr("class", "legend");

            legends_width_so_far = legends.node().getBoundingClientRect().width + settings.legend_margin.right;
            player_colour_legend.append("text")
                .attr("class", "legend-title")
                .attr("x", legends_width_so_far)
                .attr("y", legend_header_y + settings.legend_title_size)
                .text("Player line colour")
                .attr("text-anchor", "left")
                .style("alignment-baseline", "middle");
        
            player_colour_legend.selectAll(".player-colour-legend-point")
                .data(players)
                .enter()
                .append("rect")
                    .attr("x", legends_width_so_far)
                    .attr("y", (d,i) => legend_points_y + i*legend_point_height_with_margin)
                    .attr("width", settings.legend_points_width)
                    .attr("height", settings.legend_points_height)
                    .style("fill", d => colorScalePlayers(d))
            
            player_colour_legend.selectAll(".player-colour-legend-text")
                .data(players)
                .enter()
                .append("text")
                    .attr("x", legends_width_so_far + legend_text_offset_y)
                    .attr("y", (d,i) => legend_points_y + i*legend_point_height_with_margin + settings.legend_points_height/2)
                    .style("fill", d => colorScalePlayers(d))
                    .text(d => typeof(d) == "undefined" ? "all" : d)
                    .attr("text-anchor", "left")
                    .style("alignment-baseline", "middle");
    }}}
    var data = $data ;
    var box_id = "$box_id" ;
    var settings = $settings ;
    render_event_chart(data, box_id, settings);
});