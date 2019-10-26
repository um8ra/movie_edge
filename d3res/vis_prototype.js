<!DOCTYPE html>
<meta charset="utf-8">

<head>
	<!-- Load in the d3 library -->
	<script type="text/javascript" src="https://d3js.org/d3.v5.min.js"></script>
</head>
<style type="text/css">
/* 13. Basic Styling with CSS */

/* Style the lines by removing the fill and applying a stroke */
.line {
    fill: none;
   // stroke: #ffab00;
    stroke-width: 3;
}

@media print {
    div { 
		break-after:always; 
		} 
}
  
.overlay {
  fill: none;
  pointer-events: all;
}

/* Style the dots by assigning a fill and stroke */
.scatter {
    fill: steelblue;
    stroke: #000;
}
  
  .focus circle {
  fill: none;
  stroke: steelblue;
}

</style>
<!-- Body tag is where we will append our SVG and SVG objects-->
<body>

<div id = "graph"> </div>

</body>


<script>
var margin = {top: 100, right: 200, bottom: 100, left: 100};
var  width = window.innerWidth - margin.left - margin.right -100; // Use the window's width 
var  height = window.innerHeight - margin.top - margin.bottom -100; // Use the window's height
var out = 0;
var makeChart = function([coarse,medium,fine]){
	var k = 1;
	var xScale = d3.scaleLinear().domain([-50,50]).range([margin.left,width+margin.left]);
	var yScale = d3.scaleLinear().domain([-50,50]).range([height+margin.top,margin.top]);
	var zoomLevel = d3.scaleQuantize().domain([1,8]).range([1, 2, 3]);
	var currentZoom = 1;
	var svg = d3.select("#graph")
				.append('svg')
				.attr("width", width + margin.left + margin.right)
				.attr("height", height + margin.top + margin.bottom);
				//.attr("viewBox", [0, 0, width, height]);
	var g =svg.append("g")
			.attr("class","holder")
	
	g.selectAll('circle')
		.data(coarse)
		.enter()
		.append("circle")
		.attr("r",5)
		.attr("cx",function(d) {return xScale(d.x)})
		.attr("cy",function(d) {return yScale(d.y)})
		.attr("class","scatter");
		
	svg.call(d3.zoom()
      .extent([[0, 0], [width, height]])
      .scaleExtent([1, 8])
      .on("zoom", zoomed));

	  function zoomed() {
		g.attr("transform", d3.event.transform);
		k = d3.event.transform.k;
		var newZoom = zoomLevel(k)
		var data;
		if (newZoom  != currentZoom){
			if (newZoom == 1) {
				data = coarse;
			}
			if (newZoom == 2) {
				data = medium;
			}
			if (newZoom == 3) {
				data = fine;
			}
		
			console.log('Setting new zoom level ' + newZoom + ' ' +k)
			currentZoom = newZoom
			
			g.selectAll('circle')
			 .data(data)
			 .transition()
			 .attr("cx",function(d) {return xScale(d.x)})
			 .attr("cy",function(d) {return yScale(d.y)})
			 
			 .ease(d3.easeLinear)
			 .duration(1000) ;
		}
	  }
}



promises = [d3.json("./c110.json"),d3.json("./c1634.json"),d3.json("./c19246.json")]
Promise.all(promises).then(makeChart);



</script>
