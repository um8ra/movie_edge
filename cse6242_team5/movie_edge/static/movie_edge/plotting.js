	
	
function drawGraph(data,highlight,layer) {

	// remove current graph
	g.selectAll('.scatter').remove()
	
	//Got the data, now draw it.
	g.selectAll('.scatter')
	.data(data)
	.enter()
	.append("circle")
	.attr("r",zoomParams[layer]['r'])
	.attr("cx",d => xScale(d.x))
	.attr("cy",d => yScale(d.y))
	.attr("stroke-width",zoomParams[layer]['w'])
	.style('fill', d=>colorScale(d.imdb_rating))
	.attr("class",function (d) {
			if (d.ID == highlight) {
				return "selected scatter"
			}
			else {
				return "scatter"
			
			}
	} )

}
	
 //https://stackoverflow.com/questions/42695480/d3v4-zoom-coordinates-of-visible-area
function getVisibleArea(t) {
		var ul = t.invert([0, 0]),
		lr = t.invert([width, height]);
		return {left:Math.trunc(ul[0]),
					bot:Math.trunc(ul[1]),
					right:Math.trunc(lr[0]),
					top:Math.trunc(lr[1])
		}
	}	
		
function getBbox(t) {
		const pixel_bbox = getVisibleArea(t);
		const tmp = {top: yScale.invert(pixel_bbox.bot),
						bot: yScale.invert(pixel_bbox.top),
						left:xScale.invert(pixel_bbox.left),
						right:xScale.invert(pixel_bbox.right)}
		const  height = tmp.top-tmp.bot;
		const  width = tmp.right-tmp.left;			
		return {top:tmp.top+height*bbox_pad,
				bot:tmp.bot-height*bbox_pad,
				left: tmp.left-width*bbox_pad,
				right: tmp.right+width*bbox_pad}				
}
	
	
	
function  bboxFilter(d,bbox) {

	const  x_ok = (d.x >= bbox.left) && (d.x <= bbox.right);
	const  y_ok = (d.y >= bbox.bot) && (d.y <=bbox.top)
	
	return x_ok && y_ok;	
}

function animateClusters(movieData, bbox,startLevel,endLevel) {
	
	const filtered = movieData.filter(d=>bboxFilter(d,bbox));	
	//start with removing
	g.selectAll('.scatter').remove()
	
	//Put start points
	g.selectAll('.scatter')
	.data(filtered)
	.enter()
	.append("circle")
	.attr("r",zoomParams[startLevel]['r'])
	.attr("cx",function(d) {return xScale(d['L'+startLevel+'x'])})
	.attr("cy",function(d) {return yScale(d['L'+startLevel+'y'])})
	.attr("class","scatter")
	.attr("stroke-width",zoomParams[endLevel]['w'])
	.style('fill', d=>colorScale(d.imdb_rating))
	.style('opacity',1.0)
	
	
	//Transition 
	g.selectAll('.scatter').transition().duration(1000)
			.attr("cx",function(d) {return xScale(d['L'+endLevel+'x'])})
			.attr("cy",function(d) {return yScale(d['L'+endLevel+'y'])})
			.attr("r",zoomParams[endLevel]['r'])
			.attr("stroke-width",zoomParams[endLevel]['w'])
			.style('opacity',1.0)
			.end()
			.then(()=> drawGraph(payload[endLevel],0,endLevel));

	//don't forget to redraw when done!

}


function zoomed() {
		var tx = d3.event.transform
	 	g.attr("transform",tx );
		k = d3.event.transform.k;
		var newZoom = zScale(k)
		var r;
		tmp = newZoom;	
		
		// handle zoom changes
		if (newZoom  != currentZoom){ 
			var bbox = getBbox(tx);
			var data = payload[newZoom]
			console.log('Setting new zoom level ' + newZoom + ' ' +k)
			
			animateClusters(movieData, bbox,currentZoom,newZoom)
			currentZoom = newZoom;
		}
}