function drawGraph(data, highlight, layer) {
	// Remove and redraw plot 
	// data: cluster or movie data
	// highlight: ID of point to highlight
	// layer: clustering level


    // remove current graph
    g.selectAll('.scatter').remove();

    //Got the data, now draw it.
    g.selectAll('.scatter')
        .data(data)
        .enter()
        .append("circle")
        .attr("r", zoomParams[layer]['r'])
        .attr("cx", d => xScale(d.x))
        .attr("cy", d => yScale(d.y))
        .attr("stroke-width", zoomParams[layer]['w'])
        .style('fill', d => colorScale(d[IMDB_RATING]))
        .attr("class", function (d) {

            if (d.ID === highlight) {
                return "selected scatter"
            } else {
                return "scatter"

            }
        })
        .on("dblclick.zoom", selectHighlight);

}

function inputFormatCluster(r) {
	// like inputFormat in grid_interactions, only for cluster arrays
    r[GENRES] = decodeURIComponent(r[GENRES]);
    r[ACTORS] = decodeURIComponent(r[ACTORS]);
    return r
}


//https://stackoverflow.com/questions/42695480/d3v4-zoom-coordinates-of-visible-area
function getVisibleArea(t) {
	// Given a current transform event, get the visible viewport, in pixel space
    const ul = t.invert([0, 0]),
        lr = t.invert([width, height]);
    return {
        left: Math.trunc(ul[0]),
        bot: Math.trunc(ul[1]),
        right: Math.trunc(lr[0]),
        top: Math.trunc(lr[1])
    }
}



function getBbox(t) {
	// Given a current transform event, get the bounding box in x/y space
    const pixel_bbox = getVisibleArea(t);
    const tmp = {
        top: yScale.invert(pixel_bbox.bot),
        bot: yScale.invert(pixel_bbox.top),
        left: xScale.invert(pixel_bbox.left),
        right: xScale.invert(pixel_bbox.right)
    };
    const height = tmp.top - tmp.bot;
    const width = tmp.right - tmp.left;
    return {
        top: tmp.top + height * bbox_pad,
        bot: tmp.bot - height * bbox_pad,
        left: tmp.left - width * bbox_pad,
        right: tmp.right + width * bbox_pad
    }
}


 function inBbox(d, bbox) {
     const x_ok = (d.x >= bbox.left) && (d.x <= bbox.right);
     const y_ok = (d.y >= bbox.bot) && (d.y <= bbox.top);
     return x_ok && y_ok;
 }

function bboxFilter2Level(d, bbox, startLevel, endLevel) {
	// checks if coordinates of each d at startLevel or endLevel lie inside bbox 
    const x_okS = (d['L' + startLevel + 'x'] >= bbox.left) && (d['L' + startLevel + 'x'] <= bbox.right);
    const x_okE = (d['L' + endLevel + 'x'] >= bbox.left) && (d['L' + endLevel + 'x'] <= bbox.right);
    const y_okS = (d['L' + startLevel + 'y'] >= bbox.bot) && (d['L' + startLevel + 'y'] <= bbox.top);
    const y_okE = (d['L' + endLevel + 'y'] >= bbox.bot) && (d['L' + endLevel + 'y'] <= bbox.top);

    return (x_okS && y_okS) || (x_okE && y_okE);
}


function animateClusters(movieData, bbox, startLevel, endLevel) {
	// Animates the cluster transitions
	
	
    const filtered = movieData.filter(d => bboxFilter2Level(d, bbox, startLevel, endLevel));
    //start with removing
    g.selectAll('.scatter').remove();

    //Put start points
    g.selectAll('.scatter')
        .data(filtered)
        .enter()
        .append("circle")
        .attr("r", zoomParams[startLevel]['r'])
        .attr("cx", function (d) {
            return xScale(d['L' + startLevel + 'x'])
        })
        .attr("cy", function (d) {
            return yScale(d['L' + startLevel + 'y'])
        })
        .attr("class", "scatter")
        .attr("stroke-width", zoomParams[endLevel]['w'])
        .style('fill', d => colorScale(d[IMDB_RATING]))
        .style('opacity', 1.0);


    //Transition
    g.selectAll('.scatter').transition().duration(1000)
        .attr("cx", function (d) {
            return xScale(d['L' + endLevel + 'x'])
        })
        .attr("cy", function (d) {
            return yScale(d['L' + endLevel + 'y'])
        })
        .attr("r", zoomParams[endLevel]['r'])
        .attr("stroke-width", zoomParams[endLevel]['w'])
        .style('opacity', 1.0)
        .end()
        .then(() => drawGraph(payload[endLevel], 0, endLevel)); //don't forget to redraw when done!

}

//https://observablehq.com/@d3/zoom-to-bounding-box
function centerOnElement(px,py,k){
	//Transition to center on x/y (pixel coords) at scale level k
	svg.transition().duration(750).call(
        myzoom.transform,
        d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(k)
            .translate(-px, -py));
}

function resetZoom(){
	//Gets us back to default zoom
	centerOnElement(width/2,height/2,1)
}


function getViewport(center,k) {
	// gets bbox of viewport centered on x,y, at zoom level k
	//returns bbox in x/y space
	let x = center.x
	let y = center.y
	let px = xScale(x)
	let py = yScale(y)
	let t = d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(k)
            .translate(-px, -py)
	//console.log(t)
	return getBbox(t);
	
}

function getPtBbox(pts) {
	//given list pts (as IDs), find a bounding box for them at each level of zoom
	//returns x,y,k to center on the target points, in pixel space
		
	// Start with getting a bbox in x/y 
	let movies = payload[5];
	let indices = pts.map(x => decoder[x]	)
	let objs = []
	indices.forEach(i=>objs.push(movies[i]));
	let levelCoords = {}
	let centers ={}
	let lvl=0;
	for (lvl=0; lvl <6; lvl++ ){
		levelCoords[lvl] = objs.map(function (d) {return {x:d['L'+lvl+'x'],y:d['L'+lvl+'y']}})
		let xs = levelCoords[lvl].map(d=>d.x);
		let ys = levelCoords[lvl].map(d=>d.y);
		centers[lvl] = {x:(d3.max(xs)+d3.min(xs))/2,y:(d3.max(ys)+d3.min(ys))/2}
	}

	//console.log(levelCoords)
	//console.log(centers)
	// Test zoom levels
	let best = {x:width/2, y:height/2, k:1}
	let k = 0;
	for (k=1.5;k<zoomParams.maxZoom;k++){
		
		let zoomLevel = zScale(k);
		//console.log('checking k='+k+' zoomlevel='+zoomLevel)
		let levelCoord = levelCoords[zoomLevel]
		let center = centers[zoomLevel]		
		let bbox = getViewport(center,k);
		//console.log(bbox)
		//console.log(levelCoord)
		let debug = levelCoord.forEach(foo=>inBbox(foo,bbox))
		//console.log(debug)
		if (levelCoord.every(d=>inBbox(d,bbox))) {
			best = {x:center.x,y:center.y,k:k}
		}
		
	}
	return best;
}

function getPtsClusterIDatLevel(pts,level){
	//at a given zoom level, get the cluster IDs for each point in pts
	let tmp = getPtBbox(pts);
	let movies = payload[5];
	let levelStr = 'L'+zScale(tmp.k)
	let indices = pts.map(x => decoder[x]	)
	let objs = []
	indices.forEach(i=>objs.push(movies[i]));
	let clusterIDs = objs.map(d=>d[levelStr])//clusterIds
	return clusterIDs;	
}	

function highlightAndCenter(pts) {
	//given a list of movie IDs in points, highlight the clusters containing these things and center screen on them
	let ptBox = getPtBbox(pts)
	let level = zScale(ptBox.k)
	let clusters = getPtsClusterIDatLevel(pts,level)
	console.log(ptBox)
	d3.selectAll('.scatter')
	.attr('class', function(d) {
		if (clusters.includes(d.ID)) {
			return "scatter selected"
		} else {
			return "scatter"
		}
	})	
	centerOnElement(xScale(ptBox.x),yScale(ptBox.y),ptBox.k);
}





function selectHighlight() {
	//selects node and centers
    d3.event.stopPropagation();
    d3.selectAll('.scatter').attr('class', 'scatter');
    d3.select(this).attr('class', 'scatter selected');
    let px = d3.select(this).attr("cx");
    let py = d3.select(this).attr("cy");
    let  k = d3.zoomTransform(svg.node()).k;
    console.log(px, py, k);
    centerOnElement(px,py,k*1.3);

}


function zoomed() {
	// actions to take when zoom events triggered (largely mangaging zoom animation calls)
    const tx = d3.event.transform;
    g.attr("transform", tx);
    const k = d3.event.transform.k;
    const newZoom = zScale(k);
    // var r;
    // tmp = newZoom;

    // handle zoom changes
    if (newZoom !== currentZoom) {
        const bbox = getBbox(tx);
        // var data = payload[newZoom];
        console.log('Setting new zoom level ' + newZoom + ' ' + k);

        animateClusters(data, bbox, currentZoom, newZoom);
        currentZoom = newZoom;
    }
}