function inputFormatCluster(r) { // Decodes cluster data    
    r[GENRES] = decodeURIComponent(r[GENRES]);
    r[ACTORS] = decodeURIComponent(r[ACTORS]);
    return r
}

function highlight(ids) { //flags all nodes associated with movie_ids in ids
   
    const H = getViewState()
	const lvl = zScale(H.k)
	const myids = lvl === 5 ? ids : moviesToLevelID(ids,lvl)
    d3.selectAll('.scatter')
        .attr("class", d=> myids.includes(d.ID) ? "scatter selected" : "scatter")
}

function getVisibleArea(transform) { // Given a transform, return the visible viewport, in pixels 
//https://stackoverflow.com/questions/42695480/d3v4-zoom-coordinates-of-visible-area
    const uppper_left = transform.invert([0, 0]),
        lower_right = transform.invert([width, height]);
    return {
        left: Math.trunc(uppper_left[0]),
        bot: Math.trunc(uppper_left[1]),
        right: Math.trunc(lower_right[0]),
        top: Math.trunc(lower_right[1])
    }
}

function getTransform(){ // returns a d3 zoom transform object representing current viewstate
	const H = getViewState()
	return transform = d3.zoomIdentity.translate(H.x, H.y).scale(H.k);
}

function getViewState() { //returns dict with current viewstate
	return d3.zoomTransform(g.node())
}

function moviesToLevelID(movies,lvl) {// returns list of cluster ids at current lvl corresponding to movies
	const lvl_name = 'L' + lvl;
	const movie_info = payload[5];
	const subset = movie_info.filter(m=>movies.includes(m.movie_id))
	return subset.map(x=>x[lvl_name])
}

function clusterOrMovie(movieItm, clusterItm, lvl){ //returns movieItm if lvl == 5 else clusterItm
	return lvl === 5 ? movieItm : clusterItm
}

function highlightAndCenterSingle(id) { //highlights a movie and centers on it. Does not change zoom level (so a cluster could be highlighted)
    
	
	
    let itm = data.filter(x => x.ID === id)[0];
    const k = d3.zoomTransform(svg.node()).k;
    d3.selectAll('.scatter').attr('class', 'scatter');
    let currID = itm['L' + zScale(k)];

    let node = d3.selectAll('.scatter').filter(d => d.ID === currID);
    node.attr('class', 'scatter selected');
    const px = node.attr("cx");
    const py = node.attr("cy");

    centerOnElement(px, py, k);
}



function drawGraph() { //redraw plot. 
	const H = getViewState()
	const lvl = zScale(H.k)
	const data = payload[lvl]
	let moviesToHighlight = currentGrid.slice()
	moviesToHighlight.push(currentMovie)
	const clusters2Highlight = moviesToLevelID(moviesToHighlight,lvl)
	const IDs2Highlight = clusterOrMovie(moviesToHighlight,clusters2Highlight,lvl)
	
	// remove current graph
    g.selectAll('.scatter').remove();
    g.selectAll('.labels').remove();
    g.selectAll('path').remove();
	
    //Got the data, now draw it.
    drawArcs(); ;// TODO refactor 
    g.selectAll('.scatter')
        .data(data)
        .enter()
        .append("circle")
        .attr("r", sizeScales[lvl])
        .attr("cx", d => xScale(d.x))
        .attr("cy", d => yScale(d.y))
		.attr("class","scatter")
        .attr("stroke-width", zoomParams[lvl]['w'])
        .style('fill', d => colorScale(d[IMDB_RATING]))
        .on("dblclick.zoom", selectHighlight);
    g.selectAll('.scatter')
        .on('mouseover', tip.show)
        .on('mouseout', tip.hide);
    //labels
	const toCall = clusterOrMovie(applyLabelsMovies,applyLabelsClusters,lvl)
	toCall()
    //highlight grid
    highlight(IDs2Highlight); 
    highlightAndCenterSingle(currentMovie); //XXX
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
    g.selectAll('path').remove();
    g.selectAll('.labels').remove();


    //Put start points
    g.selectAll('.scatter')
        .data(filtered)
        .enter()
        .append("circle")
        .attr("r", sizeScales[startLevel])
        //zoomParams[startLevel]['r'])
        .attr("cx", function (d) {
            return xScale(d['L' + startLevel + 'x'])
        })
        .attr("cy", function (d) {
            return yScale(d['L' + startLevel + 'y'])
        })
        .attr("class", "scatter")
        .attr("stroke-width", zoomParams[startLevel]['w'])
        .style('fill', d => colorScale(d[IMDB_RATING]))
        .style('opacity', 1.0);


    //Transition to end points
    g.selectAll('.scatter').transition().duration(1000)
        .attr("cx", function (d) {
            return xScale(d['L' + endLevel + 'x'])
        })
        .attr("cy", function (d) {
            return yScale(d['L' + endLevel + 'y'])
        })
        .attr("r", zoomParams[endLevel]['r'])
        //.attr("r",sizeScales[endLevel])
        .attr("stroke-width", zoomParams[endLevel]['w'])
        .style('opacity', 1.0)
        .end()
        .then(() => drawGraph(payload[endLevel], -1, endLevel)); //don't forget to redraw when done!

}

//https://observablehq.com/@d3/zoom-to-bounding-box
function centerOnElement(px, py, k) {
    //Transition to center on x/y (pixel coords) at scale level k
    svg.transition().duration(750).call(
        myzoom.transform,
        d3.zoomIdentity
            .translate(width / 2, height / 2)
            .scale(k)
            .translate(-px, -py));
}

function resetZoom() {
    //Gets us back to default zoom
    centerOnElement(width / 2, height / 2, 1)
}


function getViewport(center, k) {
    // gets bbox of viewport centered on x,y, at zoom level k
    //returns bbox in x/y space
    const x = center.x;
    const y = center.y;
    const px = xScale(x);
    const py = yScale(y);
    const t = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(k)
        .translate(-px, -py);
    //console.log(t)
    return getBbox(t);

}

function getPtBbox(pts) {
    //given list pts (as IDs), find a bounding box for them at each level of zoom
    //returns x,y,k to center on the target points, in pixel space

    // Start with getting a bbox in x/y
    const movies = payload[5];
    const indices = pts.map(x => decoder[x]);
    const objs = [];
    indices.forEach(i => objs.push(movies[i]));
    const levelCoords = {};
    const centers = {};
    for (let lvl = 0; lvl < 6; lvl++) {
        levelCoords[lvl] = objs.map(function (d) {
            return {x: d['L' + lvl + 'x'], y: d['L' + lvl + 'y']}
        });
        let xs = levelCoords[lvl].map(d => d.x);
        let ys = levelCoords[lvl].map(d => d.y);
        centers[lvl] = {x: (d3.max(xs) + d3.min(xs)) / 2, y: (d3.max(ys) + d3.min(ys)) / 2}
    }

    console.log(levelCoords);
    console.log(centers);
    // Test zoom levels
    let best = {x: centers[0].x, y: centers[0].y, k: 1};
    for (let k = 1.5; k < zoomParams.maxZoom; k++) {

        let zoomLevel = zScale(k);
        //console.log('checking k='+k+' zoomlevel='+zoomLevel)
        let levelCoord = levelCoords[zoomLevel];
        let center = centers[zoomLevel];
        let bbox = getViewport(center, k);
        //console.log(bbox)
        //console.log(levelCoord)
        // let debug = levelCoord.forEach(foo => inBbox(foo, bbox));
        //console.log(debug)
        if (levelCoord.every(d => inBbox(d, bbox))) {
            best = {x: center.x, y: center.y, k: k}
        }

    }
    return best;
}

function getPtsClusterIDatLevel(pts, level) {
    //at a given zoom level, get the cluster IDs for each point in pts
    const tmp = getPtBbox(pts);
    const movies = payload[5];
    const levelStr = 'L' + zScale(tmp.k);
    const indices = pts.map(x => decoder[x]);
    const objs = [];
    indices.forEach(i => objs.push(movies[i]));
    return objs.map(d => d[levelStr]);
}

function highlightAndCenter(pts) {
    //given a list of movie IDs in points, highlight the clusters containing these things and center screen on them
    const ptBox = getPtBbox(pts);
    const level = zScale(ptBox.k);
    const clusters = getPtsClusterIDatLevel(pts, level);
    console.log('ptbox', ptBox);

    centerOnElement(xScale(ptBox.x), yScale(ptBox.y), ptBox.k);
    if (level < 5) {
        d3.selectAll('.scatter')
            .attr('class', function (d) {
                if (clusters.includes(d.ID)) {
                    return "scatter selected"
                } else {
                    return "scatter"
                }
            })
    } else {
        d3.selectAll('.scatter')
            .attr('class', function (d) {
                if (pts.includes(d[MOVIE_ID])) {
                    return "scatter selected"
                } else {
                    return "scatter"
                }
            })

    }


}

function array2list(str) {
    const arr = JSON.parse(str);
    return arr.map(itm => itm[0] + " (" + itm[1] + ")")
}

function str2arrayList(str) {
    let tmp = str.split('|');
    return tmp.join(', ')


}


function toolTipContentsCluster(d) {
    //console.log(d.ID)
    return '<p>Cluster: ' + (d.ID.toString()) + ' </p>' + '<p>Frequent Actors (Frequency): '
        + array2list(d[ACTORS]).slice(0, 5) + '</p><p>Frequent Genres (Frequency): '
        + array2list(d[GENRES]).slice(0, 5) + '</p><p>Average IMDB rating: ' + d[IMDB_RATING].toFixed(2) + '</p><p>Number of movies: ' + d[CLUSTER_SIZE] + '</p>'
}

function toolTipContentsMovie(d) {
    return '<p>Title: ' + d[MOVIE_TITLE] + '</p>' + '<p>Actors: '
        + str2arrayList(d[ACTORS]) + '</p><p>Genres: '
        + str2arrayList(d[GENRES]) + '</p><p>IMDB rating: ' + d[IMDB_RATING] + '</p>'
        + '<p>Director: ' + d[DIRECTOR] + '</p><p>Metascore: ' + d[METASCORE] + '</p>'
        + '<img alt="" src=' + d[POSTER_URL] + ' class="smallImg"/>'
}

function tipdir(d) {
    const tx = d3.zoomTransform(d3.select(".holder").node());
    const cx = +this.attributes.cx.value;
    const cy = +this.attributes.cy.value;
    let tmp = tx.apply([cx, cy]);
    //console.log(tmp,width,height)
    //console.log(tx,cx,cy)
    let out = '';
    if (tmp[1] < (height / 2)) {
        out = out + 's'
    } else {
        out = out + 'n';
    }
    if (tmp[0] < (width / 2)) {
        out = out + 'e';
    } else {
        out = out + 'w';
    }

    //console.log(out)
    return out
}

function toolTipContents(d) {
    //console.log(d)
    if ('movie_title' in d) {
        return toolTipContentsMovie(d);
    } else {
        return toolTipContentsCluster(d);

    }
}


function selectHighlight() {
    //selects node and centers
    d3.event.stopPropagation();
    d3.selectAll('.scatter').attr('class', 'scatter');
    d3.select(this).attr('class', 'scatter selected');
    const px = d3.select(this).attr("cx");
    const py = d3.select(this).attr("cy");
    const k = d3.zoomTransform(svg.node()).k;
    console.log(px, py, k);
    centerOnElement(px, py, k * 1.3);

}




function tipoff(d) {
    // offset calculator for tip
    const tx = d3.zoomTransform(d3.select(".holder").node());
    const cx = +this.attributes.cx.value;
    const cy = +this.attributes.cy.value;
    let tmp = tx.apply([cx, cy]);
    //console.log(tmp,width,height)
    //console.log(tx,cx,cy)
    let out = [0, 0];
    //return [(this.getBBox().height / 2)-100, 50]
    if (tmp[1] < (height / 2)) {
        out[0] = 0
    } else {
        out[0] = -this.getBBox().height
    }
    if (tmp[0] < (width / 2)) {
        out[1] = this.getBBox().width
    } else {
        out[1] = -this.getBBox().width;
    }

    return out;

}

function applyLabelsClusters() {
    // hack to just put labels on all nodes
    const k = d3.zoomTransform(svg.node()).k;
    const lvl = zScale(k);
    let clusterLabel1 = function (d) {
        return JSON.parse(d[GENRES]).slice(0, 2).map(x => x[0]).join('/') + ', with';
    };
    let clusterLabel2 = function (d) {
        return JSON.parse(d[ACTORS]).slice(0, 2).map(x => x[0]).join(' & ');
    };
    //https://stackoverflow.com/questions/16701522/how-to-linebreak-an-svg-text-within-javascript
    g.selectAll('nodeLabels')
        .data(payload[lvl])
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[lvl].r * 1.5))
        .attr("class", "labels lvl" + lvl)
        .text(clusterLabel1)
        .style("font-size", (zoomParams[lvl].r / 2) + 'px')
        .attr('text-anchor', 'middle');

    g.selectAll('nodeLabels')
        .data(payload[lvl])
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[lvl].r * 1))
        .attr("class", "labels lvl" + lvl)
        .text(clusterLabel2)
        .style("font-size", (zoomParams[lvl].r / 2) + 'px')
        .attr('text-anchor', 'middle')

}

function applyLabelsMovies() {
    // hack to just put labels on all nodes

    //https://stackoverflow.com/questions/16701522/how-to-linebreak-an-svg-text-within-javascript
    g.selectAll('nodeLabels')
        .data(data)
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[5].r * 1.5))
        .text(d => d[MOVIE_TITLE])
        .attr("class", "labels lvl5")
        .style("font-size", (zoomParams[5].r / 2) + 'px')
        .attr('text-anchor', 'middle')


}

function abstractPathDraw(edge) {
    let dx = xScale(edge.target.x) - xScale(edge.source.x),
        dy = yScale(edge.target.y) - yScale(edge.source.y),
        dr = Math.sqrt(dx * dx + dy * dy);
    return "M" +
        xScale(edge.source.x) + "," +
        yScale(edge.source.y) + "A" +
        dr + "," + dr + " 0 0,1 " +
        xScale(edge.target.x) + "," +
        yScale(edge.target.y);
}

function drawArcs() {
    // draws arcs from current movie to each item in curerntGrid

    const k = d3.zoomTransform(svg.node()).k;
    const lvl = zScale(k);

    function innerDrawArcs(currentMovieLoc, currentGridLoc) {
        let links = [];
        currentGridLoc.forEach(function (loc) {
            links.push({source: currentMovieLoc, target: loc})
        });
        console.log(links);
        g.selectAll("path")
            .data(links)
            .enter()
            .append("path")
            .attr("d", abstractPathDraw)
            .style("fill", "None")
            .style("stroke", "Black")
            .attr("class", "arc")
            .style('stroke-width', zoomParams[lvl].r / 5)
    }

    if (lvl < 5) {
        let currentMovieCluster = data.filter(x => x.ID === currentMovie)[0]['L' + lvl];
        // console.log(currentMovieCluster);
        // console.log(currentGrid);
        let currentGridCluster = currentGrid.map(i => parseInt(i)).map(q => data.filter(x => x.ID === q)[0]['L' + lvl]);
        // console.log(currentGridCluster);
        let currentMovieLoc = payload[lvl][currentMovieCluster];
        // console.log(currentMovieLoc);
        let currentGridLoc = currentGridCluster.map(q => payload[lvl][q]);
        innerDrawArcs(currentMovieLoc, currentGridLoc);
    } else {
        let currentMovieLoc = data.filter(x => x[MOVIE_ID] === currentMovie)[0];
        console.log(currentMovieLoc);
        let currentGridLoc = data.filter(x => currentGrid.includes(x[MOVIE_ID]));
        console.log(currentGridLoc);
        innerDrawArcs(currentMovieLoc, currentGridLoc);
    }
}

function drawHistory() {
    // draws arcs in history of likes

    const k = d3.zoomTransform(svg.node()).k;
    const lvl = zScale(k);
    if (lvl < 5) {
        //let currentMovieCluster = data.filter(x=>x.ID==currentMovie)[0]['L'+lvl]
        //console.log(currentMovieCluster)
        let currentHistCluster = moviesLikedOrdered.map(q => data.filter(x => x.ID === q)[0]['L' + lvl]);
        //console.log(currentGridCluster)
        //let currentMovieLoc = payload[lvl][currentMovieCluster]
        //console.log(currentMovieLoc)
        let currentHistLoc = currentHistCluster.map(q => payload[lvl][q]);
        //console.log(currentGridLoc)
        let links = [];
        let i;
        let curr = currentHistLoc[0];
        for (i = 1; i < currentHistLoc.length; i++) {
            links.push({source: curr, target: currentHistLoc[i]});
            curr = currentHistLoc[i];
        }
        console.log(links);
        console.log('xxx');
        g.append('g')
            .attr("class", "history_paths").selectAll("path")
            .data(links)
            .enter()
            .append("path")
            .attr("d", function (d) {
                let dx = xScale(d.target.x) - xScale(d.source.x),
                    dy = yScale(d.target.y) - yScale(d.source.y),
                    dr = Math.sqrt(dx * dx + dy * dy);
                let q = "M" +
                    xScale(d.source.x) + "," +
                    yScale(d.source.y) + "A" +
                    dr + "," + dr + " 0 0,1 " +
                    xScale(d.target.x) + "," +
                    yScale(d.target.y);
                console.log(q);
                return "M" +
                    xScale(d.source.x) + "," +
                    yScale(d.source.y) + "A" +
                    dr + "," + dr + " 0 0,1 " +
                    xScale(d.target.x) + "," +
                    yScale(d.target.y);
            })
            .style("fill", "None")
            .style("stroke", "steelblue")
            .attr("class", "arc")
            .style('stroke-width', zoomParams[lvl].r / 5)
    } else {
        let i;
        let tmp;
        let links = [];
        let curr = data.filter(x => x.ID === moviesLikedOrdered[0])[0];
        for (i = 1; i < moviesLikedOrdered.length; i++) {
            tmp = data.filter(x => x.ID === moviesLikedOrdered[i])[0];
            links.push({source: curr, target: tmp});
            curr = tmp;
        }

        console.log(links);
        g.append('g')
            .attr("class", "history_paths").selectAll("path")
            .data(links)
            .enter()
            .append("path")
            .attr("d", abstractPathDraw)
            .style("fill", "None")
            .style("stroke", "steelblue")
            .attr("class", "arc")
            .style('stroke-width', zoomParams[lvl].r / 5)
    }
}


function zoomed() {
    d3.select(".d3-tip").remove();
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
    g.call(tip);
    g.selectAll('.scatter')
        .on('mouseover', tip.show)
        .on('mouseout', tip.hide)
}
