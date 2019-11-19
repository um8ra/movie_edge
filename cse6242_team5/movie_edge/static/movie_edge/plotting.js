function inputFormatCluster(r) { // Decodes cluster data    
    r[GENRES] = decodeURIComponent(r[GENRES]);
    r[ACTORS] = decodeURIComponent(r[ACTORS]);
    return r
}

function highlight(ids) { //flags selected, uncertain, liked, disliked, vs ingrid
    const H = getTransform();
    const lvl = zScale(H.k);
    const centerID = moviesToLevelID([currentMovie], lvl)[0];
    const gridIDs = moviesToLevelID(currentGrid, lvl);
    const likedIDs = moviesToLevelID(Array.from(moviesLikedSet).map(x => parseInt(x)), lvl);
    const disLikedIDs = moviesToLevelID(Array.from(moviesDislikedSet).map(x => parseInt(x)), lvl);

    d3.selectAll('.scatter').attr("class", function(d) {
        isLiked = likedIDs.includes(d.ID);
        isDisliked = disLikedIDs.includes(d.ID);

        if (d.ID === centerID) {
            return "scatter selected";
        } else if (isLiked && isDisliked) {
            return "scatter uncertain";
        } else if (isLiked) {
            return "scatter liked";
        } else if (isDisliked) {
            return "scatter disliked";
        } else if (gridIDs.includes(d.ID)) {
            return "scatter ingrid";
        } else if (ids.includes(d.ID)) {
            return "scatter selected2"; // reserved class selected2
        } else {
            return 'scatter';
        }
    });
}

function toggleHighlight() { // if node is selected unselect, else select
    const me = d3.select(this);
    if (me.attr("class") === "scatter selected") {
        me.attr("class", 'scatter')

    } else {
        me.attr("class", 'scatter selected')
    }
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

function getTransform() { // returns a d3 zoom transform object representing current viewstate
    return d3.zoomTransform(svg.node()) // dunno why but it can't do this on g.
}

function moviesToLevelID(movies, lvl) {// returns list of cluster ids at lvl corresponding to movies
    // const lvl_name = 'L' + lvl;

    // sample check: 'ID' does not match with 'L5' attr for payload[5][0]
    // Changing to next line fixes node highlights at L5.
    const lvl_name = lvl === 5 ? 'ID' : 'L' + lvl;
    const movie_info = payload[5];
    return movies.map(x => movie_info[decoder[x]][lvl_name])  // decoder[x] => index of element, movie_info[?] => row of movie , we're getting the element L<lvl>
}

function clusterOrMovie(movieItm, clusterItm, lvl) { //returns movieItm if lvl == 5 else clusterItm
    return lvl === 5 ? movieItm : clusterItm
}

function getBbox(t) {// Given a  transform, get the bounding box in x/y space

    const pixel_bbox = getVisibleArea(t);
    const tmp = {
        top: yScale.invert(pixel_bbox.bot),
        bot: yScale.invert(pixel_bbox.top),
        left: xScale.invert(pixel_bbox.left),
        right: xScale.invert(pixel_bbox.right)
    };
    const height_ = tmp.top - tmp.bot;
    const width_ = tmp.right - tmp.left;
    return {
        top: tmp.top + height_ * bbox_pad,
        bot: tmp.bot - height_ * bbox_pad,
        left: tmp.left - width_ * bbox_pad,
        right: tmp.right + width_ * bbox_pad
    }
}

function getViewport(center, k) {// gets bbox (in x/y) of viewport centered on x,y, at zoom level k    
    const px = xScale(center.x);
    const py = yScale(center.y);
    // make transform and call getBbox
    const t = d3.zoomIdentity
        .translate(width / 2, height / 2)
        .scale(k)
        .translate(-px, -py);
    //console.log(t)
    return getBbox(t);

}

function inBbox(d, bbox) { // tests if point d is in bbox. Make sure both d and bbox are pixel or x/y  
    const x_ok = (d.x >= bbox.left) && (d.x <= bbox.right);
    const y_ok = (d.y >= bbox.bot) && (d.y <= bbox.top);
    return x_ok && y_ok;
}

function getPtBbox(pts) { //given list pts (as movie_ids), find a bounding box for them at each level of zoom, return x,y,k (x/y space) of center

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

//    console.log(levelCoords);
    //  console.log(centers);

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


function centerOnElement(px, py, k, now) { //Transition to center on x/y (pixel coords) at scale level k
    //https://observablehq.com/@d3/zoom-to-bounding-box
    const instant = typeof now !== 'undefined' ? now : false;
    const mytransform = d3.zoomIdentity
        .translate(width / 2, height / 2) // center to origin
        .scale(k) // scale
        .translate(-px, -py); // origin back to center
    const lvl = zScale(k);
    const applyLabels = clusterOrMovie(applyLabelsMovies, applyLabelsClusters, lvl);
    if (instant) {
        g.attr("transform", mytransform)
    } else {
        svg.transition().duration(750)
            .call(myzoom.transform, mytransform)
            .end(applyLabels)

    }

    applyLabels()
}

function highlightAndCenter(pts) { //given a list of movie IDs in points, highlight the clusters containing these things and center screen on them

    const ptBox = getPtBbox(pts); //coords in x/y
    const level = zScale(ptBox.k);
    const clusters = moviesToLevelID(pts, level);
    console.log('ptbox', ptBox);

    centerOnElement(xScale(ptBox.x), yScale(ptBox.y), ptBox.k);
    highlight(clusters);

}

function highlightAndCenterSingle(id) { //highlights a movie (not toggle) and centers on it. Does not change zoom level (so a cluster could be highlighted). Does not change any other highlighting
    const transform = getTransform();
    const k = transform.k;
    const lvl = zScale(k);
    const centerID = moviesToLevelID([id], lvl)[0];

    const row = payload[5][decoder[id]]; // information
    const px = xScale(row['L' + lvl + 'x']);
    const py = yScale(row['L' + lvl + 'y']);

    centerOnElement(px, py, k);
    highlight([centerID]);
}

function applyLabelsClusters() {//  put labels on visible clusters
    const transform = getTransform();
    const lvl = zScale(transform.k);
    const mydat = payload[lvl]; // data for this clustering level

    const bbox = getBbox(transform); // viewable area
    const items = mydat.filter(x => inBbox(x, bbox));  //stuff to plot
    console.log('applying ' + items.length + ' labels');

    let clusterLabel1 = function (d) { // 1st part of cluster label
        try {
            return JSON.parse(d[GENRES]).filter(d => d[0] !== '(no genres listed)').slice(0, 2).map(x => x[0]).join('/') + ', with';
        } catch (e) {
            return 'Unknown Genre(s), with'
        }
    };
    let clusterLabel2 = function (d) { // 2nd part of cluster label
        try {
            return JSON.parse(d[ACTORS]).filter(d => d[0] !== 'N/A').slice(0, 2).map(x => x[0]).join(' & ');
        } catch (e) {
            return ' unknown performers'
        }
    };

    g.selectAll('.labels').remove();

    //https://stackoverflow.com/questions/16701522/how-to-linebreak-an-svg-text-within-javascript
    g.selectAll('.nodeLabels')
        .data(items)
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[lvl].r * 1.5))
        .attr("class", "labels lvl" + lvl)
        .text(clusterLabel1)
        .style("font-size", (zoomParams[lvl].r / 2) + 'px')
        .attr('text-anchor', 'middle');

    g.selectAll('.nodeLabels')
        .data(items)
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[lvl].r * 1))
        .attr("class", "labels lvl" + lvl)
        .text(clusterLabel2)
        .style("font-size", (zoomParams[lvl].r / 2) + 'px')
        .attr('text-anchor', 'middle')

}

function applyLabelsMovies() {//  put labels on visible movies
    const transform = getTransform();
    const lvl = 5;
    const mydat = payload[lvl]; // data for this clustering level

    const bbox = getBbox(transform); // viewable area
    const items = mydat.filter(x => inBbox(x, bbox));  //stuff to plot
    g.selectAll('.labels').remove();
    console.log('applying ' + items.length + ' labels');
    g.selectAll('.nodeLabels')
        .data(items)
        .enter()
        .append("svg:text")
        .attr("x", d => xScale(d.x))
        .attr("y", d => yScale(d.y) - (zoomParams[5].r * 1.5))
        .text(d => d[MOVIE_TITLE])
        .attr("class", "labels lvl5")
        .style("font-size", (zoomParams[5].r / 2) + 'px')
        .attr('text-anchor', 'middle')


}

function drawGraph(center) { //Redraw plot. if center is truthy, highlight/center the current active points (gridMovies+currentMovie)
    const transform = getTransform();
    const lvl = zScale(transform.k);
    const data = payload[lvl];
    let moviesToHighlight = currentGrid.slice();
    moviesToHighlight.push(currentMovie);
    // Next line isn't used anymore.
    // const clusters2Highlight = moviesToLevelID(moviesToHighlight, lvl);

    console.log('reset graph', transform);
    // remove current graph
    g.selectAll('.scatter').remove();
    g.selectAll('.labels').remove();
    g.selectAll('path').remove();

    //Got the data, now draw it.
    drawArcs();
    // TODO: XXX
    g.selectAll('.scatter')
        .data(data)
        .enter()
        .append("circle")
        .attr("r", sizeScales[lvl])
        .attr("cx", d => xScale(d.x))
        .attr("cy", d => yScale(d.y))
        .attr("class", "scatter")
        .attr("stroke-width", zoomParams[lvl]['w'])
        .style('fill', d => colorScale(d[IMDB_RATING]))
        .on("dblclick.zoom", selectHighlight);

    g.selectAll('.scatter')
        .on('mouseover', tip.show)
        .on('mouseout', tip.hide);

    toCenter = typeof center === 'undefined' ? true : center;
    console.log(toCenter);
    if (toCenter) {
        // highlight and center grid
        // it also applies labels via centerOnElement -> clusterOrMovie
        //console.log('x',clusters2Highlight)
        highlightAndCenter(moviesToHighlight)
    } else {
        //labels
        const applyLabels = clusterOrMovie(applyLabelsMovies, applyLabelsClusters, lvl)
        applyLabels()
    }
    svg.call(tip)
}

function bboxFilter2Level(d, bbox, startLevel, endLevel) { // checks if coordinates of each d at startLevel or endLevel lie inside bbox

    const x_okS = (d['L' + startLevel + 'x'] >= bbox.left) && (d['L' + startLevel + 'x'] <= bbox.right);
    const x_okE = (d['L' + endLevel + 'x'] >= bbox.left) && (d['L' + endLevel + 'x'] <= bbox.right);
    const y_okS = (d['L' + startLevel + 'y'] >= bbox.bot) && (d['L' + startLevel + 'y'] <= bbox.top);
    const y_okE = (d['L' + endLevel + 'y'] >= bbox.bot) && (d['L' + endLevel + 'y'] <= bbox.top);

    return (x_okS && y_okS) || (x_okE && y_okE);
}

function animateClusters(startLevel, endLevel) { // Animates the cluster transitions between startLevel and endLevel

    let transform = getTransform();
    let lvl = zScale(transform.k);
    let movieData = payload[5];
    let bbox = getBbox(transform);

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
        .then(() => drawGraph(false)) //don't forget to redraw when done, but don't center
        .then(function () {
            let moviesToHighlight = currentGrid.slice();
            moviesToHighlight.push(currentMovie);
            const clusters2Highlight = moviesToLevelID(moviesToHighlight, lvl);
            highlight(clusters2Highlight)
        })
}

function zoomEnd() { // trigger events at end of zoom (aniamation and replotting labels)
    //d3.event.stopPropagation();
    const tx = getTransform();
    const k = tx.k;
    const newZoom = zScale(k);


    if (newZoom !== lastZoomLevel) {// handle zoom changes
        const bbox = getBbox(tx);
        // var data = payload[newZoom];
        console.log('Setting new zoom level ' + newZoom + ' ' + k);

        //animateClusters(data, bbox, currentZoom, newZoom);
        animateClusters(lastZoomLevel, newZoom);
        lastZoomLevel = newZoom;
        return

    }
    // REPOLOT LABELS seemed the best place for this. ugly, but trigger new labels after pan motion completes (NOT TICK BY TICK DURING PAN)
    const lvl = zScale(k);
    const applyLabels = clusterOrMovie(applyLabelsMovies, applyLabelsClusters, lvl);
    applyLabels()

}

function zoomed() { // apply zoom
    d3.select(".d3-tip").remove();
    // actions to take when zoom events triggered (largely mangaging zoom animation calls)
    const tx = d3.event.transform;
    //console.log(tx)
    g.attr("transform", tx);
    svg.call(tip);
}

function fullZoomUp() { // zoom all the way up to L0
    centerOnElement(width/2 + 90, height/2 + 60, 1);
}

function dynamicZoom() {//Resets zoom level and replots

    //g.attr("transform",d3.zoomIdentity)
    myzoom.transform(svg, d3.zoomIdentity);
    const tx = getTransform();
    console.log('trans', tx);
    const k = tx.k;
    drawGraph(true);
    const lvl = zScale(k);
    const applyLabels = clusterOrMovie(applyLabelsMovies, applyLabelsClusters, lvl);
    applyLabels()
}

function clusterToolTipHelperArray2list(str) { //string processing for cluster tooltips
    const arr = JSON.parse(str);
    return arr.filter(d => d[0] !== 'N/A').filter(d => d[0] !== '(no genres listed)').map(itm => itm[0] + " (" + itm[1] + ")")
}

function str2arrayList(str) {//string processing for movie tooltips
    let tmp = str.split('|');
    return tmp.join(', ')
}

function toolTipContentsCluster(d) { //html generator for cluster tooltips
    //console.log(d.ID)
    return '<p>Cluster: ' + (d.ID.toString()) + ' </p>' + '<p>Frequent Actors (Frequency): '
        + clusterToolTipHelperArray2list(d[ACTORS]).slice(0, 5) + '</p><p>Frequent Genres (Frequency): '
        + clusterToolTipHelperArray2list(d[GENRES]).slice(0, 5) + '</p><p>Average IMDB rating: ' + d[IMDB_RATING].toFixed(2) + '</p><p>Number of movies: ' + d[CLUSTER_SIZE] + '</p>'
}

function toolTipContentsMovie(d) {//html generator for movie tooltips
    return '<p>Title: ' + d[MOVIE_TITLE] + '</p>' + '<p>Actors: '
        + str2arrayList(d[ACTORS]) + '</p><p>Genres: '
        + str2arrayList(d[GENRES]) + '</p><p>IMDB rating: ' + d[IMDB_RATING] + '</p>'
        + '<p>Director: ' + d[DIRECTOR] + '</p><p>Metascore: ' + d[METASCORE] + '</p>'
        + '<img alt="" src=' + d[POSTER_URL] + ' class="smallImg"/>'
}

function tipdir(d) {
    const tx = getTransform();
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

function toolTipContents(d) { // selects contents of tooltip
    //console.log(d)
    if ('movie_title' in d) {
        return toolTipContentsMovie(d);
    } else {
        return toolTipContentsCluster(d);

    }
}

function selectHighlight() {//selects node and centers

    d3.event.stopPropagation();
	
	let tmp =  d3.select(this).data()[0].ID
	const lvl = zScale(getTransform().k)
	if (lvl === 5){
		console.log(d3.select(this).data()[0])
		currentMovie = d3.select(this).data()[0]['movie_id']
		tmp = currentMovie
	}
	console.log(tmp);
    if (d3.select(this).attr('class').includes('scatter')) {
        d3.select(this).attr('class', 'scatter selected2');
		highlight([tmp]); // reset highlights, otherwise multiple selected2
        //d3.select(this).attr('class', 'scatter selected2');
        const px = d3.select(this).attr("cx");
        const py = d3.select(this).attr("cy");
        const k = getTransform().k;
        console.log(px, py, k);
        centerOnElement(px, py, k * 1.3);
		console.log(tmp);
    }
	
	
	
    /*
    if ('movie_id' in d3.select(this)) {
        currentMovie = d3.select(this).movie_id;
    }*/
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

    const k = getTransform().k;
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

function drawLegend(){ // renders legend
	const frac = 0.75
	
	const canvasW = (width-2*padding)
	const canvasH = (height-2*padding)
	const legendW = canvasW * (1-frac)
	const legendH = canvasH * (1-frac)
	const leg = svg.append('g')
				.attr('transform','translate('+canvasW*frac+','+canvasH*frac+')');
	const legBG = leg.append('rect')
	.attr('height',legendH)
	.attr('width',legendW)
	.style('stroke','grey')
	.style('fill','white')
	const boxSize = (legendW-40)/11
	const legTitle = leg.append('text')
					.text('IMDB Rating')
					.attr('alignment-baseline','hanging' )
					.attr('x',boxSize*0.25)
					.attr('y',boxSize*0.25)
					.style('font-size',0.5*boxSize+'px')
	const legBoxes = [0,10,20,30,40,50,60,70,80,90,100]
	let i = 0
	
	for (i=0; i<11; i++){
		let x = i*boxSize + 20
		let y = boxSize
		leg.append('rect')
			.attr('height',boxSize)
			.attr('width',boxSize)
			.attr('x',x)
			.attr('y',y)
			.style('fill',colorScale(legBoxes[i]/10))
		if (i === 0 || !!(i && !(i%2))){
			leg.append('text')
			.attr('x',x)
			.attr('y',y+boxSize*1.3)
			.text((legBoxes[i]/10).toFixed(2))
			.attr('alignment-baseline','hanging' )
			.style('font-size',0.4*boxSize+'px')
		}
	}
	
	leg.append('circle')
		.attr('cx',20+boxSize/2)
		.attr('cy',boxSize*3.5)
		.attr('r',boxSize/2)
		.style('stroke','blue')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*3.5).text('Selected')
		.attr('x',20+boxSize*1.3)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
	leg.append('circle')
		.attr('cx',20+boxSize/2)
		.attr('cy',boxSize*5.0)
		.attr('r',boxSize/2)
		.style('stroke','steelblue')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*5.0).text('Recommended')
		.attr('x',20+boxSize*1.3)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
	leg.append('circle')
		.attr('cx',legendW/3+20+boxSize/2)
		.attr('cy',boxSize*3.5)
		.attr('r',boxSize/2)
		.style('stroke','darkgreen')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*3.5).text('Liked')
		.attr('x',20+legendW/3+boxSize*0.8+boxSize/2)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
	leg.append('circle')
		.attr('cx',legendW/3+20+boxSize/2)
		.attr('cy',boxSize*5)
		.attr('r',boxSize/2)
		.style('stroke','red')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*5).text('Disliked')
		.attr('x',20+legendW/3+boxSize*0.8+boxSize/2)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
		
	leg.append('circle')
		.attr('cx',2*legendW/3+20+0*boxSize/2)
		.attr('cy',boxSize*3.5)
		.attr('r',boxSize/2)
		.style('stroke','#e34a33')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*3.5).text('Mixed')
		.attr('x',20+2*legendW/3+boxSize*0.8+0*boxSize/2)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
		
	leg.append('circle')
		.attr('cx',2*legendW/3+20+0*boxSize/2)
		.attr('cy',boxSize*5)
		.attr('r',boxSize/2)
		.style('stroke','yellow')
		.style('fill','white')
		.style('stroke-width',3)
	leg.append('text')
		.attr('y',boxSize*5).text('Viewing')
		.attr('x',20+2*legendW/3+boxSize*0.8+0*boxSize/2)
		.attr('alignment-baseline','middle')
		.style('font-size',0.4*boxSize+'px')
	
}
