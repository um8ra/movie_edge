function inputFormat(r) {
    r[MOVIE_TITLE] = decodeURIComponent(r[MOVIE_TITLE]);
    r[DIRECTOR] = decodeURIComponent(r[DIRECTOR]);
    r[ACTORS] = decodeURIComponent(r[ACTORS]);
    return r
}

function getCookie(name) {
    // https://docs.djangoproject.com/en/2.2/ref/csrf/
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function abstractFetch(fetchPayload, fetchURL, handleKey, handleFunction) {
    console.log('Button Clicked');
    // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
    const fetchParams = {
        method: 'POST',
        headers: {
            "X-CSRFToken": getCookie('csrftoken'),
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(fetchPayload),
        credentials: 'same-origin'
    };
    console.log(fetchParams);
    // https://scotch.io/tutorials/how-to-use-the-javascript-fetch-api-to-get-data
    // https://developers.google.com/web/fundamentals/primers/promises
    fetch(fetchURL, fetchParams)
        .then((response) => response.json())
        .then(function (data) {
            console.log(data);
            // Annoyingly, the data MUST be handled in the "then"
            // You can't return values from this area.
            // Hence the further function and key abstraction
            // The key is due to the data being JSON
            handleFunction(data[handleKey]);
        })
}

function buttonClickGetRandom(fetchURL) {
    const fetchPayload = Object();
    fetchPayload[LIKE] = Array();
    fetchPayload[DISLIKE] = Array();
    fetchPayload[MOVIES_SHOWN] = Array.from(gridHistorySet);
    abstractFetch(fetchPayload, fetchURL, MOVIE_CHOICES, gridMovies);
}

function buttonClickSubmit(fetchURL) {
    const fetchPayload = Object();
    fetchPayload[LIKE] = Array.from(moviesLikedSet);
    fetchPayload[DISLIKE] = Array.from(moviesDislikedSet);
    fetchPayload[MOVIES_SHOWN] = Array.from(gridHistorySet);
    abstractFetch(fetchPayload, fetchURL, MOVIE_CHOICES, gridMovies);
}

function closurePlotGoTo(movieID) {
    // movieID isn't actually passed to plotGoTo
    // but since it's a closure, plotGoTo understands the
    // scope it was *created* in!
    
    function plotGoTo() {
        
        console.log('PlotGoTo Current Movie: ', movieID);
        currentMovie = parseInt(movieID);

        const transform = getTransform();
        const k = transform.k;
        const lvl = zScale(k);

        if (lvl === 5) {
            highlightAndCenterSingle(movieID);
        }
        else {
            highlightAndCenter([movieID]);
        }
        refreshGridBorders()
    }

    return plotGoTo;
}


function refreshGridBorders(){
    var nodes = document.querySelectorAll('.poster');
    let specialID = "ID"+currentMovie
    for (let i=0; i<nodes.length; i++) {
        //console.log(nodes[i].classList)
        nodes[i].classList.remove('selected')
        if (nodes[i].classList.contains(specialID)) {
            nodes[i].classList.add('selected')
        }
    }
    
    
    
    
    
}



function gridMovies(movieidList) {
    // const k = d3.zoomTransform(svg.node()).k;
    // const lvl = zScale(k);
    svg.call(tip);
    // highlightAndCenter(movieidList);
    // highlightAndCenter(movieidList.concat([currentMovie]));

    // const ratio = 1.48;
    const divHeight = document.getElementById("grid").offsetHeight;
    // const divHeightNormalized = (divHeight - 128) / 5 / ratio;
    // const divWidth = document.getElementById("grid").offsetWidth / 2.75;

    let poster_height = Math.min((divHeight / 5) - 50, 150);
    let poster_width = (poster_height * 2 / 3);
    // console.log("Thumbnails: ", divHeight, poster_height, poster_width);
    // let poster_width = 100;
    // let poster_height = 150;
    // if (divHeightNormalized > divWidth) {
    //     // alert('width limited');
    //     poster_width = divWidth;
    //     poster_height = divWidth * ratio;
    // } else {
    //     // alert('height limited');
    //     poster_height = divHeightNormalized;
    //     poster_width = poster_height / ratio;
    // }
    // https://stackoverflow.com/questions/3955229/remove-all-child-elements-of-a-dom-node-in-javascript
    const grid = document.getElementById(gridID);
    while (grid.firstChild) {
        grid.removeChild(grid.firstChild);
    }
    // https://stackoverflow.com/questions/2735881/adding-images-to-an-html-document-with-javascript
    currentGrid = movieidList.map(x => parseInt(x));
    movieidList.map(x => gridHistorySet.add(x));
    movieidList.forEach(function (movieId) {
            const dataIndex = decoder[movieId];
            // console.log(dataIndex);
            const divNode = document.createElement('div');
            const imgNode = document.createElement('img');
            const spanNode = document.createElement('span')
            const buttonLike = document.createElement('BUTTON');
            buttonLike.innerText = 'Like';
            buttonLike.onclick = buttonClickLike;
            buttonLike.value = movieId;
            buttonLike.id = 'LikeBtn_' + movieId;
            buttonLike.className = 'buttonLike';

            const buttonDislike = document.createElement('BUTTON');
            buttonDislike.innerText = 'Dislike';
            buttonDislike.onclick = buttonClickDislike;
            buttonDislike.value = movieId;
            buttonDislike.id = 'DislikeBtn_' + movieId;
            buttonDislike.className = 'buttonDislike';

            const thisMovieData = data[dataIndex];
            const movieTitle = thisMovieData[MOVIE_TITLE];
            imgNode.src = thisMovieData[POSTER_URL];
            imgNode.alt = movieTitle;
            imgNode.title = movieTitle;
            
            imgNode.height = poster_height;
            imgNode.width = poster_width;
            const customID = 'ID'+thisMovieData['ID']
            imgNode.classList.add(customID);
            //d3.select('.customID').datum(thisMovieData)
            
            // Don't know why these classes are getting added but no effect?
            imgNode.classList.add("poster");
            imgNode.ondblclick = closurePlotGoTo(movieId);
            if (movieId === currentMovie){
                imgNode.classList.add("selected");
            } 
            //imgNode.style.border='2px solid #FFF';
            
            spanNode.className = 'likeDislikeSpan';
            spanNode.appendChild(buttonLike);
            spanNode.appendChild(buttonDislike);
            grid.appendChild(divNode);
            divNode.appendChild(imgNode);
            divNode.appendChild(spanNode);

            // divNode.appendChild(buttonLike);
            // divNode.appendChild(buttonDislike);
            // divNode.appendChild(buttonShowMe);
            // console.log(movieId);
        }
    );
    drawGraph(true);
}

function buttonClickLike(data) {
    const movieId = data.target.value;
    if (moviesLikedSet.has(movieId)) {
        moviesLikedSet.delete(movieId);
        let theButton = d3.select(this);
        theButton.style('background', 'white');
        theButton.style('color', 'black');
    } else {
        if (moviesDislikedSet.has(movieId)) {
            moviesDislikedSet.delete(movieId);
            let theButton = d3.select('#DislikeBtn_' + movieId);
            theButton.style('background', 'white');
            theButton.style('color', 'black');
        }
        moviesLikedSet.add(movieId);
        moviesLikedOrdered.push(movieId);
        let theButton = d3.select(this);
        theButton.style('background', 'green');
        theButton.style('color', 'white');
    }
    highlight([currentMovie]); // selected, uncertain, liked, disliked, vs ingrid
    console.log('Liked: ', moviesLikedSet);
    console.log('Disiked: ', moviesDislikedSet);
}

function buttonClickDislike(data) {
    const movieId = data.target.value;
    if (moviesDislikedSet.has(movieId)) {
        moviesDislikedSet.delete(movieId);
        let theButton = d3.select(this);
        theButton.style('background', 'white');
        theButton.style('color', 'black');
    } else {
        if (moviesLikedSet.has(movieId)) {
            moviesLikedSet.delete(movieId);
            let theButton = d3.select('#LikeBtn_' + movieId);
            theButton.style('background', 'white');
            theButton.style('color', 'black');
        }
        moviesDislikedSet.add(movieId);
        moviesDislikedOrdered.push(movieId);
        let theButton = d3.select(this);
        theButton.style('background', 'red');
        theButton.style('color', 'white');
    }
    highlight([currentMovie]); // selected, uncertain, liked, disliked, vs ingrid
    console.log('Liked: ', moviesLikedSet);
    console.log('Disiked: ', moviesDislikedSet);
}

function findMovie(formBox) {
    const matchString = formBox[0].value;
    const re = new RegExp(matchString, 'i');
    const stringMatches = data.filter(x => re.test(x[MOVIE_TITLE]));
    if (Array.isArray(stringMatches) && stringMatches.length > 0) {
        movieID = stringMatches[0][MOVIE_ID]
        currentMovie = parseInt(movieID);

        // highlightAndCenter([movieID]);

        // Though highlightAndCenter([movieID]); is visually equivalent
        // use lvl check as shown below, similar to closurePlotGoTo()
        // '...Single' method avoids looping through L0 to L5 to find best
        const transform = getTransform();
        const k = transform.k;
        const lvl = zScale(k);

        if (lvl === 5) {
            highlightAndCenterSingle(movieID);
        }
        else {
            highlightAndCenter([movieID]);
        }

    }
    formBox.reset();
}