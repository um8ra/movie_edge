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
    abstractFetch(fetchPayload, fetchURL, MOVIE_CHOICES, gridMovies);
}

function buttonClickSubmit(fetchURL) {
    const fetchPayload = Object();
    fetchPayload[LIKE] = Array.from(moviesLiked);
    fetchPayload[DISLIKE] = Array.from(moviesDisliked);
    abstractFetch(fetchPayload, fetchURL, MOVIE_CHOICES, gridMovies);
}

function closurePlotGoTo(movieID) {
    // movieID isn't actually passed to plotGoTo
    // but since it's a closure, plotGoTo understands the
    // scope it was *created* in!
    function plotGoTo() {
        console.log(movieID);
        console.log('do stuff with movieID')
    }

    return plotGoTo;
}

function gridMovies(movieidList) {
    const ratio = 1.48;
    const divHeight = document.getElementById("grid").offsetHeight;
    const divHeightNormalized = (divHeight - 128) / 5 / ratio;
    const divWidth = document.getElementById("grid").offsetWidth / 2.75;

    let poster_width = 100;
    let poster_height = 150;
    if(divHeightNormalized > divWidth) {
        // alert('width limited');
        poster_width = divWidth;
        poster_height = divWidth * ratio;
    } else {
        // alert('height limited');
        poster_height = divHeightNormalized;
        poster_width = poster_height / ratio;
    }
    // https://stackoverflow.com/questions/3955229/remove-all-child-elements-of-a-dom-node-in-javascript
    const grid = document.getElementById(gridID);
    while (grid.firstChild) {
        grid.removeChild(grid.firstChild);
    }
    // https://stackoverflow.com/questions/2735881/adding-images-to-an-html-document-with-javascript
    movieidList.forEach(function (movieId) {
            const dataIndex = decoder[movieId];
            console.log(dataIndex);
            const divNode = document.createElement('div');
            const imgNode = document.createElement('img');

            const buttonLike = document.createElement('BUTTON');
            buttonLike.innerText = 'Like';
            buttonLike.onclick = buttonClickLike;
            buttonLike.value = movieId;

            const buttonDislike = document.createElement('BUTTON');
            buttonDislike.innerText = 'Dislike';
            buttonDislike.onclick = buttonClickDislike;
            buttonDislike.value = movieId;

            imgNode.src = data[dataIndex][POSTER_URL];
            imgNode.onclick = closurePlotGoTo(movieId);
            imgNode.height = poster_height;
            imgNode.width = poster_width;
            grid.appendChild(divNode);
            divNode.appendChild(imgNode);
            divNode.appendChild(buttonLike);
            divNode.appendChild(buttonDislike);
            // divNode.appendChild(buttonShowMe);
            console.log(movieId);
        }
    )
}

function buttonClickLike(data) {
    const movieId = data.target.value;
    console.log('Liked: ' + movieId);
    if (moviesDisliked.has(movieId)) {
        moviesDisliked.delete(movieId);
    }
    moviesLiked.add(movieId);
    console.log(moviesLiked);
}

function buttonClickDislike(data) {
    const movieId = data.target.value;
    console.log('Disliked: ' + movieId);
    if (moviesLiked.has(movieId)) {
        moviesLiked.delete(movieId);
    }
    moviesDisliked.add(movieId);
    console.log(moviesDisliked);
}