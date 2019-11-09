function inputFormat (r) {
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

function abstractFetch(fetchPayload, fetchURL) {
    console.log('Button Clicked');
    // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
    const fetchParams = {
        method: 'POST',
        headers: {
            "X-CSRFToken": getCookie("csrftoken"),
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
            const movieChoices = data[MOVIE_CHOICES];
            console.log(movieChoices);
            gridNine(movieChoices);
        })
}

function gridNine(movieidList) {
    // Delete current grid and redraw with new *choices*
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
            imgNode.height = POSTER_HEIGHT;
            imgNode.width = POSTER_WIDTH;
            grid.appendChild(divNode);
            divNode.appendChild(imgNode);
            divNode.appendChild(buttonLike);
            divNode.appendChild(buttonDislike);
            console.log(movieId);
        }
    )
}

function buttonClickLike(data) {
    const movieId = data.target.value;
    if (moviesDisliked.has(movieId)) {
        moviesDisliked.delete(movieId);
    }
    moviesLiked.add(movieId);
    console.log(moviesLiked);
}

function buttonClickDislike(data) {
    const movieId = data.target.value;
    if (moviesLiked.has(movieId)) {
        moviesLiked.delete(movieId);
    }
    moviesDisliked.add(movieId);
    console.log(moviesDisliked);
}