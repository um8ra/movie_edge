function inputFormat (r) {
    r[MOVIE_TITLE] = decodeURIComponent(r[MOVIE_TITLE]);
    r[DIRECTOR] = decodeURIComponent(r[DIRECTOR]);
    r[ACTORS] = decodeURIComponent(r[ACTORS]);
    return r
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