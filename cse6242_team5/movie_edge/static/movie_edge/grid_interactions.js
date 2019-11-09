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