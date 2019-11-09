function bboxFilter(d, bbox, zoomLevel) {
    const x = d['L' + zoomLevel + 'x'];
    const y = d['L' + zoomLevel + 'y'];
    const biggerBbox = resizeBbox(bbox, true);
    const x_ok = x >= biggerBbox.left && x <= biggerBbox.right;
    const y_ok = y >= biggerBbox.bot && y <= biggerBbox.top;

    return x_ok && y_ok;
}

//https://stackoverflow.com/questions/42695480/d3v4-zoom-coordinates-of-visible-area
function getVisibleArea(t) {
    const ul = t.invert([0, 0]),
        lr = t.invert([width, height]);
    return {
        left: Math.trunc(ul[0]),
        bot: Math.trunc(ul[1]),
        right: Math.trunc(lr[0]),
        top: Math.trunc(lr[1])
    }
}

function resizeBbox(bbox, bigger) {
    const height = bbox.top - bbox.bot;
    const width = bbox.right - bbox.left;
    if (bigger) {
        return {
            top: bbox.top + height * bbox_pad,
            bot: bbox.bot - height * bbox_pad,
            left: bbox.left - width * bbox_pad,
            right: bbox.right + width * bbox_pad
        }
    } else {
        return {
            top: bbox.top - height * bbox_pad,
            bot: bbox.bot + height * bbox_pad,
            left: bbox.left + width * bbox_pad,
            right: bbox.right - width * bbox_pad
        }
    }
}