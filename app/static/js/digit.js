var output1 = document.getElementById("output1");
var output2 = document.getElementById("output2");
var image = document.getElementById("image");
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

ctx.lineWidth = 2;
ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.strokeStyle = "black";
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

var prevX = 0,
    currX = 0,
    prevY = 0,
    currY = 0,
    flag = false,
    canvasWidthRatio = canvas.clientWidth / canvas.width,
    canvasHeightRatio = canvas.clientHeight / canvas.height,
    canvasBorderWidth = (canvas.offsetWidth - canvas.clientWidth) / 2,
    canvasBorderHeight = (canvas.offsetHeight - canvas.clientHeight) / 2;

canvas.addEventListener("mousemove", e => findxy('move', e), false);
canvas.addEventListener("mousedown", e => findxy('down', e), false);
canvas.addEventListener("mouseup", e => findxy('up', e), false);
canvas.addEventListener("mouseout", e => findxy('out', e), false);

// store canvas data while submitting the form
var var_img = new Image;
var_img.onload = function() {
    ctx.drawImage(var_img, 0, 0);
}
if (image.value != "") {
    var_img.src = image.value;
}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = getX(e);
        currY = getY(e);
        flag = true;
    }
    if (res == 'up' || res == "out") flag = false;
    if (res == 'move' && flag) {
        prevX = currX;
        prevY = currY;
        currX = getX(e);
        currY = getY(e);
        draw();
    }
}

const getX = (e) => (e.clientX - canvas.getBoundingClientRect().left - canvasBorderWidth) / canvasWidthRatio;
const getY = (e) => (e.clientY - canvas.getBoundingClientRect().top - canvasBorderHeight) / canvasHeightRatio;

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    output1.innerHTML = '&nbsp';
    output2.innerHTML = '&nbsp';
}

function submitCanvas() {
    if (isCanvasBlank()) {
        output1.innerHTML = '<span style="color:red">Error</span>';
        output2.innerHTML = 'Can\'t send blank image';
    } else {
        image.value = canvas.toDataURL();
        document.getElementById('myForm').submit();
    }
}

// returns true if every pixel's uint32 representation is 0xFFFFFFFF
function isCanvasBlank() {
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    return !pixelBuffer.some(color => color !== 0xFFFFFFFF);
}

// to keep scroll position while refreshing the page
document.addEventListener("DOMContentLoaded", function(event) {
    var scrollpos = localStorage.getItem('scrollpos');
    if (scrollpos) window.scrollTo(0, scrollpos);
});
window.onbeforeunload = function(e) {
    localStorage.setItem('scrollpos', window.scrollY);
};
