document.addEventListener("DOMContentLoaded", startup);

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

var currCursorPos = { x: 0, y: 0 };
var prevCursorPos = currCursorPos;
var dragging = false;

// store canvas data while submitting the form
var tempImage = new Image;
tempImage.onload = function() {
    ctx.drawImage(tempImage, 0, 0);
};
if (image.value != "") tempImage.src = image.value;

// keep scroll position while refreshing the page
window.onbeforeunload = function(e) {
    localStorage.setItem('scrollPosition', window.scrollY);
};

function startup() {
    const canvas = document.getElementById("canvas");

    // set up mouse events for drawing
    canvas.addEventListener("mousedown", function (e) {
        prevCursorPos = getCursorPos(e);
        dragging = true;
    }, false);
    canvas.addEventListener("mousemove", function (e) {
        if (dragging) {
            currCursorPos = getCursorPos(e);
            draw();
            prevCursorPos = currCursorPos;
        }
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        dragging = false;
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        dragging = false;
    }, false);

    // set up touch events for mobile, etc
    canvas.addEventListener("touchstart", function (e) {
        currCursorPos = getTouchPos(e);
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchmove", function (e) {
        var touch = e.touches[0];
        var mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }, false);
    canvas.addEventListener("touchend", function (e) {
        var mouseEvent = new MouseEvent("mouseup", {});
        canvas.dispatchEvent(mouseEvent);
    }, false);

	// prevent scrolling when touching the canvas
	document.body.addEventListener("touchstart", function (e) {
		if (e.target == canvas) {
			e.preventDefault();
		}
	}, false);
	document.body.addEventListener("touchmove", function (e) {
		if (e.target == canvas) {
			e.preventDefault();
		}
	}, false);
	document.body.addEventListener("touchend", function (e) {
		if (e.target == canvas) {
			e.preventDefault();
		}
	}, false);

    // restore scroll position
    var scrollPosition = localStorage.getItem('scrollPosition');
    if (scrollPosition) window.scrollTo(0, scrollPosition);
}

function getCursorPos(e) {
    const canvasBorderWidth = (canvas.offsetWidth - canvas.clientWidth) / 2;
    const canvasBorderHeight = (canvas.offsetHeight - canvas.clientHeight) / 2;
    var rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left - canvasBorderWidth) * canvas.width / canvas.clientWidth,
        y: (e.clientY - rect.top - canvasBorderHeight) * canvas.height / canvas.clientHeight
    };
}

function getTouchPos(e) {
    const canvasBorderWidth = (canvas.offsetWidth - canvas.clientWidth) / 2;
    const canvasBorderHeight = (canvas.offsetHeight - canvas.clientHeight) / 2;
    var rect = canvas.getBoundingClientRect();
    return {
        x: (e.touches[0].clientX - rect.left - canvasBorderWidth) * canvas.width / canvas.clientWidth,
        y: (e.touches[0].clientY - rect.top - canvasBorderHeight) * canvas.height / canvas.clientHeight
    };
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevCursorPos.x, prevCursorPos.y);
    ctx.lineTo(currCursorPos.x, currCursorPos.y);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    output1.innerHTML = '&nbsp';
    output2.innerHTML = '&nbsp';
}

function submitForm() {
    if (isCanvasBlank()) {
        output1.innerHTML = '<span style="color:red">Error</span>';
        output2.innerHTML = 'Can\'t send blank image';
    } else {
        image.value = canvas.toDataURL();
        document.getElementById('myForm').submit();
        document.getElementById('submitBtn').disabled = true;
    }
}

// returns true if every pixel's uint32 representation is 0xFFFFFFFF
function isCanvasBlank() {
    const pixelBuffer = new Uint32Array(
        ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
    );
    return !pixelBuffer.some(color => color !== 0xFFFFFFFF);
}
