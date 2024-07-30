
// Function to display the selected file name and preview the image
async function displayFileName(input) {
    const reader = new FileReader();
    const imageContainer = document.getElementById('uploaded-image');
    const loadingOverlay = document.querySelector('.loading-overlay');

    reader.onload = async function (e) {
        imageContainer.src = e.target.result;
        imageContainer.style.display = 'block';
        document.getElementById('ocr-table').style.display = 'block';

        loadingOverlay.style.display = 'flex';

        const formData = new FormData();
        formData.append('file', input.files[0]);
        formData.append('equationType', document.querySelector('input[name="equationType"]:checked').value); // Include selected radio value

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById('result').innerText = `${data.result}`;
                document.getElementById('exp').innerHTML = `${data.exp}`;
            }
        } catch (error) {
            console.error('Error:', error);
        } finally {
            loadingOverlay.style.display = 'none';
        }
    };

    reader.readAsDataURL(input.files[0]);
}

// Function to save the canvas drawing
async function saveCanvas() {
    const canvas = document.getElementById("drawingCanvas");
    const imageData = canvas.toDataURL();
    const loadingOverlay = document.querySelector('.loading-overlay');

    const base64Data = imageData.replace(/^data:image\/\w+;base64,/, '');
    const padding = '='.repeat((4 - base64Data.length % 4) % 4);
    const base64EncodedString = base64Data + padding;

    const formData = new FormData();
    formData.append('image', base64EncodedString);
    formData.append('equationType', document.querySelector('input[name="equationType"]:checked').value); // Include selected radio value

    loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('result').innerText = `${data.result}`;
            document.getElementById('exp').innerHTML = `${data.exp}`;
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        loadingOverlay.style.display = 'none';
    }

    const imageContainer = document.getElementById('uploaded-image');
    imageContainer.src = imageData;
    imageContainer.style.display = 'block';
    document.getElementById('ocr-table').style.display = 'block';
}

// Function to start camera preview when modal is shown
$('#cameraModal').on('shown.bs.modal', function (e) {
    console.log('Camera modal shown, starting camera preview...');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            var cameraPreview = document.getElementById('cameraPreview');
            cameraPreview.srcObject = stream;
            cameraPreview.play();
            console.log('Camera preview started.');
        })
        .catch(function(error) {
            console.error('Error accessing camera:', error);
            alert('Error accessing camera. Please ensure that you have allowed camera access.');
        });
});

// Function to capture photo from camera preview and send it to the backend
async function capturePhoto() {
    console.log('Capturing photo...');
    var cameraPreview = document.getElementById('cameraPreview');
    var canvas = document.createElement('canvas');
    canvas.width = cameraPreview.videoWidth;
    canvas.height = cameraPreview.videoHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
    var imageData = canvas.toDataURL();

    var imageElement = document.getElementById('uploaded-image');
    imageElement.src = imageData;
    document.getElementById('ocr-table').style.display = 'block';
    console.log('Photo captured and updated in uploaded-image.');

    $('#cameraModal').modal('hide');

    var base64Data = imageData.split(',')[1];
    var padding = '='.repeat((4 - base64Data.length % 4) % 4);
    var base64EncodedString = base64Data + padding;

    const formData = new FormData();
    formData.append('image', base64EncodedString);
    formData.append('equationType', document.querySelector('input[name="equationType"]:checked').value); // Include selected radio value

    const loadingOverlay = document.querySelector('.loading-overlay');
    loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('result').innerText = `${data.result}`;
            document.getElementById('exp').innerHTML = `${data.exp}`;
        }
    } catch (error) {
        console.error('Error:', error);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Function to close camera modal and release camera resources
$('#cameraModal').on('hidden.bs.modal', function (e) {
    console.log('Closing camera modal...');
    var cameraPreview = document.getElementById('cameraPreview');
    if (cameraPreview.srcObject) {
        cameraPreview.srcObject.getTracks().forEach(track => track.stop());
        cameraPreview.srcObject = null;
        console.log('Camera resources released.');
    }
});

// Canvas Drawing Functionality

var color = "rgb(0, 0, 0)";
var lineWidth = 5;
var $canvas = $("canvas");
var context = $canvas[0].getContext("2d");
var lastEvent;
var mouseDown = false;
var points = [];

$(document).ready(function() {
    $('#exampleModal').on('shown.bs.modal', function () {
        $("#clearButton").trigger("click");
    });
});

function setWhiteBackground() {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, $canvas.width(), $canvas.height());
}

$canvas.on("touchstart touchmove", function (e) {
    e.preventDefault();
});

$canvas.mousedown(function (e) {
    lastEvent = e;
    mouseDown = true;
}).mousemove(function (e) {
    if (mouseDown) {
        drawLine(lastEvent.offsetX, lastEvent.offsetY, e.offsetX, e.offsetY);
        lastEvent = e;
    }
}).mouseup(function () {
    mouseDown = false;
}).mouseleave(function () {
    $canvas.mouseup();
});

$canvas.on("touchstart", function (e) {
    lastEvent = e.originalEvent.touches[0];
    mouseDown = true;
}).on("touchmove", function (e) {
    if (mouseDown) {
        var touch = e.originalEvent.touches[0];
        drawLine(lastEvent.pageX - $canvas.offset().left, lastEvent.pageY - $canvas.offset().top, touch.pageX - $canvas.offset().left, touch.pageY - $canvas.offset().top);
        lastEvent = touch;
    }
}).on("touchend touchcancel", function () {
    mouseDown = false;
});

function drawLine(startX, startY, endX, endY) {
    context.beginPath();
    context.moveTo(startX, startY);
    context.lineTo(endX, endY);
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.stroke();
    
    points.push({ startX: startX, startY: startY, endX: endX, endY: endY });
}

$("#clearButton").click(function () {
    setWhiteBackground();
    points = [];
});

function redrawLines() {
    setWhiteBackground();
    points.forEach(function(point) {
        drawLine(point.startX, point.startY, point.endX, point.endY);
    });
}
