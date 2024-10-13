const canvas = document.getElementById('segmentationCanvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('localVideo');
let net;

async function loadBodyPix() {
    net = await bodyPix.load();
}

async function segmentBodyInRealTime() {
    const segmentation = await net.segmentPerson(video);

    const mask = bodyPix.toMask(segmentation);
    const faceOnly = new ImageData(new Uint8ClampedArray(mask.data), mask.width, mask.height);
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw the segmented face
    ctx.putImageData(faceOnly, 50, 50);
    
    requestAnimationFrame(segmentBodyInRealTime);
}

loadBodyPix().then(segmentBodyInRealTime);
