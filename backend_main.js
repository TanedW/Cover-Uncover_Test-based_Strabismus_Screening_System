// Import the functions you need from the SDKs you need
  import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
  import { getAnalytics } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-analytics.js";

  // Your web app's Firebase configuration
  const firebaseConfig = {
    apiKey: "AIzaSyCsbE-Bqc0YKwrNRd_94i5VxsgMJqJdBDM",
    authDomain: "control-embleded.firebaseapp.com",
    databaseURL: "https://control-embleded-default-rtdb.firebaseio.com",
    projectId: "control-embleded",
    storageBucket: "control-embleded.firebasestorage.app",
    messagingSenderId: "864593748062",
    appId: "1:864593748062:web:017c7a8ef4373be03894cf",
    measurementId: "G-0HJERJ7X93"
  };

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);

  // Cloud function URL
  const CLOUD_FUNCTION_URL = "https://controlled-j24s7uqjba-uc.a.run.app";

window.turnStart = async function() {
  if (window.isSending) {
    console.log("Already sending, please wait.");
    return;
  }
  window.isSending = true;

  let sending = true;

  async function sendLoop() {
    if (!sending) return;
    try {
      await sendImageToFastAPI();
    } catch(e) {
      console.error(e);
    }
    setTimeout(sendLoop, 500);
  }

  sendLoop();

  try {
    console.log("both turn-on");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "on", led: "led1" })
      }),
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "on", led: "led2" })
      })
    ]);
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log("led2 turn-on, led1 turn-off");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "on", led: "led2" })
      }),
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "off", led: "led1" })
      })
    ]);
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log("led2 turn-off, led1 turn-on");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "off", led: "led2" })
      }),
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "on", led: "led1" })
      })
    ]);

    await new Promise(resolve => setTimeout(resolve, 2000));

  } catch (error) {
    console.error(error);
  } finally {
    sending = false; // หยุดส่งภาพ
    console.log("both led turn-off");
    console.log("finish");

    if (overlayCanvas) {
      overlayCanvas.style.display = 'none';
    }

    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "off", led: "led1" })
      }),
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd: "off", led: "led2" })
      })
    ]);

    window.isSending = false; // ปลดล็อค
  }
};


  // Function to turn on the LED

  window.turnOnLed1 = function() {
    console.log("led1 turn-on");
    fetch(CLOUD_FUNCTION_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ cmd: "on" , led: "led1" }) // ส่งค่า led1 ไปด้วย
    })
    .then(res => res.text())
    .then(console.log)
    .catch(console.error);
  };

  // Function to turn off the LED
  window.turnOffLed1 = function() {
    console.log("led1 turn-off");
    fetch(CLOUD_FUNCTION_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ cmd: "off" , led: "led1"})
    })
    .then(res => res.text())
    .then(console.log)
    .catch(console.error);
  };

  window.turnOnLed2 = function() {
      console.log("led2 turn-on");
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ cmd: "on" , led: "led2"})
      })
      .then(res => res.text())
      .then(console.log)
      .catch(console.error);
    };
    
    // Function to turn off the LED
    window.turnOffLed2 = function() {
      console.log("led 2turn-off");
      fetch(CLOUD_FUNCTION_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ cmd: "off" , led: "led2"})
      })
      .then(res => res.text())
      .then(console.log)
      .catch(console.error);
    };

// Global variables for overlay
let overlayCtx;
let overlayCanvas; // เก็บอ้างอิงไว้ใช้ในหลายฟังก์ชัน
let lastLeftIris = null;
let lastRightIris = null;

// Initialize overlay when starting camera
async function startCamera() {
  try {
    const fullfaceVDO = document.getElementById('combinedVideoElement');
    overlayCanvas = document.getElementById('overlayCanvas');
    
    fullfaceVDO.style.transform = "scaleX(-1)";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    fullfaceVDO.srcObject = stream;
    
    // Set overlay canvas dimensions to match video
    fullfaceVDO.onloadedmetadata = () => {
      overlayCanvas.width = fullfaceVDO.videoWidth;
      overlayCanvas.height = fullfaceVDO.videoHeight;
      overlayCtx = overlayCanvas.getContext('2d');
      overlayCanvas.style.display = 'none'; // ซ่อน canvas ตอนเริ่มต้น
    };
    
    fullfaceVDO.play();
  } catch (error) {
    console.error("ไม่สามารถเปิดกล้องได้:", error);
  }
}

window.onload = startCamera;

// Function to draw the overlay
function drawOverlay(leftIris, rightIris) {
  if (!overlayCtx || (!leftIris && !rightIris)) return;

  overlayCanvas.style.display = 'block';
  overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);

  overlayCtx.save();
  overlayCtx.translate(overlayCtx.canvas.width, 0);
  overlayCtx.scale(-1, 1); // Flip horizontal

//   

  overlayCtx.restore();
}

// Function to send image to FastAPI and trigger canvas overlay
async function sendImageToFastAPI() {
  
  try {
    const video = document.getElementById('combinedVideoElement');
    if (!video) throw new Error("ไม่พบ video element");

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
    const formData = new FormData();
    formData.append('file', blob, 'snapshot.jpg');

    const response = await fetch('http://localhost:3000/detect', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const data = await response.json();

    console.log('Prediction result left:', data.left_iris);
    console.log('Prediction result right:', data.right_iris);

    // Store the last detected iris positions
    if (data.left_iris) lastLeftIris = data.left_iris;
    if (data.right_iris) lastRightIris = data.right_iris;

    // Draw the overlay only when model returns values
    if (lastLeftIris || lastRightIris) {
      drawOverlay(lastLeftIris, lastRightIris);
    }

  } catch (error) {
    console.error('Error sending image:', error);
    // ซ่อน canvas หากมีข้อผิดพลาด
    if (overlayCanvas) overlayCanvas.style.display = 'none';
  }
}
