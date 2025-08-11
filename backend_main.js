// Import Firebase SDK
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/9.22.2/firebase-analytics.js";

// Firebase config
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

// Global variables
let overlayCtx;
let overlayCanvas;
let lastLeftIris = null;
let lastRightIris = null;
let leftIrisPositions = [];
let rightIrisPositions = [];

// Function to calculate variance
function calculateVariance(positions) {
  if (positions.length === 0) return 0;

  const xs = positions.map(p => p.x);
  const ys = positions.map(p => p.y);

  const meanX = xs.reduce((sum, x) => sum + x, 0) / xs.length;
  const meanY = ys.reduce((sum, y) => sum + y, 0) / ys.length;

  const varX = xs.reduce((sum, x) => sum + Math.pow(x - meanX, 2), 0) / xs.length;
  const varY = ys.reduce((sum, y) => sum + Math.pow(y - meanY, 2), 0) / ys.length;

  return varX + varY;
}

// Start camera and setup overlay canvas
async function startCamera() {
  try {
    const video = document.getElementById('combinedVideoElement');
    overlayCanvas = document.getElementById('overlayCanvas');

    video.style.transform = "scaleX(-1)";
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
      overlayCtx = overlayCanvas.getContext('2d');
      overlayCanvas.style.display = 'none';
    };

    video.play();
  } catch (error) {
    console.error("ไม่สามารถเปิดกล้องได้:", error);
  }
}

window.onload = startCamera;

// Draw overlay on canvas (currently empty, customize as needed)
function drawOverlay(leftIris, rightIris) {
  if (!overlayCtx || (!leftIris && !rightIris)) return;

  overlayCanvas.style.display = 'block';
  overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);

  overlayCtx.save();
  overlayCtx.translate(overlayCtx.canvas.width, 0);
  overlayCtx.scale(-1, 1); // Flip horizontally

  // TODO: วาดวงกลมหรือจุดตำแหน่งตาดำได้ที่นี่

  overlayCtx.restore();
}

// Send image to FastAPI for detection
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

    // เก็บตำแหน่งตาดำลง array เพื่อวิเคราะห์ทีหลัง
    if (data.left_iris) {
      lastLeftIris = data.left_iris;
      leftIrisPositions.push(data.left_iris);
    }
    if (data.right_iris) {
      lastRightIris = data.right_iris;
      rightIrisPositions.push(data.right_iris);
    }

    if (lastLeftIris || lastRightIris) {
      drawOverlay(lastLeftIris, lastRightIris);
    }
  } catch (error) {
    console.error('Error sending image:', error);
    if (overlayCanvas) overlayCanvas.style.display = 'none';
  }
}

// Main function to start detection and control LED
window.turnStart = async function() {
  toggleSidebar(true);
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
    } catch (e) {
      console.error(e);
    }
    setTimeout(sendLoop, 50);
  }

  sendLoop();

  try {
    console.log("both turn-on");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "on", led: "led1" }) }),
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "on", led: "led2" }) })
    ]);
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log("led2 turn-on, led1 turn-off");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "on", led: "led2" }) }),
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "off", led: "led1" }) })
    ]);
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log("led2 turn-off, led1 turn-on");
    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "off", led: "led2" }) }),
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "on", led: "led1" }) })
    ]);
    await new Promise(resolve => setTimeout(resolve, 2000));

  } catch (error) {
    console.error(error);
  } finally {
    sending = false;

    console.log("both led turn-off");
    console.log("finish");

    if (overlayCanvas) {
      overlayCanvas.style.display = 'none';
    }

    await Promise.all([
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "off", led: "led1" }) }),
      fetch(CLOUD_FUNCTION_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ cmd: "off", led: "led2" }) })
    ]);

    // คำนวณความแปรปรวนของตำแหน่งตาดำ
const leftVar = calculateVariance(leftIrisPositions);
const rightVar = calculateVariance(rightIrisPositions);

const maxVar = Math.max(leftVar, rightVar);
const diffVar = Math.abs(leftVar - rightVar);
const diffPercent = (diffVar / maxVar) * 100;

console.log("ค่าความต่าง (absolute):", diffVar);
console.log("เปอร์เซ็นต์ความต่างของความแปรปรวน:", diffPercent.toFixed(2) + "%");

if (diffPercent > 45) {
  console.log("⚠️ อาจเป็นตาเข (แปรปรวนต่างกันเกิน 45%)");
  showPopup("⚠️ การคัดกรองเบื้องต้นเสร็จสิ้น", "ควรพบจักษุแพทย์เพื่อรับการตรวจอย่างละเอียด");
} else {
  console.log("✅ ตาปกติ (แปรปรวนใกล้เคียงกัน)")
  showPopup("✅ การคัดกรองเบื้องต้นเสร็จสิ้น", "ไม่พบอาการผิดปกติ");
}



    // ล้างข้อมูลตำแหน่งตาดำหลังวิเคราะห์เสร็จ
    leftIrisPositions = [];
    rightIrisPositions = [];

    window.isSending = false;
  }
};

// Functions to control LED manually
window.turnOnLed1 = function() {
  fetch(CLOUD_FUNCTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: "on", led: "led1" })
  }).then(res => res.text()).then(console.log).catch(console.error);
};

window.turnOffLed1 = function() {
  fetch(CLOUD_FUNCTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: "off", led: "led1" })
  }).then(res => res.text()).then(console.log).catch(console.error);
};

window.turnOnLed2 = function() {
  fetch(CLOUD_FUNCTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: "on", led: "led2" })
  }).then(res => res.text()).then(console.log).catch(console.error);
};

window.turnOffLed2 = function() {
  fetch(CLOUD_FUNCTION_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd: "off", led: "led2" })
  }).then(res => res.text()).then(console.log).catch(console.error);
};