import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# โหลดโมเดลและ detector แค่ครั้งเดียว
modelIrisDetection = tf.keras.models.load_model('model/left_model_adam_huber_130.h5')
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
)

def iris_bounding_box(points, expand_ratio=0.3):
    """
    หาขอบเขต bounding box ของ iris แล้วขยายขนาดตาม expand_ratio
    """
    arr = np.array(points)
    x_min, y_min = arr.min(axis=0)
    x_max, y_max = arr.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)

    # ขยายขนาดกล่องเพื่อให้มีพื้นที่รอบ ๆ iris
    size = int(max(w, h) * (1 + expand_ratio))
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = cx + size // 2
    y2 = cy + size // 2

    return x1, y1, x2, y2, cx, cy, size // 2

def detect_and_predict_iris(image: np.ndarray):
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if not results.face_landmarks:
        return {"status": "no_face"}

    landmarks = results.face_landmarks[0]
    coords = np.array([[int(p.x * width), int(p.y * height)] for p in landmarks])

    left_iris = coords[LEFT_IRIS]
    right_iris = coords[RIGHT_IRIS]

    # ฟังก์ชัน crop iris และ resize ให้ตรงกับ input โมเดล (สมมติโมเดลรับ 28x28)
    def crop_and_resize_iris(iris_points):
        x1, y1, x2, y2, cx, cy, r = iris_bounding_box(iris_points)
        # crop ภาพ iris
        crop_img = image[y1:y2, x1:x2]
        if crop_img.size == 0:
            return None, cx, cy, r
        # resize ภาพ iris ให้โมเดลรับได้
        resized = cv2.resize(crop_img, (219, 219))
        # normalize รูป (ขึ้นอยู่กับ preprocessing โมเดลคุณ)
        normalized = resized.astype(np.float32) / 255.0
        # เพิ่ม batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        return input_tensor, cx, cy, r

    # crop ซ้าย
    left_input, left_cx, left_cy, left_r = crop_and_resize_iris(left_iris)
    # crop ขวา
    right_input, right_cx, right_cy, right_r = crop_and_resize_iris(right_iris)

    # predict ด้วยโมเดล
    left_pred = modelIrisDetection.predict(left_input) if left_input is not None else None
    right_pred = modelIrisDetection.predict(right_input) if right_input is not None else None

    return {
        "status": "ok",
        "left_iris": {
            "x": left_cx,
            "y": left_cy,
            "radius": left_r,
            "prediction": left_pred.tolist() if left_pred is not None else None
        },
        "right_iris": {
            "x": right_cx,
            "y": right_cy,
            "radius": right_r,
            "prediction": right_pred.tolist() if right_pred is not None else None
        }
    }
