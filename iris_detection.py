#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:24:52 2024

@author: kwattana
"""

import tensorflow as tf
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input

TARGET_SIZE = (219, 219)

# Left and right iris
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Left and right eyes indices 
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

model_file_full = '/Users/taned/Desktop/project/iris_eye_dectect/left_model_adam_huber_130.h5'
print(model_file_full)
modelIrisDetection = tf.keras.models.load_model(model_file_full)

#------------------------------------------------------------------------------
def resize_and_pad_image(image, target_size=(219, 219)):
    """
    Resizes and pads the image to make it a square of `target_size`.
    """
    print(image.shape)
    height, width = image.shape[:2]
    
    # Calculate the resize ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_width = int(target_size[1] * aspect_ratio)
        new_height = target_size[1]
    
    # Resize image while maintaining aspect ratio
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create new square image and pad the resized image (centered)
    new_image = cv2.copyMakeBorder(image_resized, 
                                   top=(target_size[1] - new_height) // 2, 
                                   bottom=(target_size[1] - new_height + 1) // 2, 
                                   left=(target_size[0] - new_width) // 2, 
                                   right=(target_size[0] - new_width + 1) // 2, 
                                   borderType=cv2.BORDER_CONSTANT, 
                                   value=(0, 0, 0))  # Black padding
    
    # Return resized and padded image, and resizing information
    return new_image, (width, height), (new_width, new_height), (target_size[1] - new_height) // 2

#------------------------------------------------------------------------------
CB_LENGTH = 50
CB_LINE_WIDTH = 1
def draw_cross_bar(image_in, pos_x, pos_y, color):
    cv2.line(image_in, (pos_x - CB_LENGTH, pos_y), (pos_x + CB_LENGTH, pos_y), color, CB_LINE_WIDTH)
    cv2.line(image_in, (pos_x, pos_y - CB_LENGTH), (pos_x, pos_y + CB_LENGTH), color, CB_LINE_WIDTH)

#------------------------------------------------------------------------------
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
    
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
            
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
    
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
    
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())
  
    return annotated_image
  
#------------------------------------------------------------------------------
def bounding_box(points):
    min_x = 10000
    min_y = 10000
    max_x = -10000
    max_y = -10000
    for point in points:
        if (point[0] < min_x):
            min_x = point[0]
        if (point[1] < min_y):
            min_y = point[1]
        if (point[0] > max_x):
            max_x = point[0]
        if (point[1] > max_y):
            max_y = point[1]
    
    return([min_x, min_y - int(0.25 * (max_y - min_y)), max_x, max_y])

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#******************************************************************************
# Change camera index here
#******************************************************************************
camera = cv2.VideoCapture(0)

while (True):
    ret, frame = camera.read() # getting frame from camera 
    if ret: 
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('temp.jpg', frame)
        image = mp.Image.create_from_file('temp.jpg')
        image_rgba = image
        image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGBA2RGB)
        img_height, img_width = frame.shape[:2]
        
        #----------------------------------------------------------------------
        # Detect face landmarks from the input image.
        detection_result = detector.detect(image_rgba)
        
        if (detection_result.face_landmarks != []):
            #----------------------------------------------------------------------
            # Convert normalized coordinates of face landmark to screen coordinates
            image_coord_face_landmark = []
            for face_landmark in detection_result.face_landmarks[0]:
                # Convert to pb2 and extract coordinate
                face_landmark_pb2 = face_landmark.to_pb2()
                image_coord_face_landmark.append([face_landmark_pb2.x * img_width, face_landmark_pb2.y * img_height])
        
            #----------------------------------------------------------------------
            # Extract only relevant data (left and right eyes and pupils)        
            left_eye = [np.array([image_coord_face_landmark[p] for p in LEFT_EYE], dtype=np.int32)][0]
            left_eye_bb = bounding_box(left_eye)
            left_iris = [np.array([image_coord_face_landmark[p] for p in LEFT_IRIS], dtype=np.int32)][0]
            left_iris_bb = bounding_box(left_iris)
        
            right_eye = [np.array([image_coord_face_landmark[p] for p in RIGHT_EYE], dtype=np.int32)][0]
            right_eye_bb = bounding_box(right_eye)
            right_iris = [np.array([image_coord_face_landmark[p] for p in RIGHT_IRIS], dtype=np.int32)][0]
            right_iris_bb = bounding_box(right_iris)
            
            #******************************************************************
            # Show keypoints
            #******************************************************************
            #annotated_image = draw_landmarks_on_image(image, detection_result)        
            annotated_image = image
        
#            cv2.rectangle(annotated_image, (left_eye_bb[0], left_eye_bb[1]), (left_eye_bb[2], left_eye_bb[3]), (255, 255, 255), 5)
#            cv2.rectangle(annotated_image, (right_eye_bb[0], right_eye_bb[1]), (right_eye_bb[2], right_eye_bb[3]), (255, 255, 255), 5)
        
            # cv2.rectangle(annotated_image, (left_iris_bb[0], left_iris_bb[1]), (left_iris_bb[2], left_iris_bb[3]), (255, 255, 255), 5)
            # cv2.rectangle(annotated_image, (right_iris_bb[0], right_iris_bb[1]), (right_iris_bb[2], right_iris_bb[3]), (255, 255, 255), 5)
        
            left_eye_image = image[left_eye_bb[1]:left_eye_bb[3], left_eye_bb[0]:left_eye_bb[2], :]
            left_eye_image_org = left_eye_image.copy()
            left_iris_hw = int((left_iris_bb[2] - left_iris_bb[0]) / 2)
            left_iris_hh = int((left_iris_bb[3] - left_iris_bb[1]) / 2)
            left_iris_offset_x = left_iris_bb[0] - left_eye_bb[0]
            left_iris_offset_y = left_iris_bb[1] - left_eye_bb[1]
            left_iris = (left_iris_offset_x + left_iris_hw, left_iris_offset_y + left_iris_hh)
        
            right_eye_image = image[right_eye_bb[1]:right_eye_bb[3], right_eye_bb[0]:right_eye_bb[2], :]
            right_eye_image_org = right_eye_image.copy()
            right_iris_hw = int((right_iris_bb[2] - right_iris_bb[0]) / 2)
            right_iris_hh = int((right_iris_bb[3] - right_iris_bb[1]) / 2)
            right_iris_offset_x = right_iris_bb[0] - right_eye_bb[0]
            right_iris_offset_y = right_iris_bb[1] - right_eye_bb[1]
            right_iris = (right_iris_offset_x + right_iris_hw, right_iris_offset_y + right_iris_hh)
        
            if ((left_eye_image.shape[0] > 0) and (right_eye_image.shape[0] > 0)):
                left_eye_bb_ratio = left_eye_image.shape[1] / left_eye_image.shape[0]
                right_eye_bb_ratio = right_eye_image.shape[1] / right_eye_image.shape[0]
                
                scale_percent = 100 * TARGET_SIZE[1] / left_eye_image.shape[1]
                width = int(left_eye_image.shape[1] * scale_percent / 100)
                height = int(left_eye_image.shape[0] * scale_percent / 100)
                dsize = (width, height)
                image_temp = cv2.resize(left_eye_image, dsize)
                left_eye_output_image = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), np.uint8)
                left_eye_output_image_mod = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), np.uint8)

                if (image_temp.shape[0] <= TARGET_SIZE[1]):
                    left_eye_output_image[0:image_temp.shape[0], 0:image_temp.shape[1], :] = image_temp
                    left_iris = (int(left_iris[0] * (scale_percent / 100)), int(left_iris[1] * (scale_percent / 100)))

                    start_r_left = int((TARGET_SIZE[0] - image_temp.shape[0]) / 2)
                    left_eye_output_image_mod[start_r_left:start_r_left+image_temp.shape[0], 0:image_temp.shape[1], :] = image_temp
        
                #--------------------------------------------------------------
                scale_percent = 100 * TARGET_SIZE[1] / right_eye_image.shape[1]
                width = int(right_eye_image.shape[1] * scale_percent / 100)
                height = int(right_eye_image.shape[0] * scale_percent / 100)
                dsize = (width, height)
                image_temp = cv2.resize(right_eye_image, dsize)
                
                right_eye_output_image_mod = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), np.uint8)
                right_eye_output_image = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1], 3), np.uint8)

                if (image_temp.shape[0] <= TARGET_SIZE[1]):
                    right_eye_output_image[0:image_temp.shape[0], 0:image_temp.shape[1], :] = image_temp
                    right_iris = (int(right_iris[0] * (scale_percent / 100)), int(right_iris[1] * (scale_percent / 100)))

                    start_r_right = int((TARGET_SIZE[0] - image_temp.shape[0]) / 2)
                    right_eye_output_image_mod[start_r_right:start_r_right+image_temp.shape[0], 0:image_temp.shape[1], :] = image_temp
                
                #--------------------------------------------------------------
                # Resize and pad the image
                left_eye_resized, original_size, new_size, padded_y = resize_and_pad_image(left_eye_image_org, TARGET_SIZE)
                left_ratio = original_size[0] / new_size[0]
                black_eye_left = modelIrisDetection.predict(np.array([preprocess_input(left_eye_resized)]), verbose=0)  
                cv2.circle(left_eye_resized, (int(black_eye_left[0][0]), int(black_eye_left[0][1])), int(black_eye_left[0][2]), (0, 255, 0), 5)
                black_eye_left_org_image = (int(black_eye_left[0][0] * left_ratio), int((black_eye_left[0][1] - padded_y) * left_ratio), int(black_eye_left[0][2] * left_ratio))
                cv2.circle(annotated_image, (left_eye_bb[0] + black_eye_left_org_image[0], left_eye_bb[1] + black_eye_left_org_image[1]), black_eye_left_org_image[2], (255, 0, 0), 2)        

                right_eye_resized, original_size, new_size, padded_y = resize_and_pad_image(right_eye_image_org, TARGET_SIZE)
                right_ratio = original_size[0] / new_size[0]
                black_eye_right = modelIrisDetection.predict(np.array([preprocess_input(right_eye_resized)]), verbose=0)  
                cv2.circle(right_eye_resized, (int(black_eye_right[0][0]), int(black_eye_right[0][1])), int(black_eye_right[0][2]), (0, 255, 0), 5)
                black_eye_right_org_image = (int(black_eye_right[0][0] * right_ratio), int((black_eye_right[0][1] - padded_y) * right_ratio), int(black_eye_right[0][2] * right_ratio))
                cv2.circle(annotated_image, (right_eye_bb[0] + black_eye_right_org_image[0], right_eye_bb[1] + black_eye_right_org_image[1]), black_eye_right_org_image[2], (255, 0, 0), 2)        

                draw_cross_bar(annotated_image, left_eye_bb[0] + black_eye_left_org_image[0], left_eye_bb[1] + black_eye_left_org_image[1], (255, 0, 0))
                draw_cross_bar(annotated_image, right_eye_bb[0] + black_eye_right_org_image[0], right_eye_bb[1] + black_eye_right_org_image[1], (255, 0, 0))

                combined_black_eyes = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1] * 2, 3), np.uint8)    
                combined_black_eyes[:, 0:TARGET_SIZE[1], :] = right_eye_resized
                combined_black_eyes[:, TARGET_SIZE[1]:, :] = left_eye_resized
                cv2.imshow('Black Eyes', combined_black_eyes)
        
    
#                 # Now, estimate the pupil position of the left eye (from the right eye image)
#                 # pupil_left = modelLeftEyeRightPupil.predict(np.array([preprocess_input(right_eye_output_image_mod)]), verbose=0)   
#                 # pupil_right = modelRightEyeLeftPupil.predict(np.array([preprocess_input(left_eye_output_image_mod)]), verbose=0)   
# #                    right_eye_output_image_mod = cv2.cvtColor(right_eye_output_image_mod, cv2.COLOR_RGB2BGR)
#                 pupil_right = modelLeftEyeRightPupil.predict(np.array([preprocess_input(left_eye_output_image_mod)]), verbose=0)   
# #                    left_eye_output_image_mod = cv2.cvtColor(left_eye_output_image_mod, cv2.COLOR_RGB2BGR)
#                 pupil_left = modelRightEyeLeftPupil.predict(np.array([preprocess_input(right_eye_output_image_mod)]), verbose=0)   

#                 #--------------------------------------------------------------
#                 # Show them 
#                 # Pupil locations from MediaPipe
#                 # draw_cross_bar(left_eye_output_image, left_iris[0], left_iris[1], (255, 255, 255))
#                 # draw_cross_bar(right_eye_output_image, right_iris[0], right_iris[1], (255, 255, 255))

#                 # Pupil locations from Xception regression (from eye images)
#                 draw_cross_bar(left_eye_output_image, int(pupil_left[1][0][0]), int(pupil_left[1][0][1] - start_r_left), (0, 0, 255))
#                 draw_cross_bar(right_eye_output_image, int(pupil_right[1][0][0]), int(pupil_right[1][0][1] - start_r_right), (0, 0, 255))

#                 pred = modelPupilEstimationRight.predict(np.array([preprocess_input(right_eye_output_image_mod)]), verbose=0)
#                 draw_cross_bar(right_eye_output_image, int(pred[1][0][0]), int(pred[1][0][1] - start_r_right), (0, 255, 0))
#                 pred = modelPupilEstimationLeft.predict(np.array([preprocess_input(left_eye_output_image_mod)]), verbose=0)
#                 draw_cross_bar(left_eye_output_image, int(pred[1][0][0]), int(pred[1][0][1] - start_r_left), (0, 255, 0))

        
                image_final = np.zeros((TARGET_SIZE[0], TARGET_SIZE[1] * 2, 3), np.uint8)    
                image_final[:, 0:TARGET_SIZE[1], :] = right_eye_output_image
                image_final[:, TARGET_SIZE[1]:, :] = left_eye_output_image
                dsize = (image_final.shape[1] * 2, image_final.shape[0] * 2)
                image_final = cv2.resize(image_final, dsize)
            
                cv2.imshow('Eye Images', image_final)
                cv2.imshow('Output', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                                
                if cv2.waitKey(1) == ord('q'): 
                    cv2.destroyAllWindows()
                    exit(0)
        
    else:
        cv2.destroyAllWindows()
        exit(0)
