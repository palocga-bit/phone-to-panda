{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
Phone-to-Panda: Hand tracking\
\
Extracts MediaPipe hand landmarks from a personally recorded phone video\
and derives a compact wrist/pinch trajectory representation.\
"""\
\
import cv2\
import numpy as np\
import pandas as pd\
import mediapipe as mp\
\
\
def extract_hand_landmarks(video_path, model_path):\
    """\
    Extract 21 MediaPipe hand landmarks from each detected video frame.\
\
    Returns\
    -------\
    pandas.DataFrame\
        One row per detected frame containing frame index, timestamp,\
        and 21 (x, y, z) landmark coordinates.\
    """\
\
    BaseOptions = mp.tasks.BaseOptions\
    HandLandmarker = mp.tasks.vision.HandLandmarker\
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions\
    VisionRunningMode = mp.tasks.vision.RunningMode\
\
    options = HandLandmarkerOptions(\
        base_options=BaseOptions(model_asset_path=model_path),\
        running_mode=VisionRunningMode.VIDEO,\
        num_hands=1\
    )\
\
    cap = cv2.VideoCapture(video_path)\
\
    fps = cap.get(cv2.CAP_PROP_FPS)\
    rows = []\
    frame_index = 0\
\
    with HandLandmarker.create_from_options(options) as landmarker:\
\
        while True:\
            ok, frame = cap.read()\
\
            if not ok:\
                break\
\
            timestamp_ms = int(frame_index / fps * 1000)\
\
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\
\
            mp_image = mp.Image(\
                image_format=mp.ImageFormat.SRGB,\
                data=rgb\
            )\
\
            result = landmarker.detect_for_video(\
                mp_image,\
                timestamp_ms\
            )\
\
            if result.hand_landmarks:\
\
                hand = result.hand_landmarks[0]\
\
                row = \{\
                    "frame": frame_index,\
                    "time": frame_index / fps\
                \}\
\
                for i, landmark in enumerate(hand):\
                    row[f"x\{i\}"] = landmark.x\
                    row[f"y\{i\}"] = landmark.y\
                    row[f"z\{i\}"] = landmark.z\
\
                rows.append(row)\
\
            frame_index += 1\
\
    cap.release()\
\
    return pd.DataFrame(rows)\
\
\
def build_hand_trajectory(landmarks):\
    """\
    Convert landmark detections into wrist and pinch-distance features.\
\
    MediaPipe landmark indices:\
        4 = thumb tip\
        8 = index fingertip\
    """\
\
    trajectory = pd.DataFrame()\
\
    trajectory["frame"] = landmarks["frame"]\
    trajectory["time"] = landmarks["time"]\
\
    trajectory["wrist_x"] = landmarks["x0"]\
    trajectory["wrist_y"] = landmarks["y0"]\
\
    thumb = landmarks[["x4", "y4", "z4"]].to_numpy()\
    index = landmarks[["x8", "y8", "z8"]].to_numpy()\
\
    trajectory["pinch_distance"] = np.linalg.norm(\
        thumb - index,\
        axis=1\
    )\
\
    return trajectory}