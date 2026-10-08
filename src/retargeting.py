{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 """\
Phone-to-Panda: Human-to-robot trajectory retargeting\
\
Converts the human wrist trajectory extracted from phone video\
into a compact Cartesian trajectory for the Franka Panda.\
"""\
\
import numpy as np\
import pandas as pd\
\
\
def detect_main_grasp_region(\
    trajectory,\
    threshold=None,\
    min_samples=4\
):\
    """\
    Identify sustained low thumb-index distance regions.\
\
    Returns\
    -------\
    list of tuple\
        Start and end indices of sustained closed-hand regions.\
    """\
\
    pinch = trajectory["pinch_distance"].to_numpy()\
\
    if threshold is None:\
        threshold = np.percentile(pinch, 35)\
\
    closed = pinch < threshold\
\
    regions = []\
    start = None\
\
    for i, is_closed in enumerate(closed):\
\
        if is_closed and start is None:\
            start = i\
\
        elif not is_closed and start is not None:\
\
            if i - start >= min_samples:\
                regions.append((start, i - 1))\
\
            start = None\
\
    if start is not None:\
        if len(closed) - start >= min_samples:\
            regions.append((start, len(closed) - 1))\
\
    return regions\
\
\
def retarget_trajectory(\
    trajectory,\
    start_time,\
    end_time,\
    robot_start,\
    robot_end,\
    smoothing_window=5\
):\
    """\
    Retarget the human wrist trajectory between grasp and release\
    into a Cartesian Panda trajectory.\
\
    Human Y displacement is used as the primary progress signal.\
    """\
\
    segment = trajectory[\
        (trajectory["time"] >= start_time) &\
        (trajectory["time"] <= end_time)\
    ].copy()\
\
    if len(segment) == 0:\
        raise ValueError("No trajectory samples found in the selected interval.")\
\
    human_progress = (\
        segment["wrist_y"].rolling(\
            smoothing_window,\
            center=True,\
            min_periods=1\
        ).mean()\
    )\
\
    h_min = human_progress.iloc[0]\
    h_max = human_progress.iloc[-1]\
\
    if abs(h_max - h_min) < 1e-9:\
        progress = np.linspace(\
            0.0,\
            1.0,\
            len(segment)\
        )\
    else:\
        progress = (\
            (human_progress.to_numpy() - h_min) /\
            (h_max - h_min)\
        )\
\
    progress = np.clip(progress, 0.0, 1.0)\
\
    robot_start = np.asarray(robot_start, dtype=float)\
    robot_end = np.asarray(robot_end, dtype=float)\
\
    robot_positions = (\
        robot_start[None, :] +\
        progress[:, None] *\
        (robot_end - robot_start)[None, :]\
    )\
\
    # Force exact endpoints.\
    robot_positions[0] = robot_start\
    robot_positions[-1] = robot_end\
\
    retargeted = pd.DataFrame(\
        robot_positions,\
        columns=["robot_x", "robot_y", "robot_z"]\
    )\
\
    retargeted.insert(\
        0,\
        "time",\
        segment["time"].to_numpy()\
    )\
\
    return retargeted}