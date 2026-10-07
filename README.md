# Phone-to-Panda: Data-Efficient Human-to-Robot Retargeting from Self-Collected Hand Motion

## Overview

This project explores whether a small amount of personally collected human manipulation data can be used to control a different robotic embodiment through a lightweight retargeting pipeline.

I recorded manipulation demonstrations using a smartphone, extracted hand landmarks from the video, converted the observed human hand motion into a compact trajectory representation, and retargeted that trajectory to a Franka Panda robot simulated in MuJoCo.

The final demonstration uses a human hand trajectory to control a Panda robot performing a simple pick-and-place task: grasping a cube, carrying it to a target box, and releasing it inside the box.

The project deliberately prioritises data use, implementation simplicity and reproducibility rather than adding a large model solely for complexity.

## Pipeline

```text
Self-recorded phone video
        ↓
MediaPipe hand landmark detection
        ↓
Wrist trajectory + pinch/open state
        ↓
Grasp and release event detection
        ↓
Trajectory normalisation and retargeting
        ↓
Franka Panda in MuJoCo
        ↓
Cube pick → carry → release
```

## Personal Data

The challenge requires the collected data to play a role in the approach.

I personally recorded 12 manipulation videos using a smartphone. The videos show my hand interacting with a cube and a wooden shape-sorter box.

The final demonstration uses one of these recordings:

`IMG_2343.MOV`

The source video contains:

* 267 frames
* 30 FPS
* approximately 8.9 seconds
* 1080 × 1920 resolution
* side/profile view of the manipulation scene

The videos were recorded personally rather than using an existing robotics dataset.

The original personal videos are not included in the public repository. The repository instead contains derived landmark and trajectory data required to demonstrate the approach.

## Method

### 1. Hand Tracking

MediaPipe Hand Landmarker was used to detect 21 hand landmarks from the recorded video.

For each detected frame, the system stores the landmark coordinates and derives:

* wrist position
* thumb position
* index-finger position
* thumb-index distance

The thumb-index distance provides a simple proxy for the hand being open or closed.

For the final video:

* detected hand frames: 155
* detection rate: 58.05%

The relatively modest detection rate is retained rather than artificially interpolating the entire recording. The manipulation trajectory is constructed from the successfully detected frames.

### 2. Grasp and Release Detection

The thumb-index distance is used to identify sustained closing behaviour.

The main sustained closed-hand region was interpreted as the manipulation/grasp interval.

For the final trajectory:

* grasp time: approximately 4.47 s
* release time: approximately 6.47 s

This converts the continuous phone recording into a compact manipulation trajectory.

### 3. Human-to-Robot Retargeting

The human wrist trajectory is normalised between the detected grasp and release positions.

Rather than attempting to reproduce the exact human workspace, the normalised motion is mapped into a suitable workspace for the Panda.

This allows the human trajectory to control a robot with a completely different embodiment.

The final manipulation trajectory contains:

* 25 retargeted trajectory samples

The robot therefore does not reproduce the human hand or finger configuration directly. Instead, the important information is reduced to:

* motion direction
* relative displacement
* grasp timing
* release timing

This is the central retargeting idea of the project.

## Simulation

The robot simulation uses the Franka Panda in MuJoCo.

The environment contains:

* Franka Panda manipulator
* table
* red/orange cube
* green target box

The task is:

1. Move to the cube.
2. Close the gripper.
3. Carry the cube according to the retargeted phone trajectory.
4. Move above the target box.
5. Open the gripper.
6. Perform a controlled descent into the target box.

## Simplified Grasp Model

The project intentionally uses a simplified deterministic grasp/release mechanism rather than attempting to model detailed contact-rich grasping physics.

After the robot reaches the grasp position, the cube is associated with the robot hand using the measured grasp offset. During the carrying phase, the cube follows the retargeted hand trajectory.

At release, the gripper is opened and the cube is placed through a controlled descent into the target box.

This design isolates the main research question — whether personally collected human motion can be retargeted to another robotic embodiment — without making the project dependent on a complicated contact-dynamics setup.

The resulting simulation should therefore be interpreted as a **retargeting demonstration**, rather than as a physically realistic evaluation of robotic grasping.

## Results

| Metric                           |   Result |
| -------------------------------- | -------: |
| Personal videos collected        |       12 |
| Source video frames              |      267 |
| Source video FPS                 |       30 |
| Source video duration            |    8.9 s |
| Hand detection rate              |   58.05% |
| Detected hand frames             |      155 |
| Retargeted trajectory samples    |       25 |
| Simulation states                |      145 |
| Grasp target error               | 0.198 mm |
| Final horizontal placement error | 0.000 mm |
| Final vertical placement error   | 0.000 mm |
| Task success                     |     True |

The final simulation successfully placed the cube inside the target box.

## Final Demonstration

The repository includes the rendered MuJoCo demonstration:

`results/phone_to_panda_final.mp4`

The video shows the final Panda manipulation generated from the retargeted trajectory.

## Why This Approach?

A larger VLA, reinforcement-learning system or world model could potentially be added, but that would introduce substantial additional dependencies and complexity.

I deliberately chose a lightweight, hand-coded approach because the central requirement of the challenge is that personally collected data must play a role in the solution.

The project was developed in Google Colab, where keeping the pipeline compact also improved reproducibility and runtime stability.

Rather than adding a large model purely for complexity, the implementation focuses on:

* personally collected data
* extracting useful structure from that data
* transferring human motion to a different embodiment
* measurable simulation performance
* simple and reproducible implementation

## What Worked

* Smartphone video provided sufficient information to extract a useful hand trajectory.
* MediaPipe provided a practical way to obtain hand landmarks without training a vision model from scratch.
* Thumb-index distance provided a simple grasp-state signal.
* Normalised trajectory retargeting successfully transferred the manipulation motion to the Panda workspace.
* The final simulation successfully completed the pick-and-place task.
* The approach remained lightweight enough to run in Google Colab.

## What Did Not Work / Limitations

The project has several deliberate limitations.

### Hand Detection

The final video achieved a 58.05% hand detection rate. The side-view recording and occasional landmark failures mean that the full video cannot be treated as a perfectly observed trajectory.

### Simplified Grasping

The grasp is not a physically realistic contact-rich grasp. The cube is controlled using a deterministic grasp/release mechanism.

### Limited Embodiment Information

Only a small set of human motion variables is transferred. Finger-level human motion is not mapped directly to the Panda's individual joints.

### No Learned Policy

The final system does not train a large VLA, world model or reinforcement-learning policy. The contribution is instead a lightweight data-to-robot retargeting pipeline.

These limitations are intentional trade-offs that keep the implementation simple and make the role of the personally collected data explicit.

## Repository Structure

```text
phone-to-panda/
│
├── README.md
│
├── data/
│   ├── landmarks_IMG_2343.csv
│   └── hand_trajectory_IMG_2343.csv
│
├── results/
│   ├── phone_to_panda_final.mp4
│   ├── metrics.csv
│   ├── retargeted_phone_trajectory.csv
│   └── final_simulation_stages.csv
│
├── src/
│   ├── tracking.py
│   ├── retargeting.py
│   ├── environment.py
│   └── panda_phone_task.xml
│
├── notebooks/
│   └── phone_to_panda.ipynb
│
└── presentation/
    └── phone_to_panda_presentation.pptx
```

## Reproducibility

The project was developed and tested in Google Colab.

The main dependencies are:

* Python
* MediaPipe
* NumPy
* Pandas
* MuJoCo
* OpenCV
* ImageIO
* python-pptx

The notebook contains the processing and simulation workflow used to produce the final result.

### MuJoCo Panda Model

The custom task XML:

`src/panda_phone_task.xml`

includes the standard Panda model using:

```xml
<include file="panda.xml"/>
```

The base `panda.xml` is provided by the official DeepMind MuJoCo Menagerie.

To reproduce the simulation:

1. Install the Python dependencies.
2. Clone the MuJoCo Menagerie repository.
3. Locate the `franka_emika_panda` directory containing `panda.xml`.
4. Place `panda_phone_task.xml` alongside `panda.xml`.
5. Run the simulation workflow in `notebooks/phone_to_panda.ipynb`.

The custom XML defines the task-specific table, cube, target box and cameras while using the standard Panda model from the Menagerie.

## Conclusion

This project demonstrates a simple way to turn personally collected human manipulation data into robot behaviour.

The main idea is intentionally compact:

```text
record → detect → extract → retarget → simulate
```

Despite using only a small amount of self-collected data and a lightweight implementation, the resulting trajectory successfully drives a Panda robot through the target manipulation task.

The experiment suggests that even relatively simple human-motion representations can provide useful information for cross-embodiment robot control when the task and retargeting space are appropriately constrained.

