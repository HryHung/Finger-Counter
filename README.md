# Hand-gesture-app

This project is a computer vision application that utilizes MediaPipe and Streamlit to detect hand gestures, count fingers, and implement a gesture-based unlocking system.

## Installation and Setup

<img width="581" height="294" alt="image" src="https://github.com/user-attachments/assets/1faf7097-98e5-4128-9c4d-ecbb737119b4" />

To ensure compatibility and avoid library conflicts, it is recommended to use a virtual environment via Anaconda.

### Step 1: Create and Activate Virtual Environment
1. Open Anaconda Prompt from your Windows Start menu. This is the link in case you don't have anaconda: https://www.anaconda.com/download
2. Create a new environment with Python 3.10:
   ```bash
   conda create -n handges_env python=3.10
   ```
3. Activate the environment:
   ```bash
   conda activate handges_env
   ```

### Step 2: Install Required Libraries
Install the necessary dependencies into your active environment using pip:
```bash
pip install mediapipe streamlit opencv-python numpy
```

### Step 3: Configure VS Code (Optional)
If you are using Visual Studio Code:
1. Open the project folder in VS Code.
2. Press Ctrl + Shift + P (or F1).
3. Type "Python: Select Interpreter" and select the "handges_env" environment.

## How to Run

1. Open your terminal or Anaconda Prompt and navigate to the project directory:
   ```bash
   Example: cd /d F:\.Hung\Study\Year_4\computer_vision\BTL\Hand-ges
   ```
2. Launch the Streamlit application:
   ```bash
   streamlit run source.py
   ```
3. The app will open in your default browser at http://localhost:8501.

## How to Use the Web-app

### Control Part
- Enable Camera: Toggle your camera on or off.
- Set Gesture Password: Setup the gesture you want to use as your password. Click the Record button to save it.
- Unlock by Gesture: Perform the recorded gesture again. If correct, an "UNLOCK" notification will appear.

### Feature Part
- Show Finger Counting: Displays the number of fingers detected in real time.
- Show Hand Landmarks: Shows the hand skeleton defined by MediaPipe.
- Show Feature Values: Displays all raw values used to identify gestures.
