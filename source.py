import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import math

# ========================= CUSTOM CSS =========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Main background */
        .main {
            background: #0e1117;
            color: #ffffff;
        }

        /* Header gradient */
        .gradient-header {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(90deg, #00eaff, #0066ff);
            -webkit-background-clip: text;
            color: transparent;
            margin-bottom: -15px;
            margin-top: -15px;
        }

        /* Card container */
        .card {
            background: #161b22;
            padding: 18px 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
            margin-bottom: 20px;
        }

        /* Video frame */
        .stImage > img {
            border-radius: 12px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
        }

        /* Badge */
        .badge {
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 700;
            display: inline-block;
        }

        .badge-success {
            background: #00c853;
            color: white;
        }
        .badge-danger {
            background: #d50000;
            color: white;
        }

        /* Side title */
        .sidebar-title {
            font-size: 20px;
            font-weight: 700;
            margin-top: 15px;
        }

        .css-1d391kg {padding-top: 1rem;}
    </style>
""", unsafe_allow_html=True)

# ========================= INIT MEDIA PIPE =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ========================= FUNCTIONS =========================
def landmarks_distance(a, b, img_shape):
    h, w, _ = img_shape
    xa, ya = int(a.x * w), int(a.y * h)
    xb, yb = int(b.x * w), int(b.y * h)
    return np.sqrt((xa - xb)**2 + (ya - yb)**2)

def number_identify(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    count = 0
    if landmarks_distance(lm[4], lm[17], img_shape) > landmarks_distance(lm[2], lm[17], img_shape):
        count += 1
    if landmarks_distance(lm[8], lm[0], img_shape) > landmarks_distance(lm[5], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[12], lm[0], img_shape) > landmarks_distance(lm[9], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[16], lm[0], img_shape) > landmarks_distance(lm[13], lm[0], img_shape):
        count += 1
    if landmarks_distance(lm[20], lm[0], img_shape) > landmarks_distance(lm[17], lm[0], img_shape):
        count += 1
    return count

def landmarks_angular(lm1, lm2, lm3):
    v1 = (lm1.x - lm2.x, lm1.y - lm2.y)
    v2 = (lm3.x - lm2.x, lm3.y - lm2.y)
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    mag_v1 = (v1[0]**2 + v1[1]**2)**0.5
    mag_v2 = (v2[0]**2 + v2[1]**2)**0.5
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
    cos_angle = max(min(dot_product / (mag_v1 * mag_v2), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def extract_gesture_features(hand_landmarks, img_shape):
    lm = hand_landmarks.landmark
    return {
        'finger_count': number_identify(hand_landmarks, img_shape),
        'thumb_angle': landmarks_angular(lm[0], lm[2], lm[4]),
        'index_angle': landmarks_angular(lm[5], lm[6], lm[8]),
        'middle_angle': landmarks_angular(lm[9], lm[10], lm[12]),
        'ring_angle': landmarks_angular(lm[13], lm[14], lm[16]),
        'pinky_angle': landmarks_angular(lm[17], lm[18], lm[20]),
        'thumb_index_angle': landmarks_angular(lm[4], lm[0], lm[8]),
        'index_middle_angle': landmarks_angular(lm[8], lm[0], lm[12]),
        'middle_ring_angle': landmarks_angular(lm[12], lm[0], lm[16]),
        'ring_pinky_angle': landmarks_angular(lm[16], lm[0], lm[20])
    }

def compare_gestures(current, saved, tolerance=0.5):
    for key in saved:
        if key == 'finger_count':
            if current[key] != saved[key]:
                return False
        else:
            low = saved[key] * (1 - tolerance)
            high = saved[key] * (1 + tolerance)
            if not (low <= current[key] <= high):
                return False
    return True

# ========================= HEADER =========================
st.markdown("<h1 class='gradient-header'>Hand Gesture Unlock</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9aa0a6; font-size:17px'>Nhận diện cử chỉ - Bảo mật hiện đại</p>", unsafe_allow_html=True)

# ========================= SIDEBAR =========================
st.sidebar.markdown("<div class='sidebar-title'>⚙ Điều khiển</div>", unsafe_allow_html=True)

camera_on = st.sidebar.checkbox("Bật Camera")
set_new_password = st.sidebar.checkbox("Đặt mật khẩu cử chỉ")
unlock_by_gesture = st.sidebar.checkbox("Mở khóa bằng cử chỉ")

if set_new_password and st.sidebar.button("🔐 Ghi lại mật khẩu"):
    st.session_state.record_gesture = True
    st.sidebar.success("Sẵn sàng ghi lại!")

st.sidebar.markdown("<div class='sidebar-title'>👁 Hiển thị</div>", unsafe_allow_html=True)
counting_number = st.sidebar.checkbox("Hiển thị số ngón")
display_landmarks = st.sidebar.checkbox("Hiển thị landmark")
show_features = st.sidebar.checkbox("Hiển thị đặc trưng")

# ========================= SESSION STATE =========================
if 'gesture_password' not in st.session_state:
    st.session_state.gesture_password = None
if 'record_gesture' not in st.session_state:
    st.session_state.record_gesture = False
if 'last_unlock_time' not in st.session_state:
    st.session_state.last_unlock_time = 0
if 'last_wrong_time' not in st.session_state:
    st.session_state.last_wrong_time = 0

# ========================= CAMERA DISPLAY =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
FRAME_WINDOW = st.image([])
st.markdown("</div>", unsafe_allow_html=True)

if camera_on:

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập camera")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        left_fingers = 0
        right_fingers = 0
        gesture_info = None

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                label = handedness.classification[0].label
                fingers = number_identify(hand_landmarks, frame.shape)

                if label == "Left": 
                    left_fingers = fingers
                else:
                    right_fingers = fingers

                if counting_number:
                    posY = 30 if label == "Left" else 70
                    cv2.putText(frame, f"{label}: {fingers}", (10, posY),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 0), 2)

                if display_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture_info = extract_gesture_features(hand_landmarks, frame.shape)

                if st.session_state.record_gesture:
                    st.session_state.gesture_password = gesture_info
                    st.session_state.record_gesture = False
                    st.success("Đã lưu mật khẩu!")

                now = time.time()
                if unlock_by_gesture and st.session_state.gesture_password and (now - st.session_state.last_unlock_time > 3):
                    if compare_gestures(gesture_info, st.session_state.gesture_password):
                        st.session_state.last_unlock_time = now
                    else:
                        st.session_state.last_wrong_time = now

                if show_features and gesture_info:
                    y = 10
                    for key, val in gesture_info.items():
                        txt = f"{key}: {val:.2f}" if isinstance(val, float) else f"{key}: {val}"
                        cv2.putText(frame, txt, (450, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                        y += 15

        # Total fingers
        if counting_number:
            cv2.putText(frame, f"Total: {left_fingers+right_fingers}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # UNLOCK badge
        if time.time() - st.session_state.last_unlock_time <= 2:
            cv2.putText(frame, "UNLOCK", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # WRONG badge
        if time.time() - st.session_state.last_wrong_time <= 2:
            cv2.putText(frame, "WRONG PASSWORD", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.info("Camera đang tắt. Bật camera để bắt đầu.")


# ".conda/python.exe" -m streamlit run source.py
