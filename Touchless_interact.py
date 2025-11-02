import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import pyautogui
import time
import keyboard
import speech_recognition as sr
import pyaudio  # Required for microphone access
import subprocess
import threading
import os
from difflib import get_close_matches  # For fuzzy keyword matching

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

mp_drawing = mp.solutions.drawing_utils

# Default trackbar function
def setValues(x):
    pass

# Trackbars for color detection
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, setValues)
cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, setValues)

# Color points arrays
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
mpoints = [deque(maxlen=1024)]  # Maroon
wpoints = [deque(maxlen=1024)]  # White
grpoints = [deque(maxlen=1024)]  # Grey

blue_index = green_index = red_index = yellow_index = maroon_index = white_index = grey_index = 0

kernel = np.ones((5, 5), np.uint8)

# Updated colors list with new colors (BGR format)
colors = [
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 0, 255),  # Red
    (0, 255, 255),  # Yellow
    (0, 0, 128),  # Maroon
    (255, 255, 255),  # White
    (128, 128, 128)  # Grey
]

colorIndex = 0

# Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)  # Blue
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)  # Green
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)  # Red
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)  # Yellow
paintWindow = cv2.rectangle(paintWindow, (620, 1), (715, 65), colors[4], -1)  # Maroon
paintWindow = cv2.rectangle(paintWindow, (735, 1), (830, 65), colors[5], -1)  # White
paintWindow = cv2.rectangle(paintWindow, (850, 1), (945, 65), colors[6], -1)  # Grey

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "MAROON", (645, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "WHITE", (760, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREY", (875, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Virtual keyboard layout with space bar
keyboard_layout = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "     "  # Space bar (5 spaces wide)
]

key_size = 50
keyboard_width = len(keyboard_layout[0]) * key_size
keyboard_height = len(keyboard_layout) * key_size
keyboard_origin = ((1280 - keyboard_width) // 2, 720 - keyboard_height - 20)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screen_width, screen_height = pyautogui.size()

# State tracking
last_click_state = False
last_all_fingers_up_time = 0
save_cooldown = 0
screenshot_count = 0
drawing_count = 0
typing_cooldown = 0
last_key = None
keyboard_active = False

# Speech recognizer setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

# App mapping with keywords (customize as needed)
app_map = {
    "chrome": {"exe": "chrome.exe", "keywords": ["chrome", "chorme", "google chrome", "browser"]},
    "notepad": {"exe": "notepad.exe", "keywords": ["notepad", "note pad", "text editor"]},
    "calculator": {"exe": "calc.exe", "keywords": ["calculator", "calc", "math"]},
    "firefox": {"exe": "firefox.exe", "keywords": ["firefox", "fire fox", "fox browser"]},
    "paint": {"exe": "mspaint.exe", "keywords": ["paint", "ms paint", "drawing app"]},
    # Add more apps here
}

def voice_commands():
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)  # Longer noise adjustment for better accuracy
            print("Listening for voice command...")
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)  # Longer phrase limit for natural speech
            except sr.WaitTimeoutError:
                print("No speech detected, continuing to listen...")
                continue
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized: {command}")

            # Fuzzy match for app names
            all_keywords = [kw for app in app_map.values() for kw in app["keywords"]]
            matched = get_close_matches(command, all_keywords, n=1, cutoff=0.5)  # Lower cutoff for better fuzzy matching
            if matched:
                matched_kw = matched[0]
                for app_name, details in app_map.items():
                    if matched_kw in details["keywords"]:
                        if "open" in command or app_name in command:
                            subprocess.Popen(details["exe"])
                            print(f"Opened {app_name}")
                        elif "close" in command:
                            if os.name == 'nt':  # Windows
                                os.system(f"taskkill /f /im {details['exe']}")
                            else:  # macOS/Linux
                                os.system(f"pkill {app_name}")
                            print(f"Closed {app_name}")
                        break
            else:
                print("No matching app found, try again.")
        except sr.UnknownValueError:
            print("Could not understand audio, retrying...")
        except sr.RequestError as e:
            print(f"Speech service error: {e}, retrying...")
        time.sleep(0.5)  # Shorter sleep for more continuous listening

# Start voice listening in background thread
voice_thread = threading.Thread(target=voice_commands, daemon=True)
voice_thread.start()

def interpolate_points(pt1, pt2, steps=3):
    """Generate intermediate points between pt1 and pt2 for smoother lines."""
    points = []
    for i in range(1, steps):
        x = int(pt1[0] + (pt2[0] - pt1[0]) * i / steps)
        y = int(pt1[1] + (pt2[1] - pt1[1]) * i / steps)
        points.append((x, y))
    return points

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = frame.shape

    # Create an overlay for transparent keyboard
    overlay = frame.copy()

    # Trackbar values
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")

    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])

    # Color buttons on frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)  # Clear
    frame = cv2.rectangle(frame, (160, 1), (255, 65), colors[0], -1)  # Blue
    frame = cv2.rectangle(frame, (275, 1), (370, 65), colors[1], -1)  # Green
    frame = cv2.rectangle(frame, (390, 1), (485, 65), colors[2], -1)  # Red
    frame = cv2.rectangle(frame, (505, 1), (600, 65), colors[3], -1)  # Yellow
    frame = cv2.rectangle(frame, (620, 1), (715, 65), colors[4], -1)  # Maroon
    frame = cv2.rectangle(frame, (735, 1), (830, 65), colors[5], -1)  # White
    frame = cv2.rectangle(frame, (850, 1), (945, 65), colors[6], -1)  # Grey

    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "MAROON", (645, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "WHITE", (760, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREY", (875, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Mask for pointer (no display)
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    # Process hand landmarks
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    center = None
    mode = "None"
    left_hand_detected = False
    right_hand_index_tip = None
    right_hand_thumb_tip = None

    index_up = False  # Default value to prevent NameError

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_spec,
                connection_spec
            )

            # Get landmark positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Convert to pixel coordinates
            index_tip = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_tip = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            middle_tip = (int(middle_tip.x * w), int(middle_tip.y * h))
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            ring_tip = (int(ring_tip.x * w), int(ring_tip.y * h))
            pinky_tip = (int(pinky_tip.x * w), int(pinky_tip.y * h))
            middle_base_pos = (int(middle_base.x * w), int(middle_base.y * h))

            # Calculate hand size for dynamic threshold
            hand_size = np.sqrt((wrist_pos[0] - middle_base_pos[0])**2 + (wrist_pos[1] - middle_base_pos[1])**2)
            threshold = max(50, hand_size * 0.8)

            index_up = index_tip[1] < (wrist_pos[1] - threshold)
            thumb_up = thumb_tip[1] < (wrist_pos[1] - threshold)
            middle_up = middle_tip[1] < (wrist_pos[1] - threshold)
            ring_up = ring_tip[1] < (wrist_pos[1] - threshold)
            pinky_up = pinky_tip[1] < (wrist_pos[1] - threshold)

            # Determine hand side
            is_left_hand = handedness.classification[0].label == "Left"

            if is_left_hand:
                if index_up and thumb_up and middle_up and ring_up and pinky_up:
                    left_hand_detected = True
                    keyboard_active = True
                else:
                    keyboard_active = False
            else:
                right_hand_index_tip = index_tip
                right_hand_thumb_tip = thumb_tip

    # Handle keyboard mode
    if keyboard_active:
        mode = "Typing"
        # Draw semi-transparent virtual keyboard on overlay
        for i, row in enumerate(keyboard_layout):
            for j, key in enumerate(row):
                x = keyboard_origin[0] + j * key_size
                y = keyboard_origin[1] + i * key_size
                if key == " ":
                    space_width = 5 * key_size
                    cv2.rectangle(overlay, (x, y), (x + space_width, y + key_size), (200, 200, 200, 128), -1)
                    cv2.rectangle(overlay, (x, y), (x + space_width, y + key_size), (100, 100, 100), 2)
                    cv2.putText(overlay, "SPACE", (x + space_width//4, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                else:
                    cv2.rectangle(overlay, (x, y), (x + key_size, y + key_size), (200, 200, 200, 128), -1)
                    cv2.rectangle(overlay, (x, y), (x + key_size, y + key_size), (100, 100, 100), 2)
                    cv2.putText(overlay, key, (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Blend overlay with frame
        alpha = 0.5
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0.0)

        # Right hand typing
        if right_hand_index_tip and right_hand_thumb_tip and typing_cooldown == 0:
            distance = np.sqrt((right_hand_index_tip[0] - right_hand_thumb_tip[0])**2 + (right_hand_index_tip[1] - right_hand_thumb_tip[1])**2)
            if distance < 30:
                for i, row in enumerate(keyboard_layout):
                    for j, key in enumerate(row):
                        x = keyboard_origin[0] + j * key_size
                        y = keyboard_origin[1] + i * key_size
                        if key == " ":
                            if (x <= right_hand_index_tip[0] <= x + 5 * key_size and y <= right_hand_index_tip[1] <= y + key_size):
                                if last_key != " ":
                                    pyautogui.typewrite(" ")
                                    last_key = " "
                                    typing_cooldown = 20
                                break
                        else:
                            if (x <= right_hand_index_tip[0] <= x + key_size and y <= right_hand_index_tip[1] <= y + key_size):
                                if key != last_key:
                                    pyautogui.typewrite(key)
                                    last_key = key
                                    typing_cooldown = 20
                                break

    # Other modes (only active when keyboard is not active)
    elif results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Right":
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                index_tip = (int(index_tip.x * w), int(index_tip.y * h))
                thumb_tip = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                middle_tip = (int(middle_tip.x * w), int(middle_tip.y * h))
                wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                ring_tip = (int(ring_tip.x * w), int(ring_tip.y * h))
                pinky_tip = (int(pinky_tip.x * w), int(pinky_tip.y * h))
                middle_base_pos = (int(middle_base.x * w), int(middle_base.y * h))

                hand_size = np.sqrt((wrist_pos[0] - middle_base_pos[0])**2 + (wrist_pos[1] - middle_base_pos[1])**2)
                threshold = max(50, hand_size * 0.8)

                index_up = index_tip[1] < (wrist_pos[1] - threshold)
                thumb_up = thumb_tip[1] < (wrist_pos[1] - threshold)
                middle_up = middle_tip[1] < (wrist_pos[1] - threshold)
                ring_up = ring_tip[1] < (wrist_pos[1] - threshold)
                pinky_up = pinky_tip[1] < (wrist_pos[1] - threshold)

                distance = np.sqrt((index_tip[0] - thumb_tip[0])**2 + (index_tip[1] - thumb_tip[1])**2)

                if index_up and thumb_up and middle_up and ring_up and pinky_up:
                    mode = "Mouse"
                    screen_x = np.interp(index_tip[0], [0, w], [0, screen_width])
                    screen_y = np.interp(index_tip[1], [0, h], [0, screen_height])
                    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
                    if distance < 30 and not last_click_state:
                        pyautogui.click()
                        last_click_state = True
                    else:
                        last_click_state = False
                    center = None
                    current_time = time.time()
                    if save_cooldown == 0:
                        last_all_fingers_up_time = current_time
                    elif current_time - last_all_fingers_up_time < 0.5:
                        cv2.imwrite(f"drawing_{drawing_count}.png", paintWindow)
                        drawing_count += 1
                        save_cooldown = 30
                        last_all_fingers_up_time = 0
                    if distance < 30:
                        screenshot = pyautogui.screenshot()
                        screenshot.save(f"screenshot_{screenshot_count}.png")
                        screenshot_count += 1
                        time.sleep(1)
                elif index_up and not thumb_up and not middle_up and not ring_up and not pinky_up:
                    mode = "Draw"
                    center = index_tip
                    if center[1] <= 65:
                        if 40 <= center[0] <= 140:  # Clear
                            bpoints = [deque(maxlen=512)]
                            gpoints = [deque(maxlen=512)]
                            rpoints = [deque(maxlen=512)]
                            ypoints = [deque(maxlen=512)]
                            mpoints = [deque(maxlen=512)]
                            wpoints = [deque(maxlen=512)]
                            grpoints = [deque(maxlen=512)]
                            blue_index = green_index = red_index = yellow_index = maroon_index = white_index = grey_index = 0
                            paintWindow[67:, :, :] = 255
                        elif 160 <= center[0] <= 255:  # Blue
                            colorIndex = 0
                        elif 275 <= center[0] <= 370:  # Green
                            colorIndex = 1
                        elif 390 <= center[0] <= 485:  # Red
                            colorIndex = 2
                        elif 505 <= center[0] <= 600:  # Yellow
                            colorIndex = 3
                        elif 620 <= center[0] <= 715:  # Maroon
                            colorIndex = 4
                        elif 735 <= center[0] <= 830:  # White
                            colorIndex = 5
                        elif 850 <= center[0] <= 945:  # Grey
                            colorIndex = 6
                    else:
                        if colorIndex == 0:
                            bpoints[blue_index].appendleft(center)
                        elif colorIndex == 1:
                            gpoints[green_index].appendleft(center)
                        elif colorIndex == 2:
                            rpoints[red_index].appendleft(center)
                        elif colorIndex == 3:
                            ypoints[yellow_index].appendleft(center)
                        elif colorIndex == 4:
                            mpoints[maroon_index].appendleft(center)
                        elif colorIndex == 5:
                            wpoints[white_index].appendleft(center)
                        elif colorIndex == 6:
                            grpoints[grey_index].appendleft(center)
                elif index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
                    mode = "No Action"
                    center = None
                elif index_up and middle_up and ring_up and not thumb_up and not pinky_up:
                    mode = "Hotkey"
                    if typing_cooldown == 0:
                        keyboard.press_and_release('ctrl+c')
                        typing_cooldown = 30
                    center = None

    if not index_up or mode != "Draw":
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        mpoints.append(deque(maxlen=512))
        maroon_index += 1
        wpoints.append(deque(maxlen=512))
        white_index += 1
        grpoints.append(deque(maxlen=512))
        grey_index += 1

    # Draw lines (SMOOTHER)
    points = [bpoints, gpoints, rpoints, ypoints, mpoints, wpoints, grpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                pt1 = points[i][j][k - 1]
                pt2 = points[i][j][k]
                if pt1 is None or pt2 is None:
                    continue
                # Draw main line
                cv2.line(frame, pt1, pt2, colors[i], 4, cv2.LINE_AA)
                cv2.line(paintWindow, pt1, pt2, colors[i], 4, cv2.LINE_AA)
                # Interpolate and draw intermediate points for smoothness
                for interp_pt in interpolate_points(pt1, pt2, steps=4):
                    cv2.circle(frame, interp_pt, 2, colors[i], -1, cv2.LINE_AA)
                    cv2.circle(paintWindow, interp_pt, 2, colors[i], -1, cv2.LINE_AA)

    # Cooldown management
    if save_cooldown > 0:
        save_cooldown -= 1
    if typing_cooldown > 0:
        typing_cooldown -= 1
    if typing_cooldown == 0:
        last_key = None

    # Display current mode on frame
    cv2.putText(frame, f"Mode: {mode}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Show windows (removed mask)
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
