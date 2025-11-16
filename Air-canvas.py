import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# ---------------- Mediapipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- Canvas Setup ----------------
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

# 10 Colors + Commands
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255),
    (0, 128, 255), (128, 0, 128), (255, 0, 255), (255, 255, 0),
    (42, 42, 165), (0, 0, 0)
]
color_names = ["BLUE","GREEN","RED","YELLOW","ORANGE","PURPLE","PINK","CYAN","BROWN","BLACK"]
commands = ["UNDO", "REDO", "CLEAR"]
colorIndex = 0  # Default color

# Storage for drawing
points_storage = [ [deque(maxlen=1024)] for _ in range(10) ]
points_index = [0]*10

undo_stack = []
redo_stack = []

hand_present = False
drawing_mode = True  # True = drawing, False = cursor mode

# ---------------- Finger Counting ----------------
def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# ---------------- Video Capture ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)

    cursor_x, cursor_y = None, None

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        x = int(hand.landmark[8].x * frame.shape[1])
        y = int(hand.landmark[8].y * frame.shape[0])
        fingers = count_fingers(hand)

        if fingers == 0:
            drawing_mode = False
            cursor_x, cursor_y = x, y  # Cursor active

            # Prepare a new line for the current color so it won't continue the previous stroke
            if points_index[colorIndex] == len(points_storage[colorIndex]) - 1:
                points_storage[colorIndex].append(deque(maxlen=1024))
                points_index[colorIndex] += 1

        else:
            drawing_mode = True
            # Draw with selected color
            points_storage[colorIndex][points_index[colorIndex]].appendleft((x, y))

        hand_present = True
    else:
        hand_present = False

    if not drawing_mode and cursor_x is not None:
        # Check if cursor clicks a color button
        button_width = paintWindow.shape[1] // 10
        for i in range(10):
            if 0 <= cursor_y <= 40 and i*button_width <= cursor_x < (i+1)*button_width:
                colorIndex = i

        # Check command buttons below color buttons
        cmd_width = paintWindow.shape[1] // 3
        for i, cmd in enumerate(commands):
            if 50 <= cursor_y <= 90 and i*cmd_width <= cursor_x < (i+1)*cmd_width:
                if cmd == "UNDO" and points_storage[colorIndex] and len(points_storage[colorIndex])>1:
                    undo_stack.append(points_storage[colorIndex].pop())
                    points_index[colorIndex]-=1
                elif cmd == "REDO" and undo_stack:
                    points_storage[colorIndex].append(undo_stack.pop())
                    points_index[colorIndex]+=1
                elif cmd == "CLEAR":
                    paintWindow[:] = 255
                    points_storage = [ [deque(maxlen=1024)] for _ in range(10)]
                    points_index = [0]*10
                    undo_stack.clear()
                    redo_stack.clear()

    # Draw strokes
    for i, color_points in enumerate(points_storage):
        for line in color_points:
            for k in range(1, len(line)):
                if line[k-1] and line[k]:
                    cv2.line(paintWindow, line[k-1], line[k], colors[i], 2)

    # ---------------- Draw Paint App UI ----------------
    overlay = paintWindow.copy()

    # Draw color palette
    color_width = overlay.shape[1] // 10
    for i, c in enumerate(colors):
        cv2.rectangle(overlay, (i*color_width,0), ((i+1)*color_width,40), c, -1)
        cv2.putText(overlay, str(i+1), (i*color_width+10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    # Draw command buttons
    cmd_width = overlay.shape[1] // 3
    for i, cmd in enumerate(commands):
        cv2.rectangle(overlay, (i*cmd_width,50), ((i+1)*cmd_width,90), (200,200,200), -1)
        cv2.putText(overlay, cmd, (i*cmd_width+10,80), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    # Draw cursor
    if not drawing_mode and cursor_x is not None:
        cv2.circle(overlay, (cursor_x, cursor_y), 10, (0,0,255), -1)

    # Draw current color label
    cv2.putText(overlay, f"Current Color: {color_names[colorIndex]}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[colorIndex], 2)

    cv2.imshow("Finger Tracking", frame)
    cv2.imshow("Paint", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
