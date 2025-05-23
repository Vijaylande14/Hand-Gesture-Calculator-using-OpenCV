import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger indices
finger_tips = [4, 8, 12, 16, 20]
finger_pips = [2, 6, 10, 14, 18]

# Open webcam
cap = cv2.VideoCapture(0)

# Default operation
current_op = 'a'  # a = add, s = sub, m = mul, d = div

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_counts = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            landmarks = hand_landmarks.landmark
            hand_count = 0

            # Thumb logic
            if hand_label == 'Right':
                if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
                    hand_count += 1
            else:
                if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
                    hand_count += 1

            for j in range(1, 5):
                if landmarks[finger_tips[j]].y < landmarks[finger_pips[j]].y:
                    hand_count += 1

            finger_counts.append(hand_count)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Perform selected operation
    if len(finger_counts) == 2:
        a, b = finger_counts
        if current_op == 'a':
            result = f"{a} + {b} = {a + b}"
        elif current_op == 's':
            result = f"{a} - {b} = {a - b}"
        elif current_op == 'm':
            result = f"{a} * {b} = {a * b}"
        elif current_op == 'd':
            result = f"{a} / {b} = {a / b:.2f}" if b != 0 else "Division by 0"
        else:
            result = "Press a/s/m/d"
        cv2.putText(img, result, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    elif len(finger_counts) == 1:
        cv2.putText(img, f"Fingers: {finger_counts[0]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
    else:
        cv2.putText(img, "No hands detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Display current operation
    op_name = {'a': 'Addition', 's': 'Subtraction', 'm': 'Multiplication', 'd': 'Division'}
    cv2.putText(img, f"Mode: {op_name.get(current_op, 'None')}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Finger Math", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in [ord('a'), ord('s'), ord('m'), ord('d')]:
        current_op = chr(key)

cap.release()
cv2.destroyAllWindows()
