import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

# Variables to track the detected character and the time it has been held
last_detected_character = None
start_time = None
hold_duration = 3  # duration to hold the same character in seconds

# Variables to track blank frame duration
last_detection_time = time.time()
blank_frame_duration = 5  # duration to wait for adding space

# String to store detected characters
detected_string = ""

# Button coordinates
clear_button_coords = (10, 150, 110, 190)
speak_button_coords = (130, 150, 230, 190)

def check_button_click(x, y, button_coords):
    x1, y1, x2, y2 = button_coords
    return x1 <= x <= x2 and y1 <= (y-480) <= y2

def clear_string(event, x, y, flags, param):
    global detected_string
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at: ({x}, {y})")
        if check_button_click(x, y, clear_button_coords):
            print("Clear button clicked")
            detected_string = ""
        elif check_button_click(x, y, speak_button_coords):
            print("Speak button clicked")
            engine.say(detected_string)
            engine.runAndWait()

# Set the mouse callback function to capture button clicks
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', clear_string)

# Define colors
canvas_color = [66, 45, 43]  # #2b2d42 (dark blue)
button_bg_color = [41, 4, 217]  # #d90429 (red)
button_text_color = [244, 242, 237]  # #edf2f4 (light gray)
square_color =[41, 4, 217]   # Red color in BGR format

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Process only the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for landmark in hand_landmarks.landmark:  # Corrected attribute here
            x = landmark.x
            y = landmark.y
            x_.append(x)
            y_.append(y)

        for landmark in hand_landmarks.landmark:  # Corrected attribute here
            x = landmark.x
            y = landmark.y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character == last_detected_character:
            if time.time() - start_time >= hold_duration:
                detected_string += predicted_character
                last_detected_character = None
        else:
            last_detected_character = predicted_character
            start_time = time.time()

        last_detection_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        if time.time() - last_detection_time >= blank_frame_duration:
            detected_string += " "
            last_detection_time = time.time()

    # Draw the red square inside the top-left corner of the frame
    square_size = int(min(W, H) * 0.5 * 1.25)  # 125% of the original size
    square_x1 = 0
    square_y1 = 0
    square_x2 = square_x1 + square_size
    square_y2 = square_y1 + square_size
    cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), square_color, 2)

    # Create a mask for the area inside the square
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.rectangle(mask, (square_x1, square_y1), (square_x2, square_y2), (255, 255, 255), -1)

    # Blur the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

    # Combine the blurred frame with the clear area using the mask
    frame = np.where(mask == np.array([255, 255, 255]), frame, blurred_frame)

    # Create a plain color canvas
    canvas_height = 220
    canvas = np.full((canvas_height, W, 3), canvas_color, dtype=np.uint8)

    # Display the detected string on the canvas
    cv2.putText(canvas, f"Characters: {detected_string}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, button_text_color, 2, cv2.LINE_AA)

    # Draw buttons on the canvas
    button_margin = 10

    # Clear Button
    clear_x1, clear_y1, clear_x2, clear_y2 = clear_button_coords
    cv2.rectangle(canvas, (clear_x1, clear_y1), (clear_x2, clear_y2), button_bg_color, -1)
    clear_text = "Clear"
    (text_width, text_height), baseline = cv2.getTextSize(clear_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = clear_x1 + (clear_x2 - clear_x1 - text_width) // 2
    text_y = clear_y1 + (clear_y2 - clear_y1 + text_height) // 2 - baseline
    cv2.putText(canvas, clear_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, button_text_color, 2)

    # Speak Button
    speak_x1, speak_y1, speak_x2, speak_y2 = speak_button_coords
    cv2.rectangle(canvas, (speak_x1, speak_y1), (speak_x2, speak_y2), button_bg_color, -1)
    speak_text = "Speak"
    (text_width, text_height), baseline = cv2.getTextSize(speak_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = speak_x1 + (speak_x2 - speak_x1 - text_width) // 2
    text_y = speak_y1 + (speak_y2 - speak_y1 + text_height) // 2 - baseline
    cv2.putText(canvas, speak_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, button_text_color, 2)

    # Combine the frame and the canvas
    combined_frame = np.vstack((frame, canvas))

    cv2.imshow('frame', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
