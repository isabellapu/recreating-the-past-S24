import cv2
import numpy as np
import mediapipe as mp
import random

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 0.75
font_thickness = 1

letters_colors = {
    "A": (0, 255, 0),
    "G": (0, 255, 255), 
    "C": (255, 0, 0), 
    "T": (0, 0, 255)   
}

letter_grid = []
change_probability = 0.02 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)

    results = segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    mask = (results.segmentation_mask > 0.7).astype(np.uint8) * 255

    text_fill = np.zeros_like(frame)

    max_text_dimensions = max([cv2.getTextSize(letter, font, font_scale, font_thickness)[0] for letter in letters_colors])
    max_text_width, max_text_height = max_text_dimensions

    y_positions = range(0, frame.shape[0], max_text_height + 5)
    x_positions = range(0, frame.shape[1], max_text_width + 5)
    
    if not letter_grid:
        letter_grid = [[random.choice(list(letters_colors.keys())) for _ in x_positions] for _ in y_positions]

    for y_idx, y in enumerate(y_positions):
        for x_idx, x in enumerate(x_positions):
            if random.random() < change_probability:
                letter_grid[y_idx][x_idx] = random.choice(list(letters_colors.keys()))
            letter = letter_grid[y_idx][x_idx]
            color = letters_colors[letter]
            cv2.putText(text_fill, letter, (x, y), font, font_scale, color, font_thickness)

    silhouette_image = cv2.bitwise_and(text_fill, text_fill, mask=mask)

    cv2.imshow('Silhouette', silhouette_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()