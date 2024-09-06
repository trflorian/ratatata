import sys
import time
from collections import deque

import cv2
import numpy as np

from mss import mss
from pyautogui import press

border_around_figures = 10
top_line_distance = 5
detection_box_height = 50
color_box_size = 30
debounce_time_between_key_presses = 0.25  # seconds

keys = ["a", "s", "d", "f"]

template = cv2.imread("template.png")

sct = mss()

# calibrate
while True:
    sct_img = sct.grab(sct.monitors[1])
    img_rgba = np.array(sct_img)
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

    # match template
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # find the top 4 matches iteratively
    matches = []
    for i in range(4):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        matches.append(max_loc)

        result[
            max_loc[1]
            - border_around_figures : max_loc[1]
            + border_around_figures
            + template.shape[0],
            max_loc[0]
            - border_around_figures : max_loc[0]
            + border_around_figures
            + template.shape[1],
        ] = 0

    # draw rectangles around the matches
    for match in matches:
        cv2.rectangle(
            img,
            match,
            (match[0] + template.shape[1], match[1] + template.shape[0]),
            (0, 255, 0),
            2,
        )

    # get the y position n pixels above the average of the matches
    detection_line_y = int(np.mean([match[1] for match in matches]) - top_line_distance)

    # draw a line at the y position
    cv2.line(
        img, (0, detection_line_y), (img.shape[1], detection_line_y), (0, 0, 255), 2
    )

    start_x = np.min([match[0] for match in matches])
    end_x = np.max([match[0] for match in matches]) + template.shape[1]
    cv2.rectangle(
        img,
        (start_x, detection_line_y - detection_box_height // 2),
        (end_x, detection_line_y + detection_box_height // 2),
        (0, 0, 255),
        2,
    )

    cv2.imshow("Caliration", img)
    key = cv2.waitKey(1)

    if key == ord("q"):
        cv2.destroyAllWindows()
        sys.exit()
    if key == 13:  # Enter
        break

cv2.destroyAllWindows()
print("Calibration done")

# order matches by x position
matches = sorted(matches, key=lambda x: x[0])

start_x = np.min([match[0] for match in matches])
end_x = np.max([match[0] for match in matches]) + template.shape[1]
bbox = {
    "top": detection_line_y + 90,
    "left": start_x,
    "width": end_x - start_x,
    "height": detection_box_height,
    "mon": 1,
}

# debounce key presses
last_key_presses = [time.perf_counter() for _ in range(4)]

key_buffer = deque(maxlen=2)

while True:
    # capture screen from min_x to max_x of matches and 20px around the top line
    sct_img = sct.grab(bbox)
    img_rgba = np.array(sct_img)
    img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    curr_time = time.perf_counter()

    key_presses = []

    for i in range(4):
        x = matches[i][0] - start_x + template.shape[1] // 2
        dx = x - (end_x - start_x) // 2
        x = int((x - dx) + dx * 0.9)
        y = detection_box_height // 2
        crop = img_hsv[
            y - color_box_size // 2 : y + color_box_size // 2,
            x - color_box_size // 2 : x + color_box_size // 2,
        ]
        hsv = np.mean(crop, axis=(0, 1))
        sat = hsv[1]
        val = hsv[2]
        time_since_last_key_press = curr_time - last_key_presses[i]
        if sat < 50 and 50 < val < 180:
            if time_since_last_key_press > debounce_time_between_key_presses:
                last_key_presses[i] = curr_time
                key_presses.append(keys[i])

        is_clicked = time_since_last_key_press < debounce_time_between_key_presses
        col = (0, 0, 255) if is_clicked else (255, 255, 255)

        if is_clicked:
            cv2.circle(img, (x, y), 5, col, -1)
        cv2.rectangle(
            img,
            (x - color_box_size // 2, y - color_box_size // 2),
            (x + color_box_size // 2, y + color_box_size // 2),
            col,
            2,
        )

    cv2.imshow("Detection", img)

    key_buffer.append(key_presses)

    if len(key_buffer) == key_buffer.maxlen:
        for key in key_buffer[0]:
            press(key)

        keys_str = []
        for i in range(4):
            if keys[i] in key_buffer[0]:
                keys_str.append(keys[i].upper())
            else:
                keys_str.append("_")
        
        print(f"[{' '.join(keys_str)}]")

    key = cv2.waitKey(5)

    if key == ord("q"):
        break


cv2.destroyAllWindows()
