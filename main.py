import sys
import time
from collections import deque

import cv2
import numpy as np

from mss import mss
from pyautogui import press

from calibration import calibrate_with_template
from utils import capture_rgb_screen


def play_game_with_calibration(
    matches: list,
    tw: int,
    sx: int,
    ex: int,
    sy: int,
    ey: int,
    color_box_size: int = 30,
    debounce_time_between_key_presses: float = 0.25,
    keys: list = ["a", "s", "d", "f"],
    delay_frames: int = 2,
) -> None:
    """
    Play the RATATAA game using the calibration data.
    
    Args:
        matches: list of tuples containing the coordinates of the top-left corner of the matches
        tw: width of the template
        sx: x-coordinate of the start of the game window
        ex: x-coordinate of the end of the game window
        sy: y-coordinate of the start of the game window
        ey: y-coordinate of the end of the game window
        color_box_size: size of the color box
        debounce_time_between_key_presses: time in seconds to wait between key presses
        keys: list of keys to press
        delay_frames: number of frames to delay the key presses
    """

    sct = mss()

    w = ex - sx
    h = ey - sy

    bbox = {
        "left": sx,
        "width": w,
        "top": sy + 90,  # offset for the top bar
        "height": h,
        "mon": 1,
    }

    # debounce key presses
    last_key_presses = [time.perf_counter() for _ in range(4)]

    key_buffer = deque(maxlen=delay_frames)

    while True:
        img = capture_rgb_screen(sct, bbox)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        curr_time = time.perf_counter()

        key_presses = []

        for i in range(4):
            x = matches[i][0] - sx + tw // 2
            y = h // 2

            # correct x position for skewed lines
            dx = x - w // 2
            x = int((x - dx) + dx * 0.9)

            crop = img_hsv[
                y - color_box_size // 2 : y + color_box_size // 2,
                x - color_box_size // 2 : x + color_box_size // 2,
            ]

            # calculate average saturation and value
            hsv = np.mean(crop, axis=(0, 1))
            sat = hsv[1]
            val = hsv[2]

            time_since_last_key_press = curr_time - last_key_presses[i]
            if sat < 50 and 50 < val < 180:
                if time_since_last_key_press > debounce_time_between_key_presses:
                    last_key_presses[i] = curr_time
                    key_presses.append(keys[i])

            # Draw annotations
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

        cv2.imshow("Detection", img)

        key = cv2.waitKey(5)

        if key == ord("q"):
            return


def run() -> None:
    calibration_data = calibrate_with_template()

    if calibration_data is None:
        print("Calibration aborted.")
        return

    # unpack calibration data
    play_game_with_calibration(
        matches=calibration_data.matches,
        tw=calibration_data.template_width,
        sx=calibration_data.start_x,
        ex=calibration_data.end_x,
        sy=calibration_data.start_y,
        ey=calibration_data.end_y,
    )


if __name__ == "__main__":
    try:
        run()
    finally:
        cv2.destroyAllWindows()
