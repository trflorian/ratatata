import sys
import collections

import cv2
import numpy as np

from mss import mss

from utils import capture_rgb_screen

CalibrationData = collections.namedtuple(
    "CalibrationData",
    [
        "matches",
        "template_width",
        "start_x",
        "end_x",
        "start_y",
        "end_y",
    ],
)


def calibrate_with_template(
    border_around_figures: int = 10,
    top_line_distance: int = 5,
    detection_box_height: int = 50,
) -> CalibrationData:
    sct = mss()

    template = cv2.imread("template.png")

    while True:
        img = capture_rgb_screen(sct, sct.monitors[1])

        # match template
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # find the top 4 matches iteratively
        matches = []
        for _ in range(4):
            _, _, _, max_loc = cv2.minMaxLoc(result)
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

        # order matches by x position
        matches = sorted(matches, key=lambda x: x[0])

        # get the y position n pixels above the average of the matches
        detection_line_y = int(
            np.mean([match[1] for match in matches]) - top_line_distance
        )

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
            return None
        if key == 13:  # Enter
            return CalibrationData(
                matches=matches,
                template_width=template.shape[1],
                start_x=start_x,
                end_x=end_x,
                start_y=detection_line_y - detection_box_height // 2,
                end_y=detection_line_y + detection_box_height // 2,
            )
