"""Create a temporary contact sheet from a locally downloaded race video."""

import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: video_contact_sheet <input.mp4> <output.png>")
    capture = cv2.VideoCapture(sys.argv[1])
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or count <= 0:
        raise SystemExit("unable to decode video")
    canvas = np.zeros((4 * 360, 3 * 640, 3), dtype=np.uint8)
    for index, frame_index in enumerate(np.linspace(0, count - 1, 12).astype(int)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"unable to decode frame {frame_index}")
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"{frame_index / fps:.1f}s", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        row, col = divmod(index, 3)
        canvas[row * 360:(row + 1) * 360, col * 640:(col + 1) * 640] = frame
    capture.release()
    if not cv2.imwrite(sys.argv[2], canvas):
        raise SystemExit("unable to save contact sheet")
    print(sys.argv[2])


if __name__ == "__main__":
    main()
