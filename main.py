import cv2 as cv
import numpy as np

cap = cv.VideoCapture("video.mp4")

ret, prev_frame = cap.read()

if not ret:
    exit()

prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

frames = []
frame_number = 0

def resize_frame(frame, scale=0.75):
    h, w = frame.shape[:2]
    return cv.resize(frame, (int(w*scale), int(h*scale)))

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        diff = cv.absdiff(prev_gray, gray)

        score = np.mean(diff)


        if (score > 25):
            print(score)
            label = f"Frame {frame_number+1}"

            (text_width, _), _ = cv.getTextSize(
                label,
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                5
            )

            x = (frame.shape[1] - text_width) // 2
            cv.putText(
                frame,
                label,
                (x, frame.shape[0] - 100),
                cv.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                5,
                cv.LINE_AA
            )
            frames.append(resize_frame(frame, 0.5))
            frame_number += 1


            if frame_number >= 6:
                break
        prev_gray = gray
        
    else:
        break

row1 = cv.hconcat(frames[:3])
row2 = cv.hconcat(frames[3:])

grid = cv.vconcat([row1, row2])

cv.imshow("Collage", grid)

cv.waitKey(0)

cap.release()
cv.destroyAllWindows()