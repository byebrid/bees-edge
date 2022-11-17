import cv2
import string
from fps import FPS

KEYS = {
    "space": 32,
    "left": 91,  # Note this is actually "[", since left/right arrows are bugged in opencv with QT GUI (https://github.com/opencv/opencv/issues/20215)
    "right": 93,  # This is "]"
    "esc": 27,
    "none": 255,
}
# Add lowercase letters to dictionary as well so we better recognise user inputs
KEYS.update({l: ord(l) for l in string.ascii_lowercase})


cap = cv2.VideoCapture(index=0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

fps = FPS()
i = 0
while True:
    i += 1
    fps.tick()
    if i % 30 == 0:
        print(f"FPS: {fps.get_average()}")

    grabbed, frame = cap.read()
    cv2.imshow(f"Input", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == KEYS["q"]:
        print("User pressed 'q', exiting...")
        break


cap.release()
cv2.destroyAllWindows()