import cv2
import dlib
import code
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('face_landmarks.dat')

cap = cv2.VideoCapture('id2_vcd_swwp2s.mpg')
cv2.namedWindow('frame')


def part_at(shape, num):
    p = shape.part(num)
    return (p.x, p.y)


# Returns a rectangle surrounding the mouth region in the form (x, y, w, h)
def get_mouth_rect(shape):
    points = [shape.part(48), shape.part(51), shape.part(54), shape.part(57)]
    mat = np.matrix([ [p.x, p.y] for p in points ])
    rect = cv2.boundingRect(mat)

    xy0 = (rect[0], rect[1])
    xy1 = (rect[0] + rect[2], rect[1] + rect[3])

    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2

    return (cx - 50, cy - 25), (cx + 50, cy + 25)



key = 0
ret, img = cap.read()

# code.interact(local=locals())

while ret and chr(key) != 'q':

    d = detector(img)[0]
    shape = predictor(img, d)

    xy0, xy1 = get_mouth_rect(shape)

    cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 1)
    cv2.rectangle(img, xy0, xy1, (255, 0, 0), 1)
    # cv2.circle(img, part_at(shape, 57), 2, (0, 0, 255), 1)
    cv2.imshow('frame', img)
    key = cv2.waitKey(40)

    ret, img = cap.read()

cv2.destroyAllWindows()
# code.interact(local=locals())


