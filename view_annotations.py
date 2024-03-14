import numpy as np
import cv2
import os


number_of_masks = len(os.listdir(os.getcwd() + "/data/masks"))
img_rgb = cv2.imread("data/images/000000.png")
img_seg = cv2.imread("data/masks/000000.png")

imgray2 = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray2, 0,255, cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_rgb, contours, -1, (0,255,0), 1)

height, width, _ = img_rgb.shape
min_x, min_y = width, height
max_x = max_y = 0

# computes the bounding box for the contour, and draws it on the frame,
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    if w > 10 and h > 10:
        cv2.rectangle(img_rgb, (x,y), (x+w,y+h), (255, 0, 0), 1)
        
cv2.imshow(f"Image",img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()