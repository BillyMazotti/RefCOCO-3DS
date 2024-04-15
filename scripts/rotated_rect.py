import numpy
import cv2
import math

class RRect_center:
  def __init__(self, p0, s, ang):
    (self.W, self.H) = s # rectangle width and height
    self.d = 1.1 * math.sqrt(self.W**2 + self.H**2)/2.0 # distance from center to vertices    
    self.c = p0
    self.ang = ang # rotation angle
    self.alpha = math.radians(self.ang) # rotation angle in radians
    self.beta = math.atan2(self.H, self.W) # angle between d and horizontal axis
    # Center Rotated vertices in image frame
    self.P0 = (int(self.c[0] - self.d * math.cos(self.beta - self.alpha)), int(self.c[1] - self.d * math.sin(self.beta-self.alpha))) 
    self.P1 = (int(self.c[0] - self.d * math.cos(self.beta + self.alpha)), int(self.c[1] + self.d * math.sin(self.beta+self.alpha))) 
    self.P2 = (int(self.c[0] + self.d * math.cos(self.beta - self.alpha)), int(self.c[1] + self.d * math.sin(self.beta-self.alpha))) 
    self.P3 = (int(self.c[0] + self.d * math.cos(self.beta + self.alpha)), int(self.c[1] - self.d * math.sin(self.beta+self.alpha))) 

    self.verts = [self.P0,self.P1,self.P2,self.P3]

  def draw(self, image):
    # print(self.verts)
    for i in range(len(self.verts)-1):
      cv2.line(image, (self.verts[i][0], self.verts[i][1]), (self.verts[i+1][0],self.verts[i+1][1]), (0,255,0), 2)
    cv2.line(image, (self.verts[3][0], self.verts[3][1]), (self.verts[0][0], self.verts[0][1]), (0,255,0), 2)


if __name__ == "__main__":
  image = numpy.zeros((1000,1000,3))
  (W, H) = (300,600)
  ang = 0 #degrees
  P0 = (750,500)
  rr = RRect_center(P0,(W,H),ang)
  rr.draw(image)
  image[500,500,:] = [0,0,255]
  cv2.imshow("Text", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()