#!/usr/bin/env python

'''
Inpainting sample.

Inpainting repairs damage to images by floodfilling
the damage with surrounding image areas.

Usage:
  inpaint.py [<image>]

Keys:
  SPACE - inpaint
  r     - reset the inpainting mask
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
# from common import Sketcher

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'data/fruits.jpg'

    print(__doc__)

    class Sketcher:
        def __init__(self, windowname, dests, colors_func):
            self.prev_pt = None
            self.windowname = windowname
            self.dests = dests
            self.colors_func = colors_func
            self.dirty = False
            self.show()
            cv2.setMouseCallback(self.windowname, self.on_mouse)

        def show(self):
            cv2.imshow(self.windowname, self.dests[0])

        def on_mouse(self, event, x, y, flags, param):
            pt = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                self.prev_pt = pt
            elif event == cv2.EVENT_LBUTTONUP:
                self.prev_pt = None

            if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
                for dst, color in zip(self.dests, self.colors_func()):
                    cv2.line(dst, self.prev_pt, pt, color, 50)
                self.dirty = True
                self.prev_pt = pt
                self.show()

    img = cv2.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)
    sketch = Sketcher('img', [img_mark, mark], lambda : ((255, 255, 255), 100))

    while True:
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            # res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
            res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_NS)
            cv2.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()
    cv2.destroyAllWindows()
