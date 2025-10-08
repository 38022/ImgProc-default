# back_ground.py
from queue import Queue
import numpy as np
import cv2
import cv_util

class BackGround(object):
    def __init__(self, w: int, h: int):
        self.bg8 = cv_util.black(w, h, color = False, dtype = np.uint8)
        self.bg16 = cv_util.black(w, h, color = False, dtype = np.uint16)
        self.data: "Queue[cv2.Mat]" = Queue(100)

    def enq(self, src: cv2.Mat) -> cv2.Mat:
        # 追加
        self.data.put(src)

        if (self.data.full()):
            m8 = self.data.get()
            m16 = m8.astype(np.uint16)
            self.bg16 = cv2.subtract(self.bg16, m16)

        # 背景更新
        m16 = src.astype(np.uint16)
        self.bg16 = cv2.add(self.bg16, m16)

        # 平均算出
        ave = (self.bg16 / self.data.maxsize)
        return ave.astype(np.uint8)
