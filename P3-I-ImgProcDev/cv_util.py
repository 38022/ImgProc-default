# cv_util.py
import math
import numpy as np
import cv2
from typing import Dict

# 矩形
class Rect:
    def __init__(self, dict : Dict[str, int], offset: tuple[int, int] = (0, 0)):
        self.x = dict["X"] + int(offset[0])
        self.y = dict["Y"] + int(offset[1])
        self.w = dict["W"]
        self.h = dict["H"]
    
    @classmethod
    def init(cls, x: int, y: int, w: int, h: int):
        dict = {"X" : x, "Y" : y, "W" : w, "H" : h}
        return cls(dict)

    # 左上の座標（LeftTop）
    def lt(self) -> tuple[int, int]:
        return (self.x, self.y)

    # 面積
    def area(self) -> int:
        return self.w * self.h

    # pt が中に入ってる
    def contain(self, pt: tuple[int, int]) -> bool:
        return (
            self.x <= pt[0] and
            self.y <= pt[1] and
            self.x + self.w > pt[0] and
            self.y + self.h > pt[1]
        )


# 回転矩形
class RotatedRect:
    def __init__(self, rect: Rect, angle: float):
        self.ct = (
            rect.x + (rect.w * 0.5),
            rect.y + (rect.h * 0.5)
        )
        self.size = (rect.w, rect.h)
        self.angle = angle
    
    # 回転矩形を構成する4点の座標の配列
    def points(self) -> list[tuple[float, float]]:
        rad = math.radians(self.angle)
        b = math.cos(rad) * 0.5
        a = math.sin(rad) * 0.5

        p1 = [
            (self.ct[0] - a * self.size[1] - b * self.size[0],
             self.ct[1] + b * self.size[1] - a * self.size[0]),
            (self.ct[0] + a * self.size[1] - b * self.size[0],
             self.ct[1] - b * self.size[1] - a * self.size[0])
        ]
        p2 = [
            (2 * self.ct[0] - p1[0][0],
             2 * self.ct[1] - p1[0][1]),
            (2 * self.ct[0] - p1[1][0],
             2 * self.ct[1] - p1[1][1])
        ]
        return p1 + p2

    # 回転矩形の外接矩形
    def boundingRect(self) -> Rect:
        pt = self.points()
        rect = Rect.init(
            int(math.floor(min(min(min(pt[0][0], pt[1][0]), pt[2][0]), pt[3][0]))),
            int(math.floor(min(min(min(pt[0][1], pt[1][1]), pt[2][1]), pt[3][1]))),
            int(math.ceil( max(max(max(pt[0][0], pt[1][0]), pt[2][0]), pt[3][0]))),
            int(math.ceil( max(max(max(pt[0][1], pt[1][1]), pt[2][1]), pt[3][1])))
        )
        rect.w -= (rect.x - 1)
        rect.h -= (rect.y - 1)
        return rect

    # 回転矩形の中に含まれる
    def contain(self, pt: tuple[int, int]) -> bool:
        if (self.angle == 0):
            return self.boundingRect().contain(pt)

        pts = self.points()
        chk = True

        for i in range(4):
            m = (pts[i][1] - pts[(i + 1) % 4][1]) / (pts[i][0] - pts[(i + 1) % 4][0])
            c = pts[i][1] - m * pts[i][0]

            val_pt = pt[1] - m * pt[0] - c
            val_center = self.ct[1] - m * self.ct[0] - c;

            if (val_pt * val_center > 0):
                chk = True
            else:
                chk = False
                break
        
        return chk

# クロッピング
def crop(img: cv2.Mat, rect: Rect) -> cv2.Mat:
    return img[
        rect.y :         # top
        rect.y + rect.h, # bottom
        rect.x :         # left
        rect.x + rect.w  # right
    ]  

# 画像の回転
def rotate(img: cv2.Mat, angle: float) -> cv2.Mat:
    h, w = img.shape[:2]
    center = (int(w/2), int(h/2))
    trans = cv2.getRotationMatrix2D(center, angle, 1.0)
    #アフィン変換
    return cv2.warpAffine(img, trans, (w, h))

# 指定位置に貼り付け
def paste(base: cv2.Mat, img: cv2.Mat, rect: Rect) -> cv2.Mat:
    tmp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    base[
        rect.y :         # top
        rect.y + rect.h, # bottom
        rect.x :         # left
        rect.x + rect.w  # right
    ] = tmp
    return base

# 輪郭線の重心を求める
def moments(c, offset: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    mu = cv2.moments(c)
    return (
        int(mu["m10"]/mu["m00"]) + offset[0],
        int(mu["m01"]/mu["m00"]) + offset[1]
    )

# 二値化
def threshold(src: cv2.Mat, th: int) -> cv2.Mat:
    if (th < 0):
        return cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    elif (th == 0):
        return src
    else:
        return cv2.threshold(src, th, 255, cv2.THRESH_BINARY)[1]

# ガウシアンぼかし
def blur(src: cv2.Mat, size: tuple[int, int]) -> cv2.Mat:
    return cv2.GaussianBlur(src, (size), 0)

# ガンマ補正
def gamma(src: cv2.Mat, gamma: float) -> cv2.Mat:
    # ルックアップテーブルを作成
    lut = (np.arange(256.0) / 255.0) ** (1.0 / gamma) * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    # ルックアップテーブルで計算
    return cv2.LUT(src, lut)

# コントラスト補正
def contrast(src : cv2.Mat, contrast: float) -> cv2.Mat:
    if (contrast == 0.0):
        return src

    # ルックアップテーブルを作成
    lut = 255.0 / (1.0 + np.exp(-contrast * np.arange(-128, 128) / 255.0))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return cv2.LUT(src, lut)

# 矩形の描画
def drawRect(src: cv2.Mat, rect: Rect, color: tuple[int, int, int], thickness: int = 1):
    cv2.rectangle(
        src,
        (rect.x, rect.y),
        (rect.x + rect.w, rect.y + rect.h),
        color,
        thickness)

# 回転矩形の描画
def drawRRect(src: cv2.Mat, rr: RotatedRect, color: tuple[int, int, int], thickness: int = 1):
    pts = rr.points()
    for i in range(4):
        p1 = (int(pts[i][0]), int(pts[i][1]))
        p2 = (int(pts[(i + 1) % 4][0]), int(pts[(i + 1) % 4][1]))
        cv2.line(src, p1, p2, color, thickness)

# 画像を読み込んでMatにして返す
def read(path: str) -> cv2.Mat:
    buf = np.fromfile(path, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

# 黒一色画像
#def black(width: int, height: int, color: bool = True, dtype: np._DTypeLike = np.uint8) -> cv2.Mat:
#    return np.zeros((height, width, 3 if color else 1), dtype = dtype)

def black(width: int, height: int, color: bool = True, dtype = np.uint8): # type: ignore
    return np.zeros((height, width, 3 if color else 1), dtype)
