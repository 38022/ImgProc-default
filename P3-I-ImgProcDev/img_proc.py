# img_proc.py
import datetime as dt
import os
import math
import numpy as np
import cv2
from typing import Any

import back_ground as bg
import cv_util

COLOR_CROP = (50, 50, 50)
COLOR_TEMPLATE = (0, 0, 255)
COLOR_RRECT = (230, 230, 230)
COLOR_BRECT = (170, 170, 170)
COLOR_CONTOUR = (30, 200, 30)
COLOR_EDGE = (0, 255, 0)

class ImgProc(object):
    # 生成
    def __init__(self, w: int, h: int, template: str):
        # 背景
        self.BG = bg.BackGround(w, h)
        self.background: cv2.Mat
        # テンプレートマッチング用のテンプレート
        if (template != None and os.path.isfile(template)):
            self.template = cv2.cvtColor(cv_util.read(template), cv2.COLOR_BGR2GRAY)
        
        self.last_tm_time = dt.datetime(2022, 1, 1)
        self.offset = (0, 0)
        # 1フレーム前の画像
        self.prev = cv_util.black(w, h, color = False, dtype = np.uint8)
        

    def __del__(self):
        pass

    def proc(self, src: cv2.Mat, r: Any, draw: bool) -> tuple[cv2.Mat, list[float]]:
        # クロッピング
        if (r["IsCrop"]):
            crop_rect = cv_util.Rect(r["CropRect"])
            src = cv_util.crop(src, crop_rect)

        # 回転
        if ("RotateAngle" in r and r["RotateAngle"] != 0):
            src = cv_util.rotate(src, r["RotateAngle"])

        # 出力画像
        dst = src
        
        # グレースケール変換
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 背景減算
        self.background = self.BG.enq(src)

        # テンプレートマッチング
        if (r["OffsetImagePath"] != None):
            # 1分に一回
            if (dt.datetime.now() - self.last_tm_time > dt.timedelta(minutes = 1)):
                # テンプレートマッチング
                res = cv2.matchTemplate(src, self.template, cv2.TM_CCOEFF)
                _, _, _, max = cv2.minMaxLoc(res)
                self.offset = (
                    max[0] - r["OffsetZeroPoint"]["X"],
                    max[1] - r["OffsetZeroPoint"]["Y"]
                    )

                self.last_tm_time = dt.datetime.now()
            
            if (draw):
                # 描画
                h, w = self.template.shape[:2]
                x = self.offset[0] + r["OffsetZeroPoint"]["X"]
                y = self.offset[1] + r["OffsetZeroPoint"]["Y"]
                tr = cv_util.Rect.init(x, y, w, h)
                cv_util.drawRect(dst, tr, COLOR_TEMPLATE)
                    

        result: list[float] = []
        for p in r["Procs"]:
            if (p["Mode"] == 0):
                # 1フレーム
                result += self.frame_one(src, dst, p["ROI"], draw)
            if (p["Mode"] == 1):
                # 2フレーム
                result += self.frame_sub(src, dst, p["ROI"], draw)
            if (p["Mode"] == 2):
                # 幅測定
                result += self.measure_length(src, dst, p["Edge1"], p["Edge2"], draw)
            if (p["Mode"] == 3 or p["Mode"] == 4):
                result += self.clip_henkei(src, dst, p, draw)
            # ここに新たなモードを追加


        return dst, result

    def frame_one(self, src, dst, roi, draw) -> list[float]:
        result: list[float] = []

        # 本来の矩形
        org = cv_util.Rect(roi["Rect"], self.offset)
        # 回転矩形
        rr = cv_util.RotatedRect(org, roi["Angle"])
        # 外接矩形
        br = rr.boundingRect()

        if (draw):
            cv_util.drawRRect(dst, rr, COLOR_RRECT)
            cv_util.drawRect(dst, br, COLOR_BRECT)

        # クロッピング
        crop = cv_util.crop(src, br)

        if (roi["IsSubBG"]):
            # 背景減算
            crop = cv2.subtract(crop, cv_util.crop(self.background, br))

        morph_size = (5, 5) if (br.area() > 50000) else (3, 3)
            
        # ぼかし
        crop = cv_util.blur(crop, morph_size)
        # 二値化
        crop = cv_util.threshold(crop, roi["Thresh"])
        # オープン
        crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, morph_size)


        # 輪郭抽出
        cts1 = cv2.findContours(crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        cts2 = [c for c in cts1
            # 面積でフィルタ
            if cv2.contourArea(c) > roi["ContourTh"]
            # 回転矩形の中に入ってる
            and rr.contain(cv_util.moments(c, br.lt()))
        ]

        if (draw):
            cv2.drawContours(dst, cts2, -1, COLOR_CONTOUR, offset = br.lt())

        # 面積の合計
        result.append(sum([cv2.contourArea(c) for c in cts2]))
        
        return result

    def frame_sub(self, src, dst, roi, draw) -> list[float]:
        result: list[float] = []

        # 減算画像
        sub = cv2.subtract(src, self.prev)

        # 本来の矩形
        org = cv_util.Rect(roi["Rect"], self.offset)
        # 回転矩形
        rr = cv_util.RotatedRect(org, roi["Angle"])
        # 外接矩形
        br = rr.boundingRect()

        # クロッピング
        crop = cv_util.crop(sub, br)

        morph_size = (5, 5) if (br.area() > 50000) else (3, 3)
            
        # ぼかし
        crop = cv_util.blur(crop, morph_size)
        # 二値化
        crop = cv_util.threshold(crop, roi["Thresh"])
        # オープン
        crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, morph_size)

        if (draw):
            cv_util.drawRect(dst, br, COLOR_BRECT)

        # 輪郭抽出
        cts1 = cv2.findContours(crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

        cts2 = [c for c in cts1
            # 面積でフィルタ
            if cv2.contourArea(c) > roi["ContourTh"]
            # 回転矩形の中に入ってる
            and rr.contain(cv_util.moments(c, br.lt()))
        ]

        if (draw):
            cv2.drawContours(dst, cts2, -1, COLOR_CONTOUR, offset = br.lt())

        # 面積の合計
        result.append(sum([cv2.contourArea(c) for c in cts2]))
        
        self.prev = src
        return result

    def measure_length(self, src, dst, r1, r2, draw) -> list[float]:
        result: list[float] = []

        # 本来の矩形
        rect1 = cv_util.Rect(r1["Rect"], self.offset)
        rect2 = cv_util.Rect(r2["Rect"], self.offset)
            

        # クロッピング
        crop1 = cv_util.crop(src, rect1)
        crop2 = cv_util.crop(src, rect2)

        # ガンマ補正
        crop1 = cv_util.gamma(crop1, r1["Gamma"])
        crop2 = cv_util.gamma(crop2, r2["Gamma"])
        # コントラスト補正
        crop1 = cv_util.contrast(crop1, r1["Contrast"])
        crop2 = cv_util.contrast(crop2, r2["Contrast"])
        # ぼかし
        blur_size1 = (r1["BlurSize"]["W"], r1["BlurSize"]["H"])
        blur_size2 = (r2["BlurSize"]["W"], r2["BlurSize"]["H"])
        crop1 = cv_util.blur(crop1, blur_size1)
        crop2 = cv_util.blur(crop2, blur_size2)
        # オープン
        morph_size1 = (r1["MorphSize"]["W"], r1["MorphSize"]["H"])
        morph_size2 = (r2["MorphSize"]["W"], r2["MorphSize"]["H"])
        crop1 = cv2.morphologyEx(crop1, cv2.MORPH_OPEN, morph_size1)
        crop2 = cv2.morphologyEx(crop2, cv2.MORPH_OPEN, morph_size2)
        
        if (draw):
            dst = cv_util.paste(dst, crop1, rect1)
            dst = cv_util.paste(dst, crop2, rect2)
            cv_util.drawRect(dst, rect1, COLOR_BRECT)
            cv_util.drawRect(dst, rect2, COLOR_BRECT)

        edge_l: list[int] = []
        edge_r: list[int] = []

        caliper1 = min(r1["EdgeCaliperNum"], rect1.h)
        step1 = int(rect1.h / caliper1)

        for y in range(0, rect1.h, step1):
            v = edge_search(crop1, r1["EdgeDiffRate"], y)
            if (v != None):
                # 見つかった
                edge_l.append(v + rect1.x)
                if (draw):
                    cv2.circle(dst, (v + rect1.x, y + rect1.y), 2, COLOR_EDGE, -1)
        if (len(edge_l) == 0):
            # 見つからなかった
            result.append(np.NaN)
            return result

        caliper2 = min(r2["EdgeCaliperNum"], rect2.h)
        step2 = int(rect2.h / caliper2)

        for y in range(0, rect2.h, step2):
            v = edge_search(crop2, r2["EdgeDiffRate"], y, True)
            if (v != None):
                # 見つかった
                edge_r.append(v + rect2.x)
                if (draw):
                    cv2.circle(dst, (v + rect2.x, y + rect2.y), 2, COLOR_EDGE, -1)
        if (len(edge_r) == 0):
            # 見つからなかった
            result.append(np.NaN)
            return result

        ll = rate_median(edge_l, r1["EdgeAdoptRate"])
        rr = rate_median(edge_r, r2["EdgeAdoptRate"])

        result.append(abs(rr - ll))

        return result

    def clip_henkei(self, src: cv2.Mat, dst: cv2.Mat, r, draw: bool) -> list[float]:
        result: list[float] = []

        r1 = r["Edge1"]
        rect = cv_util.Rect(r1["Rect"])
        
        crop = cv_util.crop(src, rect)
        crop = cv_util.gamma(crop, r1["Gamma"])
        crop = cv_util.contrast(crop, r1["Contrast"])
        blur_size = (r1["BlurSize"]["W"], r1["BlurSize"]["H"])
        crop = cv_util.blur(crop, blur_size)
        morph_size = (r1["MorphSize"]["W"], r1["MorphSize"]["H"])
        crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, morph_size)
        if (draw):
            dst = cv_util.paste(dst, crop, rect)
            cv_util.drawRect(dst, rect, COLOR_BRECT)

        edge_c: list[int] = []

        caliper = min(r1["EdgeCaliperNum"], rect.h)
        step = int(rect.h / caliper)

        for y in range(0, rect.h, step):
            # DS側の時は右から左
            v = edge_search(crop, r1["EdgeDiffRate"], y, (r["Mode"] == 3))
            if (v != None):
                # 見つかった
                edge_c.append(v)
                if (draw):
                    cv2.circle(dst, (v + rect.x, y + rect.y), 2, COLOR_EDGE, -1)
        if (len(edge_c) == 0):
            # 見つからなかった
            result.append(np.NaN)
            result.append(np.NaN)
            return result
        
        # オフセット量
        offset = int(rate_median(edge_c, r1["EdgeAdoptRate"]))
        result.append(offset)

        rL = r["Edge2"]
        rR = r["Edge3"]
        # 本来の矩形
        rect1 = cv_util.Rect(rL["Rect"], (offset, 0))
        rect2 = cv_util.Rect(rR["Rect"], (offset, 0))
            
        # クロッピング
        crop1 = cv_util.crop(src, rect1)
        crop2 = cv_util.crop(src, rect2)
        # ガンマ補正
        crop1 = cv_util.gamma(crop1, rL["Gamma"])
        crop2 = cv_util.gamma(crop2, rR["Gamma"])
        # コントラスト補正
        crop1 = cv_util.contrast(crop1, rL["Contrast"])
        crop2 = cv_util.contrast(crop2, rR["Contrast"])
        # ぼかし
        blur_size1 = (rL["BlurSize"]["W"], rL["BlurSize"]["H"])
        blur_size2 = (rR["BlurSize"]["W"], rR["BlurSize"]["H"])
        crop1 = cv_util.blur(crop1, blur_size1)
        crop2 = cv_util.blur(crop2, blur_size2)
        # オープン
        morph_size1 = (rL["MorphSize"]["W"], rL["MorphSize"]["H"])
        morph_size2 = (rR["MorphSize"]["W"], rR["MorphSize"]["H"])
        crop1 = cv2.morphologyEx(crop1, cv2.MORPH_OPEN, morph_size1)
        crop2 = cv2.morphologyEx(crop2, cv2.MORPH_OPEN, morph_size2)

        if (draw):
            dst = cv_util.paste(dst, crop1, rect1)
            dst = cv_util.paste(dst, crop2, rect2)
            cv_util.drawRect(dst, rect1, COLOR_BRECT)
            cv_util.drawRect(dst, rect2, COLOR_BRECT)

        edge_l: list[int] = []
        edge_r: list[int] = []

        caliper1 = min(rL["EdgeCaliperNum"], rect1.h)
        step1 = int(rect1.h / caliper1)

        for y in range(0, rect1.h, step1):
            v = edge_search(crop1, rL["EdgeDiffRate"], y)
            if (v != None):
                # 見つかった
                edge_l.append(v + rect1.x)
                if (draw):
                    cv2.circle(dst, (v + rect1.x, y + rect1.y), 2, COLOR_EDGE, -1)
        if (len(edge_l) == 0):
            # 見つからなかった
            result.append(np.NaN)
            return result

        caliper2 = min(rR["EdgeCaliperNum"], rect2.h)
        step2 = int(rect2.h / caliper2)

        for y in range(0, rect2.h, step2):
            v = edge_search(crop2, rR["EdgeDiffRate"], y, True)
            if (v != None):
                # 見つかった
                edge_r.append(v + rect2.x)
                if (draw):
                    cv2.circle(dst, (v + rect2.x, y + rect2.y), 2, COLOR_EDGE, -1)
        if (len(edge_r) == 0):
            # 見つからなかった
            result.append(np.NaN)
            return result

        ll = rate_median(edge_l, rL["EdgeAdoptRate"])
        rr = rate_median(edge_r, rR["EdgeAdoptRate"])

        result.append(abs(rr - ll))

        return result

# エッジ検索
def edge_search(src: cv2.Mat, diffrate: float, pin: int, r: bool = False):
    line = src[pin, :]
    end = len(line) - 1

    if (r):
        # 右側だったら右から左
        line = np.flip(line)

    for i, p in enumerate(line[0 : end]):
        diff = abs(float(line[i+1]) - float(line[i])) / 255.0
        if (diff > diffrate):
            return (end - i) if r else i

    return None

# 真ん中へんのrate%の平均
def rate_median(list: list[int], rate: float) -> float:
    num = len(list)
    start = int((num * 0.5) - (num * rate * 0.5))
    end = start + int(math.ceil(num * rate))
    return np.average(sorted(list)[start : end])




# 以下はテストコードなので、こんなかんじで静止画で動作確認してもよい
import recipe

if __name__ == "__main__":
    img_path = r"Q:\download\newplot (1).png"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    

    rcp_path = r"C:\P3I\recipe\テンター出クリップ痕OS.json"
    rcp = recipe.Recipe(rcp_path)
    if (rcp.recipe["IsCrop"]):
        cr = cv_util.Rect(rcp.recipe["CropRect"])
        w = cr.w
        h = cr.h
    
    proc = ImgProc(w, h, rcp.recipe["OffsetImagePath"])
    dst, result = proc.proc(img, rcp.recipe, True)
    print(result)
    cv2.imshow("image", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


