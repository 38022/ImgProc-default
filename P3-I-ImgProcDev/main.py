import cv2

import recipe
import cv_util
import img_proc
import matplotlib.pyplot as plt

# NGは5/14の2:04～2:14
# OKはその24時間前（たぶん）
video_path = r"\\mars.pbo.konicaminolta.jp\public\media\20250527_横山_PM_L1動画\OK.mp4"
# recipe_path = r"\\mars.pbo.konicaminolta.jp\public\media\20250527_横山_PM_L1動画\T機入OSクリップ位置.json"
recipe_path = r"T機入OSクリップ位置.json"

# 動画読み込み
video = cv2.VideoCapture(video_path)
if (not video.isOpened):
    print("動画読み込みエラー")
    exit

# レシピ読み込み
rcp = recipe.Recipe(recipe_path)

# 画像処理クラスの設定
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (rcp.recipe["IsCrop"]):
    cr = cv_util.Rect(rcp.recipe["CropRect"])
    w = cr.w
    h = cr.h
proc = img_proc.ImgProc(w, h, rcp.recipe["OffsetImagePath"])

# 240秒経過したら自動でストップ
fps = video.get(cv2.CAP_PROP_FPS)
max_frames = int(fps * 600)  # 240秒分のフレーム数
frame_count = 0

results_list = []


while True:
    # 1フレーム読み出し
    success, img = video.read()
    if (not success or img.size == 0):
        # 読み出し失敗
        break
    else:
        # 読み出し成功
        # 画像処理
        dst, result = proc.proc(img, rcp.recipe, True)
        results_list.append(result)
        if (result != None):
            # 結果を画像上に出力
            for i, r in enumerate(result):
                hh = 60 + (30 * i)
                text = "{:.1f}".format(r)
                cv2.putText(dst, text, (10, hh), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            
        # ウィンドウ表示
        cv2.imshow("window name", dst)
        # この辺のコードを変えてキー入力でコマ送りにしたり
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count >= max_frames:
            break

time_axis = [i / fps for i in range(len(results_list))]
values = [result[0] for result in results_list]
plt.plot(time_axis, values, label="diff 1")
plt.xlabel("Time[sec]")
plt.ylabel("Result Value")
plt.title("frame diff")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

cv2.destroyWindow("window name")
video.release()
