#recipe.py
import json
import os
import pathlib
import datetime

class Recipe(object):
    def __init__(self, path: str):
        self.path = path
        p = pathlib.Path(path)
        st = p.stat()
        self.dt = datetime.date.fromtimestamp(st.st_mtime)

        with open(path, "r", encoding = "utf-8") as file:
            self.recipe = json.load(file)

        self.change_offset_img_path()
    
    # テンプレート画像のパスを修正
    def change_offset_img_path(self):
        path = self.recipe["OffsetImagePath"]
        if (path != None):
            # TODO: nullじゃなくて""のときもダメ
            dir = os.path.dirname(self.path)
            self.recipe["OffsetImagePath"] = os.path.join(dir, path)




