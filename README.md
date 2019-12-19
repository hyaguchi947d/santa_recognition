# santa_recognition

## 試した環境

Ubuntu 18.04 LTS, OpenCV 3.2, Python2, sklearn

## 準備

以下のような配置でデータセットを配置してください。

- train
  - p
    - <train positive images>
  - n
    - <train negative images>
- test
  - p
    - <test positive images>
  - n
    - <test negative images>

また、本文中で三枚の画像を最終評価目標として設定しています。

- dr_hiro_avator_00.png
- avatar_santa.png
- santa_kamosirenai.png

これらは適宜変更してください。

## 使い方

### 色ヒストグラム

```
$ python colorhist.py
```

### Bag of Visual Words (色情報なし)

```
$ python bag_of_visual_words.py
```

### Bag of Visual Words (色情報あり)

```
$ python bag_of_visual_words_color.py
```
