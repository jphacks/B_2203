# **SUGAO**(**S**aving **U**nmasking face **G**enerated by **A**I **O**peration)

[![IMAGE ALT TEXT HERE](https://jphacks.com/wp-content/uploads/2022/08/JPHACKS2022_ogp.jpg)](https://www.youtube.com/watch?v=LUPQFB4QyVo)

## 製品概要

### 背景(製品開発のきっかけ、課題等）

<!--担当：中野-->

皆様，一度はこのように思った経験ありませんか？

「あれ，この人のマスクを外したときの顔ってこんな感じなんだ」

コロナ禍でマスクの着用が定着し，マスク顔がデフォルトとなっているこの時代，マスクを外したときの顔に新鮮さを覚える人は少なくないと思います．マスクによって口元の表情が読み取りにくくなり，感染防止のために写真を撮る機会も減少しました．

私たちはこの課題に対して，AI 技術を活用し，「日常からマスクを払拭できるようなアプリを開発したい！！」と考えました．

具体的には，写真の中からマスクをつけた人の顔を検出し，画像生成系 AI である Cycle GAN を用いてマスクを外したときの画像を保存できるようなアプリを作成しました．

このアプリケーションによって写真を撮る機会を促進し，『新たな思い出の 1 ページを！！』作って頂ければ幸いです．

### 製品説明（具体的な製品の説明）

<!--担当：三好-->

本製品は、フェイスマスク着用時の写真撮影を支援する目的とした、フェイスマスクを撮影された写真から除去（素顔の予測）を行う Web アプリケーションである。

具体的には、インターネット上に公開された Web ページにデバイス（スマートフォン・パソコンなど）からアクセスし、フェイスマスクを着用した写真を選択する。選択された画像は事前処理を行い画像生成系 AI"CycleGAN"によってフェイスマスク部分の素顔を予測し、予測された画像（フェイスマスクが除去された画像）が出力される。また、出力された結果画像を保存することも可能である。

#### システムフローチャート

<!--ここは削ってもよいかも-->

![flowchart](/figs/flowchart.svg)

#### システム概要

本システムの使用手順は以下の 6 つ工程を順に行っている。<br>
① デバイス（スマートフォン・パソコンなど）から指定の Web ページにアクセス<br>
② デバイス内に保存された画像データを選択<br>
③ 選択した画像データを Web を経由してサーバ側に送信<br>
④ サーバ側で受信された画像データを事前処理・学習済みモデルを使用し、フェイスマスクを部分の素顔を予測<br>
⑤ 予測された画像データを Web ページ上に結果画像として表示<br>
⑥ 表示された画像を任意で保存(保存機能あり)<br>

![System](/figs/system_img.svg)

### 特長

<!--担当：森下-->

#### 1. マスクを外さずにマスクを外した写真を撮れる

#### 2. スマホで撮影した写真をその場で変換できる

#### 3. Web アプリなのでインストールの必要がない

### 解決出来ること

- 写真を撮る機会を促進できる
- 日常の写真からマスクを取り除くことができる

### 今後の展望

### 注力したこと（こだわり等）

#### 1. Cycle GAN を学習させるために必要なマスクをつけた人の画像集め

課題として，マスクを着けていな人の画像データを収取することが困難であった．
そこで，Cycle GAN が双方向変換(mask→nonmask, nonmask→mask)であることを利用した．数百枚程度の少ないデータ学習させた nonmask→mask モデルの GAN を用いることでマスクを着けていない人の画像から，マスクをつけた人の画像データを 5,000 枚程度入手することに成功した．

#### 2. 限られた時間における効率的な CycleGAN の学習方法の開発

CycleGAN が画像の変換を学習する過程で，生成された画像から顔の輪郭推定を行い，その結果を重みとして Loss に乗算することで，顔の輪郭生成に特化したモデルを作成した．これにより数千枚程度の画像データで顔の輪郭を生成できるモデルの学習が可能となった．

#### 3.ウォーターフォール開発による個性を活かした開発

本システム開発ではウォーターフォール開発を採用しており、チームメンバの個々の専門性を活かした役割分担を行うことで開発効率の向上を図った。

## 開発技術

### 活用した技術

<!--担当：全員-->

#### API・データ

- CelebA(人の顔画像のデータセット)
-

#### フレームワーク・ライブラリ・モジュール

<!--担当：森下-->

##### フロントエンド

- HTML
- CSS

<!--担当：三好-->

##### バックエンド

- Python
- Flask
- ngork

<!--担当：中野-->

##### 学習モデル

- Dlib
- OpenCV
- Pytorch
- Cycle GAN
-

#### デバイス

- PC
- スマートフォン
- タブレット

### 独自技術

#### ハッカソンで開発した独自機能・技術

- 独自で開発したものの内容をこちらに記載してください
- 特に力を入れた部分をファイルリンク、または commit_id を記載してください。

#### 製品に取り入れた研究内容（データ・ソフトウェアなど）（※アカデミック部門の場合のみ提出必須）

-
-
