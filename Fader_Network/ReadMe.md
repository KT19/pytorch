# Fader-Network Implementation

## simpleなfader-networkの実装

元論文のものよりモデルのサイズも小さく、また、アーキテクチャの構造や学習の仕方なども若干異なっていますが、最低限の動作を確認することができます。

実際にどのように変化するかは関数eval内で、属性(attribute)を指定することで確認することができます。

学習にはcelebAデータセットが必要です。
dataフォルダを作成して、その中にcelebAの画像フォルダ(img_celebA)および画像と属性の対応を示したテキスト(list_attr_celeba.txt)を置いて学習をさせてください。

モデルの層の深さやフィルタの枚数などのハイパーパラメータの設定はソースコード(module.py)を直接書き換えることで行うことができます。学習はtrain.pyに記載されています。また、utils.pyに画像のサイズなどの定義が記載されています。

学習の際はmain.pyにtrain()と記入して
***
python3 main.py
***
で実行できます。

実際に評価を行う際はmain.pyにeval()と記入して
***
python3 main.py
***
で実行できます。
