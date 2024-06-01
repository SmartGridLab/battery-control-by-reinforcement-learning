## これはなに？
強化学習による蓄電池の最適な充放電を実現するためのコードです。  

## Gettering Started
### Installation / 導入
VSCodeのremote connectionを使ってdocker fileを読み込んでください。  

### Usage / 実行
[main.py](/Battery-Control-By-Reinforcement-Learning/main.py)を実行してください。  

## Others / その他
### Directroy / ディレクトリ構成
ディレクトリ構成は以下の２つを参考にしています：
1. The Hitchhiker's Guide to Python    
https://python-guideja.readthedocs.io/ja/latest/writing/structure.html    
2. Cookiecutter: Better Project Templates  
https://cookiecutter.readthedocs.io/en/latest/index.html  


Project Organization / プロジェクトの構成  
------------  
    ├── .devcontainer      <- docker関連ファイル
    │       ├── Dockerfile
    │       └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── .gihub             <- gothub関連ファイル
    │      └── workflows   <- pullreqされると走るもの
    │
    ├── Battery-Control-By-Reinforcement-Learning   <- Source code for use in this project.
    │       ├── __init__.py    <- Makes source codes a Python module
    │       │── main.py
    │       └── paramaters.py   <- 機械学習の調整パラメータを記述
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── tests              <- テストコード
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so the main code can be imported
    

### Information for the code / コードの図解

**クラス図：第4版**  
クラス図でわかること：pyファイル一つに1クラスが対応。クラス間の依存関係がわかる。
![Class_Diagram](https://github.com/SmartGridLab/battery-control-by-reinforcement-learning/assets/43132698/a31ae2c7-c4e2-4a38-b5e8-99effb74261f)

- Battery-Control-By-Reinforcement-Learning/Class_Diagram.pumlに記述しています
- Battery-Control-By-Reinforcement-Learning/out/class_diagram/Class_Diagram.pngで見れるようになっています
- 機能：
　.pyファイルを全て列挙  
　.csvファイルを全て列挙  
　ファイル関係をラベルと色で記述  
　　- import(ラベル：import)、subprocess(ラベル：use)→黒色  
　　- read_csv(ラベル：read)→緑色  
　　- to_csv(ラベル：create)→青色  
　　- pyファイルは白黒、csvファイルは緑、pdfファイルは赤  

今後はファイルを追加したらその都度変更できるようにしたいですが、毎回以下の設定が必要      
 javaのインストール  
 graphvizのインストール  
 PlantUMLのインストール  

**プログラム構成**  
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/d9158e4d-da82-469f-afc9-2c56ad89a311)

**ファイル名対応表**  
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/973445c6-0a90-44ee-b8ce-6ee51c32daae)

**スポット市場締切までの時系列**  
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/9c10e329-46ea-4e74-b875-e27ef819efff)

**リアルタイム制御時の時間帯ごとに使用するGPVファイル対応表**  
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/19ef0bce-2628-477c-8490-72c2d2f1248d)




