## これはなに？
強化学習による蓄電池の最適な充放電を実現するためのコードです。  

------------  
2023年10月から IEEE PES のコンペ用に開発しています。Projectはこちら：[Hybrid energy forecasting and trading competition](https://github.com/users/daisukekodaira/projects/4/views/1)   

■コンペの概要  
主催：IEEE Power & Energy Society Working Group  
　　　（電力システムの分野で一番大きな学会のworking groupが主催）  
賞金：2100ドル、320万円  
何をするのか？：発電量の確率的予測と電力取引（インバランス取引）のアルゴリズムを開発する  
URL：[LinkedInのリンク](https://www.linkedin.com/posts/rbessa_are-you-a-data-scientist-are-you-interested-activity-7098248306677460992-5DO8?utm_source=share&utm_medium=member_desktop)

■情報  
[Google Drive](https://drive.google.com/drive/folders/1hwIEnC7d4CLOBH7XhkvrMrnH0HQpmEEG?usp=sharing)に情報を集約しています  

■スケジュール  
2023/11/01：IEEE DataPortを通じて登録が開始
2023/11/01：開発およびテストのためのコンペティションプラットフォームがOpen
2023/11/31：PV,wind予測、価格予測、強化学習モデルの結合テスト(1回目)
2023/12/15：API経由での送信テスト(結合テスト2回目)
2024/01/31：コンペティション期間の最初の提出（2024年2月1日の予測および入札）
2024/04/29：コンペティション期間の最後の提出
2024/05/20：最終リーダーボードおよび賞の発表

■作業内容  
　web上でデータが提供される  
　↓  
　自分たちのアルゴリズムによる予測と取引の決定  
　↓  
　APIによって自動的に送信される（APIの実装はサポートを受けられる）  

■作業の分担  
　1. 取りまとめ：小平  
　2. 電力価格予測:大曽根  
　3. PV発電量予測:野村  
　3. 強化学習モデル：後藤  
　4. インバランス設計：小平  

■進め方  
全員が集まる定期MTを開催（週に１度　水13:00-14:00）  
小平との個別のMTは都度  

------------

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

プログラム構成
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/d9158e4d-da82-469f-afc9-2c56ad89a311)

ファイル名対応表
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/973445c6-0a90-44ee-b8ce-6ee51c32daae)

スポット市場締切までの時系列
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/9c10e329-46ea-4e74-b875-e27ef819efff)

リアルタイム制御時の時間帯ごとに使用するGPVファイル対応表
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/19ef0bce-2628-477c-8490-72c2d2f1248d)




