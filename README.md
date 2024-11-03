# shugiin2024
衆議院選挙NHK候補者アンケートデータと集計


## ページのURL
Pages: [https://jfk.github.io/shugiin2024/](https://jfk.github.io/shugiin2024/)

## 準備

### 仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate
```

### ライブラリのインストール
```bash
(venv) pip install -r requirements.txt
```

## データの取得
```bash
(venv) python election_data_processor.py
```

## データの集計
```bash
(venv) python election_data_to_html.py
```
