"""
election_data_processor.py

このスクリプトは、NHKの2024年衆議院選挙のデータをダウンロードし、JSONファイルをCSVに変換します。
JSONデータが既に存在する場合は、ダウンロードをスキップします。

使用方法:
    python election_data_processor.py

必要なライブラリ:
    - requests (インストール: pip install requests)
    - csv
    - json
    - os
    - sys

出力:
    - ダウンロードしたJSONファイルは 'downloaded_data' ディレクトリに保存されます。
    - 変換されたCSVファイルは 'output.csv' として出力されます。
"""

import requests
import csv
import json
import io
import os


def download_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content.decode("utf-8")
    else:
        raise Exception(f"Failed to download CSV from {url}")


def download_json(url, filename):
    if os.path.exists(filename):
        print(f"既に存在します。ダウンロードをスキップします: {filename}")
        return
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)
        print(f"ダウンロードしました: {filename}")
    else:
        print(f"ダウンロードに失敗しました: {url}")


def convert_json_to_csv(json_files, questions_file, output_csv):
    # 英語フィールド名と日本語フィールド名の対応辞書
    field_mapping = {
        "candidateID": "候補者ID",
        "lastname": "姓",
        "firstname": "名",
        "lastname_kana": "姓（かな）",
        "firstname_kana": "名（かな）",
        "sex": "性別",
        "touha": "党派ID",
        "touha_name": "党派名",
        "genshoku_flg": "現職フラグ",
        "genshoku_flg_name": "現職フラグ名",
        "rikkouhoji_tousennsu": "立候補時当選回数",
        "rikkouhoji_tousennsu_shu": "立候補時当選回数（衆）",
        "rikkouhoji_tousennsu_san": "立候補時当選回数（参）",
        "katagaki1": "肩書き1",
        "katagaki2": "肩書き2",
        "senkyoID": "選挙ID",
        "senkyo_name": "選挙名",
        "senkyo_touhyoubi": "選挙投票日",
        "age": "年齢",
        "senkyokuID": "選挙区ID",
        "senkyoku_name": "選挙区名",
        "todoufukenID": "都道府県ID",
        "todokede": "届出順",
        "suisen": "推薦",
        "shiji": "支持",
        "answerDatetime": "回答日時",
        # "qa" は個別に処理するのでここには含めない
    }

    # 質問の読み込み
    with open(questions_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # 全ての質問IDと対応する質問文を取得
    question_id_to_text = {}
    for pref_code, questions in questions_data.items():
        for question in questions:
            qid = question["id"]
            q_text = question["question"]
            question_id_to_text[qid] = q_text

    # CSVに書き出すフィールド名（日本語）
    csv_fieldnames = [
        "候補者ID",
        "姓",
        "名",
        "姓（かな）",
        "名（かな）",
        "性別",
        "党派ID",
        "党派名",
        "現職フラグ",
        "現職フラグ名",
        "立候補時当選回数",
        "立候補時当選回数（衆）",
        "立候補時当選回数（参）",
        "肩書き1",
        "肩書き2",
        "選挙ID",
        "選挙名",
        "選挙投票日",
        "年齢",
        "選挙区ID",
        "選挙区名",
        "都道府県ID",
        "届出順",
        "推薦",
        "支持",
        "回答日時",
        # 質問IDごとの回答を列として追加
    ]

    # 質問IDのリストを取得（全候補者の質問を網羅）
    question_ids = set()
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for candidate in data:
                for qa in candidate.get("qa", []):
                    question_ids.add(qa["id"])

    # 質問IDをソートするためのキー関数を定義
    def question_id_key(qid):
        parts = qid.split("_")
        key = []
        for part in parts:
            # 数値部分は整数に変換、そうでなければそのまま文字列
            key.append(int(part) if part.isdigit() else part)
        return key

    # キー関数を使用して質問IDをソート
    question_ids = sorted(question_ids, key=question_id_key)

    # 質問IDを日本語の列名として追加
    for qid in question_ids:
        question_text = question_id_to_text.get(qid, f"質問{qid}")
        csv_fieldnames.append(f"{qid}:{question_text}")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                candidates = json.load(f)
                for candidate in candidates:
                    row = {}
                    for en_field, ja_field in field_mapping.items():
                        row[ja_field] = candidate.get(en_field, "")

                    # 都道府県IDはリストなので、カンマ区切りの文字列に変換
                    if isinstance(candidate.get("todoufukenID"), list):
                        row["都道府県ID"] = ",".join(
                            map(str, candidate.get("todoufukenID"))
                        )

                    # qa内の回答を処理
                    for qa in candidate.get("qa", []):
                        qid = qa["id"]
                        answers = qa.get("answers", [])
                        # 回答を"|"で結合して文字列にする
                        answer_str = "|".join(answers)
                        question_text = question_id_to_text.get(qid, f"質問{qid}")
                        column_name = f"{qid}:{question_text}"
                        row[column_name] = answer_str  # 回答を行に追加

                    writer.writerow(row)

    print(f"CSVファイルを出力しました: {output_csv}")


def main():
    csv_url = "https://www3.nhk.or.jp/senkyo-data/database/shugiin/2024/00/search/sindex.csv?1729421136274"
    base_url = "https://www3.nhk.or.jp/senkyo-data/database/shugiin/2024/survey/"

    # ダウンロード用のディレクトリを作成
    os.makedirs("downloaded_data", exist_ok=True)

    # CSVファイルをダウンロード
    csv_content = download_csv(csv_url)

    # CSVデータを読み込む
    csv_reader = csv.reader(io.StringIO(csv_content))
    next(csv_reader)  # ヘッダーをスキップ

    # 最初の行から election_id を取得し、questionsをダウンロード
    first_row = next(csv_reader)
    election_id = first_row[1]
    questions_url = f"{base_url}{election_id}questions.json"  # 質問データ
    questions_filename = os.path.join("downloaded_data", f"{election_id}questions.json")
    download_json(questions_url, questions_filename)

    # CSVリーダーを最初から再初期化
    csv_reader = csv.reader(io.StringIO(csv_content))
    next(csv_reader)  # ヘッダーをスキップ

    json_files = []
    for row in csv_reader:
        district_id = row[3]

        # 選挙区ごとのJSONをダウンロード
        url = f"{base_url}{district_id}.json"  # 回答データ
        filename = os.path.join("downloaded_data", f"{district_id}.json")
        download_json(url, filename)
        json_files.append(filename)

    # JSONファイルをCSVに変換
    output_csv = "output.csv"
    convert_json_to_csv(json_files, questions_filename, output_csv)


if __name__ == "__main__":
    main()
