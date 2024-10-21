import re
import pandas as pd
import numpy as np
from scipy.stats import entropy
import plotly.express as px
import os
import json
import platform

from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def set_japanese_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    """日本語フォントを設定する"""
    if platform.system() == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
    elif platform.system() == "Windows":
        font_path = r"C:\Windows\Fonts\meiryo.ttc"
    else:  # Linux or other
        font_path = (
            "/usr/share/fonts/truetype/ipafont-gothic/ipagp.ttf"  # 適切なパスに変更してください
        )

    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = font_prop.get_name()
        print(f"フォントを設定しました: {font_prop.get_name()}")
    else:
        print(f"指定したフォントが見つかりません: {font_path}")
        print("デフォルトフォントを使用します。")


def load_data(file_path):
    """CSVファイルからデータを読み込む"""
    return pd.read_csv(file_path)


def load_party_colors(party_colors_file):
    """政党カラーのデータを読み込む"""
    party_colors_df = pd.read_csv(party_colors_file)
    return party_colors_df


def save_questions_to_csv(questions_json, output_file):
    """質問と選択肢のデータをCSVに保存する"""
    questions_data = []
    question_ids = set()
    for question in questions_json:
        q_id = str(question["id"])  # 質問IDを文字列型に変換
        if q_id in question_ids:
            print(f"重複した質問IDが見つかりました: {q_id}")
            continue  # 重複をスキップ
        question_ids.add(q_id)
        q_title = question["title"]
        q_question = question["question"]
        q_type = question["type"]
        selects = question["selects"]
        for idx, select in enumerate(selects):
            questions_data.append(
                {
                    "id": q_id,
                    "title": q_title,
                    "question": q_question,
                    "type": q_type,
                    "select_index": idx,
                    "select": select,
                }
            )
    df_questions = pd.DataFrame(questions_data)
    df_questions.to_csv(output_file, index=False)
    print(f"質問と選択肢のデータを {output_file} に保存しました。")


def preprocess_data(df, questions_json):
    """回答データを数値化する"""
    # 対象となる質問列を取得
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    # 質問IDと選択肢のマッピングを作成
    answer_mapping = {}
    inverse_answer_mapping = {}
    question_types = {}
    for question in questions_json:
        q_id = str(question["id"])
        q_type = question.get("type", 1)
        question_types[q_id] = q_type
        selects = question.get("selects", [])
        if q_type != 3 and selects:
            # 質問ごとにマッピングを保持
            answer_mapping[q_id] = {}
            inverse_answer_mapping[q_id] = {}
            for idx, select in enumerate(selects):
                # 選択肢を数値にマッピング（インデックス + 1）
                answer_mapping[q_id][select] = idx + 1
                inverse_answer_mapping[q_id][idx + 1] = select
            # 「回答なし」を特定の値にマッピング（例：0）
            answer_mapping[q_id]["回答なし"] = 0
            inverse_answer_mapping[q_id][0] = "回答なし"
        else:
            # 自由回答や選択肢がない場合
            answer_mapping[q_id] = {}
            inverse_answer_mapping[q_id] = {}

    # 質問ごとに処理
    for col in question_columns:
        q_id = col.split(":")[0]
        q_type = question_types.get(q_id, 1)
        if q_type != 3:
            # 質問ごとのマッピングを適用
            df[col] = df[col].map(answer_mapping[q_id])
            # 未知の回答や欠損値を0（「回答なし」）に置き換える
            df[col] = df[col].fillna(0)
        else:
            # 自由回答は除外
            df = df.drop(columns=[col])

    return df, inverse_answer_mapping  # 逆マッピングを返す


def cluster_candidates(df, party_colors_df, questions_json):
    """候補者の回答データをクラスタリングし、結果を可視化する"""
    # クラスタリングに使用する質問の列を特定
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    # クラスタリング用データを抽出
    clustering_data = df[question_columns].copy()

    # 「回答なし（0）」も含めて数値化されていることを確認
    clustering_data = clustering_data.fillna(0)

    # データの標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # 主成分分析（PCA）で次元削減
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # クラスタ数の決定（ここでは例としてクラスタ数を5に設定）
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    # クラスタリング結果の可視化
    fig = go.Figure()

    # クラスタごとにプロット
    for cluster in range(num_clusters):
        cluster_data = df[df["Cluster"] == cluster]
        # 候補者名の列を確認
        if "候補者名" in df.columns:
            candidate_text = cluster_data["候補者名"]
        elif "氏名" in df.columns:
            candidate_text = cluster_data["氏名"]
        elif "候補者" in df.columns:
            candidate_text = cluster_data["候補者"]
        else:
            candidate_text = None  # 候補者名がない場合

        fig.add_trace(
            go.Scatter(
                x=cluster_data["PCA1"],
                y=cluster_data["PCA2"],
                mode="markers",
                name=f"クラスタ {cluster + 1}",
                marker=dict(
                    size=10,
                    color=party_colors_df.set_index("政党名（略）").loc[
                        cluster_data["党派名"], "政党カラー"
                    ],
                    opacity=0.7,
                ),
                text=candidate_text,  # 候補者名が存在する場合のみ
            )
        )

    fig.update_layout(
        title="候補者のクラスタリング結果（PCAを使用）",
        xaxis_title="主成分1",
        yaxis_title="主成分2",
        template="plotly_white",
        legend_title="クラスタ",
    )

    # HTMLファイルとして保存
    fig.write_html(os.path.join("html", "interactive_candidate_clustering.html"))
    print("クラスタリング結果の可視化が完了しました。'interactive_candidate_clustering.html' を確認してください。")


def debug_entropy_calculation(df, questions_json):
    """エントロピー計算前に回答分布を確認するデバッグ関数"""
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]
    question_dict = {}
    for q in questions_json:
        q_id = str(q["id"])
        if q_id in question_dict:
            continue
        question_dict[q_id] = q["title"]

    for col in question_columns:
        q_id = col.split(":")[0]
        if q_id not in question_dict:
            continue
        question_title = question_dict[q_id]
        print(f"\n質問: {question_title}")
        for party_name, group in df.groupby("党派名"):
            answers = group[col].fillna(0).astype(int)
            value_counts = answers.value_counts(normalize=True, sort=False)
            print(f"政党: {party_name}")
            print(value_counts.to_dict())
            print("---")


def compute_answer_entropy(df, questions_json):
    """各質問に対して政党ごとの回答のエントロピーを計算する"""
    # 対象となる質問列を取得
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    # 質問IDと質問文のマッピングを作成（重複を除去）
    question_dict = {}
    for q in questions_json:
        q_id = str(q["id"])  # 質問IDを文字列型に変換
        if q_id in question_dict:
            continue  # 重複をスキップ
        question_dict[q_id] = q["title"]

    # 結果を格納するリスト
    results = []

    # 各質問ごとにエントロピーを計算
    for col in question_columns:
        q_id = col.split(":")[0]
        if q_id not in question_dict:
            continue
        question_title = question_dict[q_id]

        # 政党ごとにエントロピーを計算
        for party_name, group in df.groupby("党派名"):
            answers = group[col].fillna(0).astype(int)  # 0は「回答なし」を表す
            value_counts = answers.value_counts(normalize=True, sort=False)
            # 確率分布を取得
            probabilities = value_counts.values
            # エントロピーを計算（base=2）
            entropy_value = entropy(probabilities, base=2)
            results.append(
                {
                    "質問ID": q_id,
                    "質問タイトル": question_title,
                    "政党名": party_name,
                    "エントロピー": entropy_value,
                }
            )

    # 結果をデータフレームに変換
    df_entropy = pd.DataFrame(results)
    df_entropy.to_csv("answer_entropy.csv", index=False)
    print("質問ごとの政党別回答のエントロピーを answer_entropy.csv に保存しました。")

    return df_entropy


def visualize_question_party_entropy(df_entropy):
    """質問ごとの政党別回答のエントロピーを可視化する"""
    if df_entropy.empty:
        print("df_entropy が空です。エントロピーの可視化をスキップします。")
        return

    os.makedirs("html", exist_ok=True)

    # 質問ごとにエントロピーの棒グラフを作成
    question_ids = df_entropy["質問ID"].unique()
    for q_id in sorted(question_ids, key=lambda x: int(x)):
        df_question = df_entropy[df_entropy["質問ID"] == q_id]
        question_title = df_question["質問タイトル"].iloc[0]

        fig = px.bar(
            df_question,
            x="政党名",
            y="エントロピー",
            title=f"{question_title} における政党別回答のエントロピー",
            labels={"政党名": "政党名", "エントロピー": "エントロピー"},
            color="エントロピー",
            color_continuous_scale="Viridis",  # カラースケールを追加
            hover_data=["エントロピー"],
        )
        fig.update_layout(
            xaxis_title="政党名",
            yaxis_title="エントロピー",
            template="plotly_white",
        )

        # HTMLファイルとして保存
        filename = f"question_party_entropy_{q_id}.html"
        fig.write_html(os.path.join("html", filename))
    print("質問ごとの政党別回答のエントロピーの可視化が完了しました。")


def analyze_party_response_ratios(df, questions_json, inverse_answer_mapping):
    """すべての質問に対して各政党がどの割合で回答しているかを分析する"""
    # 質問IDと質問文のマッピングを作成（重複を除去）
    question_dict = {}
    for q in questions_json:
        q_id = str(q["id"])  # 質問IDを文字列型に変換
        if q_id in question_dict:
            print(f"重複した質問IDが見つかりました: {q_id}")
            continue  # 重複をスキップ
        question_dict[q_id] = q

    # 政党名のリスト
    parties = df["党派名"].unique()

    # 結果を格納するリスト
    results = []

    for q_id, question in question_dict.items():
        # 質問列名を特定
        question_column = None
        for col in df.columns:
            if col.startswith(f"{q_id}:"):
                question_column = col
                break
        if question_column is None:
            continue

        # 各政党ごとに回答を集計
        for party in parties:
            df_party = df[df["党派名"] == party]
            total_candidates = len(df_party)
            if total_candidates == 0:
                continue
            # 回答ごとに集計（欠損値も含む）
            response_counts = (
                df_party[question_column].fillna(0).astype(int).value_counts()
            )
            for response_value, count in response_counts.items():
                ratio = count / total_candidates
                # 数値の回答をテキストに変換（質問ごとのマッピングを使用）
                response_text = inverse_answer_mapping[q_id].get(response_value, "不明")
                results.append(
                    {
                        "質問ID": q_id,
                        "質問タイトル": question["title"],
                        "質問文": question["question"],
                        "政党名": party,
                        "回答": response_text,
                        "人数": count,
                        "候補者数": total_candidates,
                        "割合": ratio,
                    }
                )

    # 結果をデータフレームに変換
    df_results = pd.DataFrame(results)
    # 結果をCSVに保存
    df_results.to_csv("party_response_ratios.csv", index=False)
    print("政党ごとの回答割合を party_response_ratios.csv に保存しました。")

    return df_results


def visualize_party_response_ratios(df_results, party_colors_df):
    """回答割合を可視化する"""
    os.makedirs("html", exist_ok=True)

    # 質問ごとに可視化
    question_ids = df_results["質問ID"].unique()
    processed_questions = set()
    for q_id in sorted(question_ids, key=lambda x: int(x)):
        if q_id in processed_questions:
            continue
        processed_questions.add(q_id)
        df_question = df_results[df_results["質問ID"] == q_id]
        question_title = df_question["質問タイトル"].iloc[0]
        question_text = df_question["質問文"].iloc[0]

        # 政党カラーのマッピングを作成
        party_color_map = dict(zip(party_colors_df["政党名（略）"], party_colors_df["政党カラー"]))

        fig = px.bar(
            df_question,
            x="回答",
            y="割合",
            color="政党名",
            barmode="group",
            hover_data=["人数", "候補者数"],
            title=f"{question_title}",
            color_discrete_map=party_color_map,
        )
        fig.update_layout(
            xaxis_title="回答",
            yaxis_title="割合",
            yaxis_tickformat=".0%",
            legend_title="政党名",
            template="plotly_white",
        )

        # HTMLファイルとして保存
        filename = f"party_response_ratio_question_{q_id}.html"
        fig.write_html(os.path.join("html", filename))
    print("回答割合の可視化が完了しました。")


def generate_index_html(df_results, df_entropy):
    """index.html を生成し、各集計結果へのリンクと説明を含める"""
    os.makedirs("html", exist_ok=True)
    index_html_path = os.path.join("html", "index.html")

    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>選挙データ分析結果</title>
        <!-- Bootstrap CSS を使用してレスポンシブデザインを実現 -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                padding-top: 20px;
                padding-bottom: 20px;
            }
            .container {
                max-width: 960px;
            }
            .nav-link {
                color: #007bff;
            }
            .nav-link:hover {
                color: #0056b3;
            }
            .card {
                margin-bottom: 20px;
            }
            @media (max-width: 576px) {
                h1 {
                    font-size: 1.5rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">選挙データ分析結果</h1>
            <p>こちらのページでは、候補者の回答データを基にした様々な分析結果をご覧いただけます。</p>

            <div class="card">
                <div class="card-header">
                    <h2>クラスタリング分析</h2>
                </div>
                <div class="card-body">
                    <p>候補者の政策スタンスを基にクラスタリングを行い、類似したスタンスを持つ候補者同士をグループ化しました。これにより、政党を超えた政策的な共通点を持つ候補者を視覚的に理解できます。</p>
                    <p><a href="interactive_candidate_clustering.html" class="btn btn-primary">クラスタリング結果を見る</a></p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>質問ごとの政党別回答割合</h2>
                </div>
                <div class="card-body">
                    <p>各質問に対して、政党ごとにどのような回答をしているか、その割合を分析しました。以下の質問ごとに結果をご覧いただけます。</p>
                    <ul>
    """

    # 質問ごとのリンクを作成（重複を除去）
    question_ids = df_results["質問ID"].unique()
    processed_questions = set()
    for q_id in sorted(question_ids, key=lambda x: int(x)):
        if q_id in processed_questions:
            continue
        processed_questions.add(q_id)
        question_title = df_results[df_results["質問ID"] == q_id]["質問タイトル"].iloc[0]
        filename = f"party_response_ratio_question_{q_id}.html"
        html_content += f'        <li><a href="{filename}" class="nav-link">{question_title}</a></li>\n'

    html_content += """
                    </ul>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2>質問ごとの政党別回答のエントロピー</h2>
                </div>
                <div class="card-body">
                    <p>各質問に対して、政党ごとの回答の多様性をエントロピーとして表示します。以下の質問ごとに結果をご覧いただけます。</p>
                    <ul>
    """

    # 質問ごとのエントロピーグラフへのリンクを追加
    entropy_question_ids = df_entropy["質問ID"].unique()
    processed_questions = set()
    for q_id in sorted(entropy_question_ids, key=lambda x: int(x)):
        if q_id in processed_questions:
            continue
        processed_questions.add(q_id)
        question_title = df_entropy[df_entropy["質問ID"] == q_id]["質問タイトル"].iloc[0]
        filename = f"question_party_entropy_{q_id}.html"
        html_content += f'        <li><a href="{filename}" class="nav-link">{question_title}</a></li>\n'

    html_content += """
                    </ul>
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("index.html を生成しました。")


def main():
    # フォント設定（必要に応じて有効化）
    # set_japanese_font()

    df = load_data("output.csv")

    # 政党カラーのデータを読み込む
    party_colors_df = load_party_colors("party_colors.csv")

    # 質問と選択肢のデータをJSONから読み込み
    with open("questions.json", "r", encoding="utf-8") as f:
        questions_json = json.load(f)

    # 質問と選択肢をCSVに保存
    save_questions_to_csv(questions_json, "questions.csv")

    # データの前処理と逆マッピングの取得
    df, inverse_answer_mapping = preprocess_data(df, questions_json)

    # デバッグ: エントロピー計算前の回答分布を確認
    debug_entropy_calculation(df, questions_json)

    # 政党ごとの回答割合を分析
    df_results = analyze_party_response_ratios(
        df, questions_json, inverse_answer_mapping
    )

    # 回答割合を可視化
    visualize_party_response_ratios(df_results, party_colors_df)

    # **候補者のクラスタリングを実行**
    cluster_candidates(df, party_colors_df, questions_json)

    # 質問ごとの政党別回答のエントロピーを計算
    df_entropy = compute_answer_entropy(df, questions_json)

    # エントロピーを可視化
    visualize_question_party_entropy(df_entropy)

    # index.html を生成
    generate_index_html(df_results, df_entropy)

    print("すべての処理が完了しました。'html' フォルダ内の index.html をブラウザで開いてご覧ください。")


if __name__ == "__main__":
    main()
