import re
import pandas as pd
from scipy.stats import entropy
import plotly.express as px
import plotly.io as pio
import os
import json
import platform

import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots


def set_japanese_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    """日本語フォントを設定する"""
    if platform.system() == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
    elif platform.system() == "Windows":
        font_path = r"C:\Windows\Fonts\meiryo.ttc"
    else:  # Linux or other
        font_path = "/usr/share/fonts/truetype/ipafont-gothic/ipagp.ttf"  # 適切なパスに変更してください

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
    party_colors_df["政党ID"] = party_colors_df["政党ID"].astype(str)
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


def visualize_question_details(df_results, df_entropy, party_colors_df, questions_json):
    """質問ごとに政党別回答割合とエントロピーのチャートを同じページに表示する"""
    os.makedirs("html", exist_ok=True)

    # 質問IDからタイトルと質問文をマッピング
    question_info = {}
    for q in questions_json:
        q_id = str(q["id"])
        if q_id not in question_info:
            question_info[q_id] = {"title": q["title"], "question": q["question"]}

    # 各質問ごとに処理
    for q_id in sorted(question_info.keys(), key=lambda x: int(x)):
        title = question_info[q_id]["title"]
        question_text = question_info[q_id]["question"]

        # 回答割合データの抽出
        df_resp = df_results[df_results["質問ID"] == q_id]

        # エントロピーデータの抽出
        df_ent = df_entropy[df_entropy["質問ID"] == q_id]

        # 回答割合のグラフ作成
        fig_resp = px.bar(
            df_resp,
            x="回答",
            y="割合",
            color="政党名",
            barmode="group",
            hover_data=["人数", "候補者数"],
            title=f"{title} における政党別回答割合",
            color_discrete_map=dict(
                zip(party_colors_df["政党名（略）"], party_colors_df["政党カラー"])
            ),
        )
        fig_resp.update_layout(
            xaxis_title="回答",
            yaxis_title="割合",
            yaxis_tickformat=".0%",
            legend_title="政党名",
            template="plotly_white",
        )

        # エントロピーのグラフ作成
        fig_ent = px.bar(
            df_ent,
            x="政党名",
            y="エントロピー",
            title=f"{title} における政党別回答のエントロピー",
            labels={"政党名": "政党名", "エントロピー": "エントロピー"},
            color="エントロピー",
            color_continuous_scale="Viridis",
        )
        fig_ent.update_layout(
            xaxis_title="政党名",
            yaxis_title="エントロピー",
            template="plotly_white",
        )

        # Plotlyの図をHTMLとして取得
        resp_div = pio.to_html(fig_resp, full_html=False, include_plotlyjs="cdn")
        ent_div = pio.to_html(
            fig_ent, full_html=False, include_plotlyjs=False
        )  # PlotlyJSは最初の図で読み込まれる

        # HTMLテンプレートを使用してページを生成
        combined_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>NHKの衆議院選挙2024各候補者アンケート集計結果 - {title}</title>
            <meta name="description" content="{question_text}">
            <meta name="keywords" content="NHK, 衆議院選挙, 2024, アンケート, 集計, 各候補者, 政党, クラスタリング">
            <meta property="og:title" content="NHKの衆議院選挙2024各候補者アンケート集計結果 - {title}">
            <meta property="og:description" content="{question_text}">
            <meta property="og:type" content="website">
            <meta property="og:url" content="{os.path.abspath(os.path.join('html', f'question_details_{q_id}.html'))}">
            <meta property="og:image" content="">
            <!-- Bootstrap CSS -->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body {{
                    padding-top: 20px;
                    padding-bottom: 20px;
                }}
                .container {{
                    max-width: 960px;
                }}
                .chart {{
                    margin-bottom: 40px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center">NHKの衆議院選挙2024各候補者アンケート集計結果</h1>
                <h2>{title}</h2>
                <p>{question_text}</p>
                
                <div class="chart">
                    {resp_div}
                </div>
                <div class="chart">
                    {ent_div}
                </div>
                
                <a href="index.html" class="btn btn-primary">戻る</a>
            </div>
        </body>
        </html>
        """

        # ファイルに保存
        filename = f"question_details_{q_id}.html"
        with open(os.path.join("html", filename), "w", encoding="utf-8") as f:
            f.write(combined_html)
        print(f"質問ID {q_id} の詳細ページを '{filename}' として生成しました。")


def cluster_candidates(df, party_colors_df, questions_json):
    """候補者の回答データをクラスタリングし、結果を可視化する"""

    # クラスタリングに使用する質問の列を特定
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    if not question_columns:
        print("クラスタリングに使用する質問列が見つかりません。")
        return

    # クラスタリング用データを抽出
    clustering_data = df[question_columns].copy()

    # 「回答なし（0）」も含めて数値化されていることを確認
    clustering_data = clustering_data.fillna(0)

    # データの標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # 主成分分析（PCA）で次元削減
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # 政党IDと政党名のマッピングを作成
    party_id_to_name = dict(zip(df["党派ID"].astype(str), df["党派名"]))

    fig = go.Figure()

    # 政党ごとにプロット
    for party_id, party_name in sorted(party_id_to_name.items(), key=lambda x: x[1]):
        party_data = df[df["党派ID"].astype(str) == party_id]

        if party_data.empty:
            continue

        try:
            party_color = party_colors_df.set_index("政党ID").loc[
                party_id, "政党カラー"
            ]
        except KeyError:
            party_color = "#000000"  # デフォルトカラー（黒）を設定

        hover_text = [
            f"氏名: {row['姓']} {row['名']}<br>"
            f"党派: {party_name}<br>"
            f"年齢: {row['年齢']}<br>"
            f"選挙区: {row['選挙区名']}"
            for _, row in party_data.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=party_data["PCA1"],
                y=party_data["PCA2"],
                mode="markers",
                name=party_name,
                marker=dict(
                    size=10,
                    color=party_color,
                    opacity=0.7,
                ),
                text=hover_text,
                hoverinfo="text",
                legendgroup=party_name,
                showlegend=True,
            )
        )

    # レイアウトの更新
    fig.update_layout(
        title="候補者のクラスタリング結果（PCAを使用）",
        xaxis_title="主成分1",
        yaxis_title="主成分2",
        template="plotly_white",
        legend_title="政党",
        legend=dict(
            groupclick="togglegroup",
            tracegroupgap=5,
        ),
    )

    # HTMLファイルとして保存
    fig.write_html(os.path.join("html", "interactive_candidate_clustering.html"))
    print(
        "クラスタリング結果の可視化が完了しました。'interactive_candidate_clustering.html' を確認してください。"
    )


def cluster_candidates_kmeans(df, party_colors_df, questions_json, num_clusters=10):
    """候補者の回答データをクラスタリングし、結果を可視化する"""
    # クラスタリングに使用する質問の列を特定
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    if not question_columns:
        print("クラスタリングに使用する質問列が見つかりません。")
        return

    # クラスタリング用データを抽出
    clustering_data = df[question_columns].copy()

    # 「回答なし（0）」も含めて数値化されていることを確認
    clustering_data = clustering_data.fillna(0)

    # データの標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # 主成分分析（PCA）で次元削減
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # K-Means クラスタリング
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    # 政党IDと政党名のマッピングを作成
    party_id_to_name = dict(zip(df["党派ID"].astype(str), df["党派名"]))

    fig = go.Figure()

    # クラスターごとにプロット
    for cluster in range(num_clusters):
        cluster_data = df[df["Cluster"] == cluster]

        hover_text = [
            f"氏名: {row['姓']} {row['名']}<br>"
            f"党派: {party_id_to_name[str(row['党派ID'])]}<br>"
            f"年齢: {row['年齢']}<br>"
            f"選挙区: {row['選挙区名']}"
            for _, row in cluster_data.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=cluster_data["PCA1"],
                y=cluster_data["PCA2"],
                mode="markers",
                name=f"クラスタ {cluster + 1}",
                marker=dict(
                    size=10,
                    color=cluster,
                    colorscale="Viridis",
                    opacity=0.7,
                ),
                text=hover_text,
                hoverinfo="text",
                legendgroup=f"クラスタ {cluster + 1}",
                showlegend=True,
            )
        )

    # レイアウトの更新
    fig.update_layout(
        title="候補者のクラスタリング結果（PCAを使用）",
        xaxis_title="主成分1",
        yaxis_title="主成分2",
        template="plotly_white",
        legend_title="クラスタ",
        legend=dict(
            groupclick="toggleitem",
            tracegroupgap=5,
        ),
    )

    # HTMLファイルとして保存
    fig.write_html(os.path.join("html", "interactive_candidate_clustering_kmeans.html"))
    print(
        "クラスタリング結果の可視化が完了しました。'interactive_candidate_clustering_kmeans.html' を確認してください。"
    )

    return df


def visualize_cluster_party_distribution(df, party_colors_df):
    """各クラスター内の政党分布を可視化する"""
    # クラスターごとの政党分布を計算
    cluster_party_distribution = (
        df.groupby(["Cluster", "党派名"]).size().unstack(fill_value=0)
    )

    # 割合に変換
    cluster_party_distribution = cluster_party_distribution.div(
        cluster_party_distribution.sum(axis=1), axis=0
    )

    # 政党カラーの辞書を作成
    party_colors = dict(
        zip(party_colors_df["政党名（略）"], party_colors_df["政党カラー"])
    )

    # プロットの作成
    fig = go.Figure()

    for party in cluster_party_distribution.columns:
        fig.add_trace(
            go.Bar(
                name=party,
                x=cluster_party_distribution.index,
                y=cluster_party_distribution[party],
                marker_color=party_colors.get(party, "#000000"),  # デフォルトカラーは黒
            )
        )

    fig.update_layout(
        title="各クラスター内の政党分布",
        xaxis_title="クラスター",
        yaxis_title="政党の割合",
        barmode="stack",
        legend_title="政党",
        template="plotly_white",
    )

    # HTMLファイルとして保存
    fig.write_html(os.path.join("html", "cluster_party_distribution.html"))
    print(
        "クラスター内の政党分布の可視化が完了しました。'cluster_party_distribution.html' を確認してください。"
    )


def cluster_candidates_kmeans2(df, party_colors_df, questions_json):
    """候補者の回答データをクラスタリングし、結果を可視化する"""

    # クラスタリングに使用する質問の列を特定
    question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

    if not question_columns:
        print("クラスタリングに使用する質問列が見つかりません。")
        return

    # クラスタリング用データを抽出
    clustering_data = df[question_columns].copy()

    # 「回答なし（0）」も含めて数値化されていることを確認
    clustering_data = clustering_data.fillna(0)

    # データの標準化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # 主成分分析（PCA）で次元削減
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # クラスタ数を政党数に設定
    unique_parties = df["党派名"].unique()
    num_clusters = len(unique_parties)
    print(f"クラスタ数を政党数に設定します: {num_clusters} クラスター")

    # K-Means クラスタリング
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    # 政党IDと政党名のマッピングを作成
    party_id_to_name = dict(zip(df["党派ID"].astype(str), df["党派名"]))

    # 政党ごとのクラスターをグループ化
    party_clusters = {}
    for cluster in range(num_clusters):
        cluster_data = df[df["Cluster"] == cluster]
        party_id = cluster_data["党派ID"].mode().iloc[0]  # 最頻値の政党IDを取得
        party_name = party_id_to_name[str(party_id)]
        if party_name not in party_clusters:
            party_clusters[party_name] = []
        party_clusters[party_name].append(cluster)

    # 政党名でソートしたクラスターリストを作成
    sorted_clusters = [
        cluster
        for party in sorted(party_clusters.keys())
        for cluster in party_clusters[party]
    ]

    fig = go.Figure()

    # クラスターごとにプロット（政党名順）
    for cluster in sorted_clusters:
        cluster_data = df[df["Cluster"] == cluster]
        party_id = cluster_data["党派ID"].mode().iloc[0]
        party_name = party_id_to_name[str(party_id)]

        try:
            party_color = party_colors_df.set_index("政党ID").loc[
                str(party_id), "政党カラー"
            ]
        except KeyError:
            party_color = "#000000"  # デフォルトカラー（黒）を設定

        hover_text = [
            f"氏名: {row['姓']} {row['名']}<br>"
            f"党派: {party_id_to_name[str(row['党派ID'])]}<br>"
            f"年齢: {row['年齢']}<br>"
            f"選挙区: {row['選挙区名']}"
            for _, row in cluster_data.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=cluster_data["PCA1"],
                y=cluster_data["PCA2"],
                mode="markers",
                name=f"{party_name} (クラスタ {cluster + 1})",
                marker=dict(
                    size=10,
                    color=party_color,
                    opacity=0.7,
                ),
                text=hover_text,
                hoverinfo="text",
                customdata=cluster_data["党派名"],
                legendgroup=party_name,
                showlegend=True,
            )
        )

    # レイアウトの更新
    fig.update_layout(
        title="候補者のクラスタリング結果（PCAを使用）",
        xaxis_title="主成分1",
        yaxis_title="主成分2",
        template="plotly_white",
        legend_title="政党とクラスタ",
        legend=dict(
            groupclick="toggleitem",
            tracegroupgap=5,
        ),
    )

    # HTMLファイルとして保存
    fig.write_html(os.path.join("html", "interactive_candidate_clustering_kmeans.html"))
    print(
        "クラスタリング結果の可視化が完了しました。'interactive_candidate_clustering_kmeans.html' を確認してください。"
    )


def compute_average_entropy(df_entropy):
    """政党別のエントロピーの平均を計算する"""
    average_entropy = df_entropy.groupby("政党名")["エントロピー"].mean().reset_index()
    average_entropy.columns = ["政党名", "平均エントロピー"]
    return average_entropy


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


def generate_index_html(df_results, df_entropy, questions_json, average_entropy):
    """index.html を生成し、各集計結果へのリンクと説明を含める"""
    os.makedirs("html", exist_ok=True)
    index_html_path = os.path.join("html", "index.html")

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>NHKの衆議院選挙2024各候補者アンケート集計結果</title>
        <meta name="description" content="NHKの衆議院選挙2024各候補者アンケートの集計結果を分析・可視化したページです。">
        <meta name="keywords" content="NHK, 衆議院選挙, 2024, アンケート, 集計, 各候補者, 政党, クラスタリング">
        <meta property="og:title" content="NHKの衆議院選挙2024各候補者アンケート集計結果">
        <meta property="og:description" content="NHKの衆議院選挙2024各候補者アンケートの集計結果を分析・可視化したページです。">
        <meta property="og:type" content="website">
        <meta property="og:url" content="{os.path.abspath(os.path.join('html', 'index.html'))}">
        <meta property="og:image" content="">
        <!-- Bootstrap CSS を使用してレスポンシブデザインを実現 -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{
                padding-top: 20px;
                padding-bottom: 20px;
            }}
            .container {{
                max-width: 960px;
            }}
            .card {{
                margin-bottom: 20px;
            }}
            @media (max-width: 576px) {{
                h1 {{
                    font-size: 1.5rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center">NHKの衆議院選挙2024各候補者アンケート集計結果</h1>
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
    """
    # 平均エントロピーの表示部分
    html_content += """
            <div class="card">
                <div class="card-header">
                    <h2>政党別エントロピーの平均</h2>
                </div>
                <div class="card-body">
                    <table class="table">
                        <thead>
                            <tr>
                                <th>政党名</th>
                                <th>平均エントロピー</th>
                            </tr>
                        </thead>
                        <tbody>
    """

    for _, row in average_entropy.iterrows():
        html_content += f"""
            <tr>
                <td>{row['政党名']}</td>
                <td>{row['平均エントロピー']:.4f}</td>
            </tr>
        """

    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <h2>質問ごとの回答割合とエントロピー</h2>
                </div>
                <div class="card-body">
                    <p>各質問に対して、政党ごとの回答割合と回答の多様性（エントロピー）を分析しました。以下の質問ごとに詳細な結果をご覧いただけます。</p>
                    <ul>
    """

    # 質問ごとのリンクを作成
    question_ids = df_results["質問ID"].unique()
    processed_questions = set()
    for q_id in sorted(question_ids, key=lambda x: int(x)):
        if q_id in processed_questions:
            continue
        processed_questions.add(q_id)
        question_title = df_results[df_results["質問ID"] == q_id]["質問タイトル"].iloc[
            0
        ]
        filename = f"question_details_{q_id}.html"
        html_content += f'        <li><a href="{filename}" class="nav-link">{question_title}</a></li>\n'

    # 「無所属」のリンクを追加
    html_content += f'        <li><a href="independent_details.html" class="nav-link">無所属の詳細</a></li>\n'

    html_content += """
                    </ul>
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    # ファイルに保存
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("index.html を生成しました。")


def generate_independent_details_page(df, party_colors_df, inverse_answer_mapping):
    """無所属の詳細ページを生成する"""
    os.makedirs("html", exist_ok=True)
    filename = "independent_details.html"
    filepath = os.path.join("html", filename)

    # 「無所属」データを抽出
    df_independent = df[df["党派名"] == "無所属"].copy()

    # デバッグ: データフレームの列名を確認
    print("データフレームの列名:", df_independent.columns.tolist())
    print("無所属の候補者数:", len(df_independent))

    if len(df_independent) == 0:
        # 無所属の候補者がいない場合のメッセージを生成
        message = "無所属の候補者はいません。"
        combined_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>無所属の詳細 - NHKの衆議院選挙2024アンケート集計結果</title>
            <!-- Bootstrap CSS -->
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <h1 class="text-center">無所属の詳細</h1>
                <p>{message}</p>
                <a href="index.html" class="btn btn-primary">戻る</a>
            </div>
        </body>
        </html>
        """
    else:
        # 対象となる質問列を取得
        question_columns = [col for col in df.columns if re.match(r"^\d+:", col)]

        # デバッグ: 質問列を確認
        print("質問列:", question_columns)

        # 回答割合を計算
        independent_results = []
        for col in question_columns:
            question_id = col.split(":")[0]
            question_title = (
                col.split(":", 1)[1] if ":" in col else col
            )  # 質問のタイトルを抽出
            answers = df_independent[col].fillna("回答なし")
            value_counts = answers.value_counts(normalize=True)
            for response_text, ratio in value_counts.items():
                independent_results.append(
                    {
                        "質問ID": question_id,
                        "質問タイトル": question_title,
                        "回答": response_text,
                        "割合": ratio,
                    }
                )

        df_independent_results = pd.DataFrame(independent_results)

        # デバッグ: 結果データフレームの列名を確認
        print("結果データフレームの列名:", df_independent_results.columns.tolist())
        print("結果データフレームの行数:", len(df_independent_results))

        if len(df_independent_results) == 0:
            # 回答データがない場合のメッセージを生成
            message = "無所属の候補者はいますが、回答データがありません。"
            combined_html = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <title>無所属の詳細 - NHKの衆議院選挙2024アンケート集計結果</title>
                <!-- Bootstrap CSS -->
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            </head>
            <body>
                <div class="container">
                    <h1 class="text-center">無所属の詳細</h1>
                    <p>{message}</p>
                    <a href="index.html" class="btn btn-primary">戻る</a>
                </div>
            </body>
            </html>
            """
        else:
            # グラフ作成（既存のコード）
            fig = px.bar(
                df_independent_results,
                x="質問ID",
                y="割合",
                color="回答",
                title="無所属における質問ごとの回答割合",
                hover_data=["質問タイトル"],
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(
                xaxis_title="質問ID",
                yaxis_title="割合",
                yaxis_tickformat=".0%",
                legend_title="回答",
                template="plotly_white",
            )
            fig.update_xaxes(tickangle=45)

            # Plotlyの図をHTMLとして取得
            fig_div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

            # HTMLテンプレートを使用してページを生成（既存のコード）
            combined_html = f"""
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <title>無所属の詳細 - NHKの衆議院選挙2024アンケート集計結果</title>
                <!-- Bootstrap CSS -->
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
                <style>
                    body {{
                        padding-top: 20px;
                        padding-bottom: 20px;
                    }}
                    .container {{
                        max-width: 960px;
                    }}
                    .chart {{
                        margin-bottom: 40px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="text-center">無所属の詳細</h1>
                    <p>こちらのページでは、無所属の候補者に対するアンケート集計結果をご覧いただけます。</p>
                    <div class="chart">
                        {fig_div}
                    </div>
                    <a href="index.html" class="btn btn-primary">戻る</a>
                </div>
            </body>
            </html>
            """

    # ファイルに保存
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(combined_html)
    print(f"無所属の詳細ページを '{filename}' として生成しました。")


def generate_cluster_details_page(df, party_colors_df, num_clusters):
    """クラスターごとの詳細ページを生成する"""
    os.makedirs("html", exist_ok=True)

    # 政党カラーの辞書を作成
    party_colors = dict(
        zip(party_colors_df["政党名（略）"], party_colors_df["政党カラー"])
    )

    for cluster in range(num_clusters):
        cluster_data = df[df["Cluster"] == cluster]

        # クラスター内の政党分布を計算
        party_distribution = cluster_data["党派名"].value_counts()
        party_distribution_percentage = party_distribution / len(cluster_data) * 100

        # 政党分布の円グラフを作成
        fig_pie = px.pie(
            values=party_distribution,
            names=party_distribution.index,
            title=f"クラスター {cluster + 1} の政党分布",
            color=party_distribution.index,
            color_discrete_map=party_colors,
        )

        # クラスター内の候補者の散布図を作成
        fig_scatter = px.scatter(
            cluster_data,
            x="PCA1",
            y="PCA2",
            color="党派名",
            hover_data=["姓", "名", "年齢", "選挙区名"],
            title=f"クラスター {cluster + 1} の候補者分布",
            color_discrete_map=party_colors,
        )

        # サブプロットを作成
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]]
        )
        fig.add_trace(fig_pie.data[0], row=1, col=1)
        fig.add_trace(fig_scatter.data[0], row=1, col=2)

        # レイアウトの更新
        fig.update_layout(
            title_text=f"クラスター {cluster + 1} の詳細", height=600, width=1200
        )

        # HTMLファイルとして保存
        fig.write_html(os.path.join("html", f"cluster_{cluster + 1}_details.html"))

        # 候補者リストのテーブルを作成
        candidate_table = cluster_data[
            ["姓", "名", "党派名", "年齢", "選挙区名"]
        ].to_html(index=False)

        # HTMLコンテンツの作成
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>クラスター {cluster + 1} の詳細</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <h1 class="text-center">クラスター {cluster + 1} の詳細</h1>
                <div id="charts">
                    <iframe src="cluster_{cluster + 1}_details.html" width="100%" height="620" frameborder="0"></iframe>
                </div>
                <h2>政党分布</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>政党</th>
                            <th>候補者数</th>
                            <th>割合</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"<tr><td>{party}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>" for party, count, percentage in zip(party_distribution.index, party_distribution, party_distribution_percentage))}
                    </tbody>
                </table>
                <h2>候補者リスト</h2>
                {candidate_table}
                <a href="index.html" class="btn btn-primary mt-3">戻る</a>
            </div>
        </body>
        </html>
        """

        # ファイルに保存
        with open(
            os.path.join("html", f"cluster_{cluster + 1}_details_full.html"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(html_content)

    print(f"{num_clusters}個のクラスター詳細ページを生成しました。")


def update_index_html_with_clusters(num_clusters):
    """index.htmlにクラスター詳細へのリンクを追加する"""
    index_path = os.path.join("html", "index.html")

    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    # クラスター詳細へのリンクを追加
    cluster_links = "\n".join(
        [
            f'<li><a href="cluster_{i + 1}_details_full.html">クラスター {i + 1} の詳細</a></li>'
            for i in range(num_clusters)
        ]
    )

    new_content = content.replace(
        "<!-- クラスター詳細リンク -->",
        f"""
        <div class="card mt-4">
            <div class="card-header">
                <h2>クラスター詳細</h2>
            </div>
            <div class="card-body">
                <p>各クラスターの詳細情報を確認できます：</p>
                <ul>
                    {cluster_links}
                </ul>
            </div>
        </div>
        <!-- クラスター詳細リンク -->
        """,
    )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("index.htmlにクラスター詳細へのリンクを追加しました。")


def main():
    # ... (既存のコード)

    # クラスタリング(kmeans)手法の実行と結果の取得
    num_clusters = 10
    df_clustered = cluster_candidates_kmeans(
        df, party_colors_df, questions_json, num_clusters=num_clusters
    )

    # クラスター内の政党分布を可視化
    visualize_cluster_party_distribution(df_clustered, party_colors_df)

    # クラスターごとの詳細ページを生成
    generate_cluster_details_page(df_clustered, party_colors_df, num_clusters)

    # index.htmlにクラスター詳細へのリンクを追加
    update_index_html_with_clusters(num_clusters)

    # ... (既存のコード)


def main():
    df = load_data("output.csv")

    # 政党カラーのデータを読み込む
    party_colors_df = load_party_colors("party_colors.csv")

    # 「無所属」がparty_colors_dfに含まれているか確認・追加
    if "無" not in party_colors_df["政党名（略）"].values:
        # 無所属のカラーを設定（例として灰色を使用）
        party_colors_df = party_colors_df.append(
            {"政党名（略）": "無", "政党カラー": "#808080"}, ignore_index=True  # グレー
        )
        print("party_colors_dfに「無所属」を追加しました。")

    # 質問と選択肢のデータをJSONから読み込み
    with open("questions.json", "r", encoding="utf-8") as f:
        questions_json = json.load(f)

    # 質問と選択肢をCSVに保存
    save_questions_to_csv(questions_json, "questions.csv")

    # データの前処理と逆マッピングの取得
    df, inverse_answer_mapping = preprocess_data(df, questions_json)

    # デバッグ: エントロピー計算前の回答分布を確認
    # debug_entropy_calculation(df, questions_json)

    # 政党ごとの回答割合を分析
    df_results = analyze_party_response_ratios(
        df, questions_json, inverse_answer_mapping
    )

    # 質問ごとの政党別回答のエントロピーを計算
    df_entropy = compute_answer_entropy(df, questions_json)

    # 政党別のエントロピーの平均を計算
    average_entropy = compute_average_entropy(df_entropy)

    # 回答割合とエントロピーを同一ページに可視化
    visualize_question_details(df_results, df_entropy, party_colors_df, questions_json)

    # クラスタリング手法の実行
    cluster_candidates(df, party_colors_df, questions_json)

    # クラスタリング(kmeans)手法の実行と結果の取得
    num_clusters = 10
    df_clustered = cluster_candidates_kmeans(
        df, party_colors_df, questions_json, num_clusters=num_clusters
    )

    # クラスター内の政党分布を可視化
    visualize_cluster_party_distribution(df_clustered, party_colors_df)

    # クラスターごとの詳細ページを生成
    generate_cluster_details_page(df_clustered, party_colors_df, num_clusters)

    # index.htmlにクラスター詳細へのリンクを追加
    update_index_html_with_clusters(num_clusters)

    # 「無所属」の詳細ページを生成（必要に応じて）
    generate_independent_details_page(df, party_colors_df, inverse_answer_mapping)

    # index.html を生成（新しい関数に渡す引数を修正）
    generate_index_html(df_results, df_entropy, questions_json, average_entropy)

    print(
        "すべての処理が完了しました。'html' フォルダ内の index.html をブラウザで開いてご覧ください。"
    )


if __name__ == "__main__":
    main()
