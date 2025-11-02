import pandas as pd
import numpy as np
import json
import os
import argparse
import math
from tqdm import tqdm
import yaml # config.yaml を読み込むため

def print_log(message):
    """タイムスタンプ付きでログを出力する"""
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_data(output_dir):
    """ベクトルデータとメタデータを読み込む"""
    print_log(f"データ読み込みを開始します。出力ディレクトリ: {output_dir}")
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print_log(f"エラー: メタデータファイルが見つかりません: {metadata_path}")
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    df_metadata = pd.DataFrame(metadata)
    print_log(f"メタデータ ({len(df_metadata)}件) を読み込みました。")

    return df_metadata

def generate_question_list_html(category_df):
    """指定されたDataFrameの問題リストHTMLスニペットを生成する"""
    html_list = ""
    for index, row in category_df.iterrows():
        html_list += f"""
            <li class="question-item">
                <div class="question-id">問題番号: {row['問題番号']}</div>
                <a href="{row['リンク']}" target="_blank">{row['問題名']}</a>
                <div class="source">出典: {row['出典']}</div>
            </li>
        """
    return html_list

def main():
    parser = argparse.ArgumentParser(description='ベクトルデータとメタデータから統合HTMLレポートを生成します。')
    parser.add_argument('--output_dir', type=str, default='../02_vectorize/output', help='ベクトルデータとメタデータがあるディレクトリ')
    parser.add_argument('--html_output_root_dir', type=str, default='html_reports', help='生成されるHTMLファイルの出力先ディレクトリ')
    parser.add_argument('--sort_by_similarity', action='store_true', help='分野内でベクトル類似度に基づいて問題をソートします (未実装)。')
    args = parser.parse_args()

    print_log("HTMLレポート生成処理を開始します。")

    # 02_vectorize の config.yaml を読み込む
    config_path = '../02_vectorize/config.yaml'
    if not os.path.exists(config_path):
        print_log(f"エラー: 設定ファイルが見つかりません: {config_path}")
        return
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    all_models_config = config['models'] # config.yaml に記載されている全モデル

    df_metadata = load_data(args.output_dir)
    if df_metadata is None:
        return

    # HTMLレポート全体のルートディレクトリを作成
    os.makedirs(args.html_output_root_dir, exist_ok=True)
    report_file_path = os.path.join(args.html_output_root_dir, "report.html")

    # HTMLコンテンツを構築
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>基本情報技術者試験 類似問題レポート</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-bottom: 1px dashed #ccc; padding-bottom: 5px; } /* モデル名 */
            h3 { color: #555; margin-top: 20px; } /* 分野名 */
            .tabs { display: flex; flex-wrap: wrap; margin-bottom: 20px; border-bottom: 2px solid #ddd; } /* タブコンテナ */
            .tab-button { padding: 10px 20px; cursor: pointer; border: 1px solid #ddd; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; background-color: #f0f0f0; margin-right: 5px; transition: background-color 0.3s, color 0.3s; } /* タブボタン */
            .tab-button.active { background-color: #3498db; color: white; border-color: #3498db; } /* アクティブなタブボタン */
            .tab-content { display: none; padding: 20px 0; border-top: 2px solid #3498db; } /* タブコンテンツ */
            .tab-content.active { display: block; } /* アクティブなタブコンテンツ */
            .question-list { list-style: none; padding: 0; } /* 問題リスト */
            .question-item { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 15px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); } /* 問題アイテム */
            .question-item a { text-decoration: none; color: #3498db; font-weight: bold; } /* 問題名リンク */
            .question-item a:hover { text-decoration: underline; } /* 問題名リンクホバー */
            .question-id { font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px; } /* 問題ID */
            .source { font-size: 0.8em; color: #95a5a6; margin-top: 5px; } /* 出典 */
        </style>
        <script>
            function openModel(evt, modelId) {
                var i, tabcontent, tabbuttons;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tabbuttons = document.getElementsByClassName("tab-button");
                for (i = 0; i < tabbuttons.length; i++) {
                    tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                }
                document.getElementById(modelId).style.display = "block";
                evt.currentTarget.className += " active";
            }
            document.addEventListener("DOMContentLoaded", function() {
                // ページロード時に最初のタブをアクティブにする
                var firstTabButton = document.querySelector(".tab-button");
                if (firstTabButton) {
                    firstTabButton.click();
                }
            });
        </script>
    </head>
    <body>
        <h1>基本情報技術者試験 類似問題レポート</h1>
        <p>このレポートでは、各AIモデルでベクトル化された問題の分野別リストを統合して表示しています。</p>
        
        <div class="tabs">