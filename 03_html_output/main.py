import pandas as pd
import numpy as np
import json
import os
import argparse
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm

def print_log(message):
    """タイムスタンプ付きでログを出力する"""
    print(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_config(config_path='../02_vectorize/config.yaml'):
    """設定ファイルを読み込む"""
    print_log(f"設定ファイル '{config_path}' の読み込みを開始します...")
    if not os.path.exists(config_path):
        print_log(f"エラー: 設定ファイルが見つかりません: {config_path}")
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print_log("設定ファイルの読み込みが完了しました。")
    return config

def load_vectorized_csv(csv_path):
    """ベクトル化済みCSVを読み込む"""
    print_log(f"CSVファイル '{csv_path}' の読み込みを開始します...")
    if not os.path.exists(csv_path):
        print_log(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print_log(f"CSVファイルの読み込みが完了しました。({len(df)}行)")
    return df

def get_vector_column_name(model_name):
    """モデル名からベクトル列名を生成"""
    return f"vector_{model_name.replace('/', '_').replace('.', '_').replace(':', '_')}"

def extract_vector(value):
    """文字列からベクトルを抽出"""
    if pd.isna(value) or value == '' or value == 'None' or value == '[]':
        return None
    try:
        vector = json.loads(value)
        if isinstance(vector, list) and len(vector) > 0:
            return np.array(vector)
    except (json.JSONDecodeError, ValueError):
        pass
    return None

def calculate_similarities(df, vector_column):
    """各問題間の類似度を計算"""
    print_log(f"列 '{vector_column}' の類似度を計算しています...")
    
    # ベクトルを抽出
    vectors = []
    valid_indices = []
    for idx, row in df.iterrows():
        vec = extract_vector(row[vector_column])
        if vec is not None:
            vectors.append(vec)
            valid_indices.append(idx)
    
    if len(vectors) == 0:
        print_log(f"エラー: 有効なベクトルが見つかりませんでした。")
        return None
    
    vectors = np.array(vectors)
    print_log(f"有効なベクトル {len(vectors)}件を抽出しました。")
    
    # コサイン類似度を計算
    similarities = cosine_similarity(vectors)
    
    return similarities, valid_indices

def group_by_middle_category(df, vector_column):
    """中項目ごとにグループ化し、各問題の類似問題を特定"""
    print_log("中項目ごとにグループ化しています...")
    
    # 中項目でグループ化
    grouped = defaultdict(list)
    for idx, row in df.iterrows():
        middle_cat = row['中項目']
        vec = extract_vector(row[vector_column])
        if vec is not None:
            grouped[middle_cat].append({
                'index': idx,
                'vector': vec,
                'data': row.to_dict()
            })
    
    print_log(f"{len(grouped)}個の中項目を検出しました。")
    
    # 各中項目内で類似度を計算
    category_similarities = {}
    for middle_cat, items in tqdm(grouped.items(), desc="類似度計算中"):
        if len(items) < 2:
            continue
        
        vectors = np.array([item['vector'] for item in items])
        similarities = cosine_similarity(vectors)
        
        # 各問題について、類似度が高い順に並べる
        similar_problems = []
        for i, item in enumerate(items):
            # 自分自身を除外して類似度順にソート
            sim_scores = [(j, similarities[i][j]) for j in range(len(items)) if i != j]
            sim_scores.sort(key=lambda x: x[1], reverse=True)
            
            similar_list = []
            for j, score in sim_scores:
                similar_list.append({
                    'data': items[j]['data'],
                    'similarity': float(score)
                })
            
            similar_problems.append({
                'main_problem': item['data'],
                'similar_problems': similar_list
            })
        
        category_similarities[middle_cat] = similar_problems
    
    return grouped, category_similarities

def generate_index_page(categories, output_dir, model_name):
    """インデックスページを生成"""
    print_log("インデックスページを生成しています...")
    
    # 大項目でグループ化
    major_categories = defaultdict(list)
    for middle_cat in sorted(categories.keys()):
        # 中項目から大項目を取得（最初の問題データから）
        if categories[middle_cat]:
            major_cat = categories[middle_cat][0]['main_problem']['大項目']
            major_categories[major_cat].append(middle_cat)
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>基本情報技術者試験 類似問題ファインダー</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Hiragino Sans', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .model-info {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
            color: #666;
        }}
        .major-category {{
            border-bottom: 1px solid #e0e0e0;
        }}
        .major-title {{
            background: #f8f9fa;
            padding: 15px 20px;
            font-weight: bold;
            color: #333;
            font-size: 16px;
        }}
        .middle-category-list {{
            list-style: none;
        }}
        .middle-category-item {{
            border-bottom: 1px solid #f0f0f0;
        }}
        .middle-category-item:last-child {{
            border-bottom: none;
        }}
        .middle-category-link {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            text-decoration: none;
            color: #333;
            transition: background 0.2s;
        }}
        .middle-category-link:hover {{
            background: #f8f9fa;
        }}
        .middle-category-link:active {{
            background: #e9ecef;
        }}
        .category-name {{
            font-size: 15px;
        }}
        .problem-count {{
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
        }}
        .arrow {{
            color: #999;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 類似問題ファインダー</h1>
            <p>基本情報技術者試験</p>
        </div>
        <div class="model-info">
            使用モデル: {model_name}
        </div>
"""
    
    for major_cat in sorted(major_categories.keys()):
        html += f"""
        <div class="major-category">
            <div class="major-title">{major_cat}</div>
            <ul class="middle-category-list">
"""
        for middle_cat in sorted(major_categories[major_cat]):
            count = len(categories[middle_cat])
            safe_filename = middle_cat.replace('/', '_').replace('\\', '_').replace(':', '_')
            html += f"""
                <li class="middle-category-item">
                    <a href="pages/{safe_filename}.html" class="middle-category-link">
                        <span class="category-name">{middle_cat}</span>
                        <div>
                            <span class="problem-count">{count}問</span>
                            <span class="arrow">›</span>
                        </div>
                    </a>
                </li>
"""
        html += """
            </ul>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    
    print_log("インデックスページを保存しました。")

def generate_category_page(middle_cat, problems, output_dir):
    """中項目ごとの詳細ページを生成"""
    safe_filename = middle_cat.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{middle_cat} - 類似問題</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Hiragino Sans', sans-serif;
            background: #f5f5f5;
            padding-bottom: 60px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .back-button {{
            display: inline-block;
            color: white;
            text-decoration: none;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .header h1 {{
            font-size: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .problem-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .main-problem {{
            border-left: 4px solid #667eea;
            padding-left: 16px;
            margin-bottom: 20px;
        }}
        .problem-number {{
            font-size: 12px;
            color: #999;
            margin-bottom: 5px;
        }}
        .problem-title {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }}
        .problem-link {{
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .problem-source {{
            font-size: 12px;
            color: #999;
        }}
        .similar-section {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }}
        .similar-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        .similar-item {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
        }}
        .similarity-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <a href="../index.html" class="back-button">‹ 戻る</a>
        <h1>{middle_cat}</h1>
    </div>
    <div class="container">
"""
    
    for item in problems:
        main = item['main_problem']
        html += f"""
        <div class="problem-card">
            <div class="main-problem">
                <div class="problem-number">問題番号: {main['問題番号']}</div>
                <div class="problem-title">{main['問題名']}</div>
                <a href="{main['リンク']}" target="_blank" class="problem-link">問題を見る ›</a>
                <div class="problem-source">出典: {main['出典']}</div>
            </div>
"""
        
        if item['similar_problems']:
            html += """
            <div class="similar-section">
                <div class="similar-title">📊 類似問題（類似度順）</div>
"""
            for similar in item['similar_problems'][:5]:  # 上位5件まで表示
                sim_percent = similar['similarity'] * 100
                sim = similar['data']
                html += f"""
                <div class="similar-item">
                    <span class="similarity-badge">{sim_percent:.1f}%</span>
                    <div class="problem-number">問題番号: {sim['問題番号']}</div>
                    <div class="problem-title">{sim['問題名']}</div>
                    <a href="{sim['リンク']}" target="_blank" class="problem-link">問題を見る ›</a>
                    <div class="problem-source">出典: {sim['出典']}</div>
                </div>
"""
            html += """
            </div>
"""
        
        html += """
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    pages_dir = os.path.join(output_dir, 'pages')
    os.makedirs(pages_dir, exist_ok=True)
    
    with open(os.path.join(pages_dir, f'{safe_filename}.html'), 'w', encoding='utf-8') as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description='中項目ごとに類似問題を表示するHTMLを生成します。')
    parser.add_argument('--csv_path', type=str, required=True, help='ベクトル化済みCSVファイルのパス')
    parser.add_argument('--config_path', type=str, default='../02_vectorize/config.yaml', help='config.yamlのパス')
    parser.add_argument('--output_dir', type=str, default='similar_finder', help='HTMLの出力先ディレクトリ')
    parser.add_argument('--model', type=str, help='使用するモデル名（指定しない場合は最初のモデル）')
    args = parser.parse_args()

    print_log("類似問題ファインダーの生成を開始します。")

    # 設定ファイルを読み込む
    config = load_config(args.config_path)
    if config is None:
        return
    
    models_config = config.get('models', [])
    if not models_config:
        print_log("エラー: config.yamlにモデル設定が見つかりません。")
        return

    # CSVを読み込む
    df = load_vectorized_csv(args.csv_path)
    if df is None:
        return
    
    # 必要な列が存在するか確認
    required_columns = ['大項目', '中項目', '問題番号', '問題名', 'リンク', '出典']
    for col in required_columns:
        if col not in df.columns:
            print_log(f"エラー: 必要な列 '{col}' が見つかりません。")
            return

    # モデルを選択
    if args.model:
        model_config = next((m for m in models_config if m['name'] == args.model), None)
        if not model_config:
            print_log(f"エラー: 指定されたモデル '{args.model}' が見つかりません。")
            return
    else:
        model_config = models_config[0]
    
    model_name = model_config['name']
    vector_column = get_vector_column_name(model_name)
    
    print_log(f"使用モデル: {model_name}")
    print_log(f"ベクトル列: {vector_column}")
    
    # ベクトル列が存在するかチェック
    if vector_column not in df.columns:
        print_log(f"エラー: ベクトル列 '{vector_column}' が見つかりません。")
        return

    # 中項目ごとにグループ化して類似度を計算
    grouped, category_similarities = group_by_middle_category(df, vector_column)
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # インデックスページを生成
    generate_index_page(category_similarities, args.output_dir, model_name)
    
    # 各中項目ページを生成
    print_log("中項目ごとのページを生成しています...")
    for middle_cat, problems in tqdm(category_similarities.items(), desc="ページ生成中"):
        generate_category_page(middle_cat, problems, args.output_dir)
    
    print_log("\n========== すべての処理が完了しました ==========")
    print_log(f"出力ディレクトリ: {args.output_dir}")
    print_log(f"インデックスページ: {os.path.join(args.output_dir, 'index.html')}")
    print_log(f"生成された中項目ページ数: {len(category_similarities)}")

if __name__ == '__main__':
    main()