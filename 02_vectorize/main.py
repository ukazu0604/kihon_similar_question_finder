
import yaml
import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import argparse
import json
import time

def print_log(message):
    """タイムスタンプ付きでログを出力する"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_config(config_path='config.yaml'):
    """設定ファイルを読み込む"""
    print_log("設定ファイル 'config.yaml' の読み込みを開始します...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print_log("設定ファイルの読み込みが完了しました。")
    return config

def load_data(file_path):
    """CSVデータを読み込む"""
    print_log(f"入力ファイル '{file_path}' の読み込みを開始します...")
    if not os.path.exists(file_path):
        print_log(f"エラー: 入力ファイルが見つかりません: {file_path}")
        return None
    df = pd.read_csv(file_path)
    print_log(f"入力ファイルの読み込みが完了しました。({len(df)}行)")
    return df

def get_texts_to_embed(df, text_column):
    """ベクトル化するテキストのリストを取得する"""
    print_log(f"'{text_column}' カラムからテキストの抽出を開始します...")
    if text_column not in df.columns:
        print_log(f"エラー: 指定されたテキストカラム '{text_column}' が見つかりません。")
        return None
    
    texts = df[text_column].fillna('').tolist()
    print_log(f"テキストの抽出が完了しました。({len(texts)}件)")
    return texts

def vectorize_with_ollama(model_name, texts):
    """Ollamaを使ってテキストをベクトル化する"""
    print_log(f"Ollamaモデル '{model_name}' でベクトル化を実行します...")
    vectors = []
    try:
        for text in tqdm(texts, desc=f"Vectorizing with {model_name}"):
            response = ollama.embed(model=model_name, prompt=text)
            vectors.append(response['embedding'])
    except Exception as e:
        print_log(f"エラー: Ollamaでのベクトル化中にエラーが発生しました。Ollamaが起動しているか確認してください。")
        print_log(f"詳細: {e}")
        return None
    print_log(f"Ollamaモデル '{model_name}' でのベクトル化が完了しました。")
    return np.array(vectors)

def vectorize_with_st(model_name, texts):
    """Sentence-Transformersを使ってテキストをベクトル化する"""
    print_log(f"Sentence-Transformersモデル '{model_name}' でベクトル化を実行します...")
    print_log("--- 注意: 初回実行時はモデルのダウンロードが始まります。(数分〜数十分かかる場合があります) ---")
    print_log("--- ダウンロード中は進捗が表示されないことがありますが、フリーズではありません。ご安心ください。 ---")
    try:
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        print_log(f"エラー: Sentence-Transformersでのベクトル化中にエラーが発生しました。インターネット接続を確認してください。")
        print_log(f"詳細: {e}")
        return None
    print_log(f"Sentence-Transformersモデル '{model_name}' でのベクトル化が完了しました。")
    return vectors

def save_results(output_dir, model_name, vectors, df, metadata_columns):
    """ベクトルとメタデータを保存する"""
    print_log("結果の保存を開始します...")
    os.makedirs(output_dir, exist_ok=True)

    model_filename = model_name.replace('/', '__') # ファイル名に使えない文字を置換
    vector_path = os.path.join(output_dir, f"vectors_{model_filename}.npy")
    np.save(vector_path, vectors)
    print_log(f"ベクトルを保存しました: {vector_path}")

    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print_log("メタデータファイルを新規作成します...")
        metadata = df[metadata_columns].to_dict(orient='records')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print_log(f"メタデータを保存しました: {metadata_path}")
    else:
        print_log("メタデータファイルは既に存在するため、作成をスキップしました。")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='設定ファイルに基づいてテキストをベクトル化します。')
    parser.add_argument('--model', type=str, help='実行するモデルのnameを個別に指定します。')
    parser.add_argument('--force', action='store_true', help='このフラグを立てると、既存のファイルがあっても強制的に再実行します。')
    args = parser.parse_args()

    print_log("ベクトル化処理を開始します。")
    config = load_config()
    
    df = load_data(config['input_file'])
    if df is None: return
        
    texts = get_texts_to_embed(df, config['text_column'])
    if texts is None: return

    models_to_run = config['models']
    if args.model:
        models_to_run = [m for m in models_to_run if m['name'] == args.model]
        if not models_to_run:
            print_log(f"エラー: 指定されたモデル名 '{args.model}' がconfig.yamlに見つかりません。")
            return

    print_log(f"{len(models_to_run)}件のモデル処理を開始します。")
    for model_config in models_to_run:
        model_name = model_config['name']
        model_type = model_config['type']
        
        model_filename = model_name.replace('/', '__')
        output_path = os.path.join(config['output_dir'], f"vectors_{model_filename}.npy")

        print_log(f"\n========== モデル '{model_name}' の処理を開始します ==========")

        if os.path.exists(output_path) and not args.force:
            print_log(f"出力ファイルが既に存在するため、スキップします: {output_path}")
            print_log("再実行したい場合は --force フラグを使用してください。")
            continue

        vectors = None
        if model_type == 'ollama':
            vectors = vectorize_with_ollama(model_name, texts)
        elif model_type == 'sentence-transformers':
            huggingface_name = model_config['huggingface_name']
            vectors = vectorize_with_st(huggingface_name, texts)
        else:
            print_log(f"警告: 未知のモデルタイプです: {model_type}。スキップします。")
            continue
        
        if vectors is not None:
            save_results(config['output_dir'], model_name, vectors, df, config['metadata_columns'])

    print_log("\n========== すべてのモデル処理が完了しました ==========")

if __name__ == '__main__':
    main()
