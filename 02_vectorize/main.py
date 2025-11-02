
import yaml
import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import argparse
import json

def load_config(config_path='config.yaml'):
    """設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """CSVデータを読み込む"""
    if not os.path.exists(file_path):
        print(f"エラー: 入力ファイルが見つかりません: {file_path}")
        return None
    return pd.read_csv(file_path)

def get_texts_to_embed(df, text_column):
    """ベクトル化するテキストのリストを取得する"""
    if text_column not in df.columns:
        print(f"エラー: 指定されたテキストカラム '{text_column}' が見つかりません。")
        return None
    
    # 欠損値を空文字列に置き換える
    return df[text_column].fillna('').tolist()

def vectorize_with_ollama(model_name, texts):
    """Ollamaを使ってテキストをベクトル化する"""
    print(f"Ollamaモデル '{model_name}' でベクトル化を実行中...")
    vectors = []
    for text in tqdm(texts, desc=f"Vectorizing with {model_name}"):
        response = ollama.embed(model=model_name, prompt=text)
        vectors.append(response['embedding'])
    return np.array(vectors)

def vectorize_with_st(model_name, texts):
    """Sentence-Transformersを使ってテキストをベクトル化する"""
    print(f"Sentence-Transformersモデル '{model_name}' でベクトル化を実行中...")
    print("（初回実行時はモデルのダウンロードに時間がかかります）")
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return vectors

def save_results(output_dir, model_name, vectors, df, metadata_columns):
    """ベクトルとメタデータを保存する"""
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)

    # ベクトルを保存
    vector_path = os.path.join(output_dir, f"vectors_{model_name.replace('/', '_')}.npy")
    np.save(vector_path, vectors)
    print(f"ベクトルを保存しました: {vector_path}")

    # メタデータを保存 (まだ保存されていなければ)
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        metadata = df[metadata_columns].to_dict(orient='records')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"メタデータを保存しました: {metadata_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='設定ファイルに基づいてテキストをベクトル化します。')
    parser.add_argument('--model', type=str, help='実行するモデルのnameを個別に指定します。')
    parser.add_argument('--force', action='store_true', help='このフラグを立てると、既存のファイルがあっても強制的に再実行します。')
    args = parser.parse_args()

    # 設定ファイルを読み込む
    config = load_config()
    input_file = config['input_file']
    text_column = config['text_column']
    metadata_columns = config['metadata_columns']
    output_dir = config['output_dir']
    
    # データを読み込む
    df = load_data(input_file)
    if df is None:
        return
        
    texts = get_texts_to_embed(df, text_column)
    if texts is None:
        return

    # 実行するモデルを決定
    models_to_run = config['models']
    if args.model:
        models_to_run = [m for m in models_to_run if m['name'] == args.model]
        if not models_to_run:
            print(f"エラー: 指定されたモデル名 '{args.model}' がconfig.yamlに見つかりません。")
            return

    # 各モデルで処理を実行
    for model_config in models_to_run:
        model_name = model_config['name']
        model_type = model_config['type']
        
        # 出力ファイル名を生成
        output_filename = f"vectors_{model_name.replace('/', '_')}.npy"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n--- モデル '{model_name}' の処理を開始 ---")

        # 差分実行のチェック
        if os.path.exists(output_path) and not args.force:
            print(f"出力ファイルが既に存在するため、スキップします: {output_path}")
            print("再実行したい場合は --force フラグを使用してください。")
            continue

        vectors = None
        if model_type == 'ollama':
            vectors = vectorize_with_ollama(model_name, texts)
        elif model_type == 'sentence-transformers':
            huggingface_name = model_config['huggingface_name']
            vectors = vectorize_with_st(huggingface_name, texts)
        else:
            print(f"警告: 未知のモデルタイプです: {model_type}。スキップします。")
            continue
        
        if vectors is not None:
            save_results(output_dir, model_name, vectors, df, metadata_columns)

    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    main()
