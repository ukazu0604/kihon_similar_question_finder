
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
import math

def print_log(message):
    """タイムスタンプ付きでログを出力する"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_config(config_path='config.yaml'):
    """設定ファイルを読み込む"""
    # スクリプトの場所を基準に設定ファイルのパスを解決
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, config_path)

    print_log(f"設定ファイル '{absolute_config_path}' の読み込みを開始します...")
    with open(absolute_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)    
    print_log("設定ファイルの読み込みが完了しました。")
    return config

def load_data(file_path):
    """CSVデータを読み込む"""
    print_log(f"入力ファイル '{file_path}' の読み込みを開始します...")
    if not os.path.exists(file_path):
        print_log(f"エラー: 入力ファイルが見つかりません: {file_path}")
        return None
    df = pd.read_csv(file_path, encoding='utf-8-sig')
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

def process_in_batches(texts, model_name, model_type, huggingface_name=None, batch_size=32, output_dir='output'):
    """バッチ処理とレジューム機能付きでベクトル化を実行する"""
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    model_filename = model_name.replace('/', '__')
    final_path = os.path.join(output_dir, f"vectors_{model_filename}.npy")
    tmp_path = f"{final_path}.tmp"

    processed_vectors = []
    start_index = 0

    # 中間ファイルが存在すれば、そこから再開する
    if os.path.exists(tmp_path):
        print_log(f"中間ファイルが見つかりました: {tmp_path}")
        try:
            processed_vectors = list(np.load(tmp_path, allow_pickle=True))
            start_index = len(processed_vectors)
            print_log(f"処理を再開します。{start_index}件が既に処理済みです。")
        except Exception as e:
            print_log(f"警告: 中間ファイルの読み込みに失敗しました。最初から処理を開始します。詳細: {e}")
            processed_vectors = []
            start_index = 0

    texts_to_process = texts[start_index:]
    if not texts_to_process:
        print_log("すべてのテキストが既に処理済みです。")
    else:
        print_log(f"残り{len(texts_to_process)}件のテキストをベクトル化します...")
        
        # モデルのロード
        model = None
        if model_type == 'sentence-transformers':
            print_log(f"Sentence-Transformersモデル '{huggingface_name}' をロードします...")
            print_log("--- 注意: 初回実行時はモデルのダウンロードが始まります。(数分〜数十分かかる場合があります) ---")
            model = SentenceTransformer(huggingface_name)
            print_log("モデルのロードが完了しました。")

        # バッチ処理
        num_batches = math.ceil(len(texts_to_process) / batch_size)
        for i in tqdm(range(num_batches), desc=f"Vectorizing with {model_name}"):
            batch_texts = texts_to_process[i * batch_size : (i + 1) * batch_size]
            
            batch_vectors = []
            try:
                if model_type == 'ollama':
                    for text in batch_texts:
                        response = ollama.embed(model=model_name, input=text)
                        batch_vectors.append(response['embedding'])
                elif model_type == 'sentence-transformers' and model is not None:
                    batch_vectors = model.encode(batch_texts, convert_to_numpy=True).tolist()
            except Exception as e:
                print_log(f"\nエラー: バッチ処理中にエラーが発生しました。モデル: {model_name}")
                print_log(f"詳細: {e}")
                print_log("ここまでの進捗を保存して処理を中断します。")
                return False # 処理失敗

            processed_vectors.extend(batch_vectors)
            
            # バッチ完了ごとに中間ファイルを保存
            try:
                np.save(tmp_path, np.array(processed_vectors, dtype=object), allow_pickle=True)
            except Exception as e:
                print_log(f"\n警告: 中間ファイルの保存に失敗しました。詳細: {e}")

    # すべての処理が完了したら、一時ファイルをリネーム
    try:
        os.rename(tmp_path, final_path)
        print_log(f"最終的なベクトルファイルを保存しました: {final_path}")
        return True # 処理成功
    except FileNotFoundError:
        if os.path.exists(final_path) or not texts_to_process:
             print_log("既に最終ファイルが存在します。正常に完了しています。")
             return True
        else:
             print_log("エラー: 処理対象のテキストがありませんでした。")
             return False
    except Exception as e:
        print_log(f"エラー: 最終ファイルへのリネームに失敗しました。詳細: {e}")
        return False

def save_metadata(output_dir, df, metadata_columns):
    """メタデータを保存する"""
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

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
    
    # config.yamlからの相対パスを、config.yamlの場所を基準にした絶対パスに変換
    config_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yaml')))
    input_file_path = os.path.join(config_dir, config['input_file'])
    # パスを正規化して '..' などを解決
    input_file_path = os.path.normpath(input_file_path)

    df = load_data(input_file_path)
    if df is None: return
        
    texts = get_texts_to_embed(df, config['text_column'])
    if texts is None: return

    # メタデータを先に保存
    save_metadata(config['output_dir'], df, config['metadata_columns'])

    models_to_run = config['models']
    if args.model:
        models_to_run = [m for m in models_to_run if m['name'] == args.model]
        if not models_to_run:
            print_log(f"エラー: 指定されたモデル名 '{args.model}' がconfig.yamlに見つかりません。")
            return

    print_log(f"{len(models_to_run)}件のモデル処理を開始します。")
    for model_config in models_to_run:
        model_name = model_config['name']
        model_filename = model_name.replace('/', '__')
        final_path = os.path.join(config['output_dir'], f"vectors_{model_filename}.npy")

        print_log(f"\n========== モデル '{model_name}' の処理を開始します ==========")

        # 差分実行のチェック (個別実行のレジュームはprocess_in_batches内で行う)
        if os.path.exists(final_path) and not args.force:
            print_log(f"最終出力ファイルが既に存在するため、スキップします: {final_path}")
            print_log("再実行したい場合は --force フラグを使用してください。")
            continue
        
        # 強制再実行の場合は、既存のファイルを削除
        if args.force:
            if os.path.exists(final_path):
                print_log(f"--forceフラグが指定されたため、既存のファイルを削除します: {final_path}")
                os.remove(final_path)
            tmp_path = f"{final_path}.tmp"
            if os.path.exists(tmp_path):
                print_log(f"--forceフラグが指定されたため、既存の中間ファイルを削除します: {tmp_path}")
                os.remove(tmp_path)

        process_in_batches(
            texts=texts,
            model_name=model_name,
            model_type=model_config['type'],
            huggingface_name=model_config.get('huggingface_name'),
            batch_size=config['batch_size'],
            output_dir=config['output_dir']
        )

    print_log("\n========== すべてのモデル処理が完了しました ==========")

if __name__ == '__main__':
    main()
