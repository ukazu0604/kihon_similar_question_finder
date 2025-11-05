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
import sys
import io
import subprocess

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
    return config, script_dir

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

def get_vector_column_name(model_name):
    """モデル名からベクトル列名を生成"""
    return f"vector_{model_name.replace('/', '_').replace('.', '_').replace(':', '_')}"

def process_in_batches_csv(df, texts, model_config, batch_size=32, output_path=None, text_column='text', debug=False):
    """バッチ処理とレジューム機能付きでベクトル化を実行し、CSVに書き込む"""
    if debug:
        print_log(f"DEBUG: process_in_batches_csv called with model_config: {model_config}")

    model_name = model_config['name']
    model_type = model_config['type']
    huggingface_name = model_config.get('huggingface_name')
    
    vector_column = get_vector_column_name(model_name)

    # 既存のCSVを読み込むか、新規作成
    if os.path.exists(output_path):
        print_log(f"既存の出力ファイルが見つかりました: {output_path}")
        output_df = pd.read_csv(output_path, encoding='utf-8-sig')
        print_log(f"出力ファイルを読み込みました。({len(output_df)}行)")
    else:
        print_log(f"新規の出力ファイルを作成します: {output_path}")
        output_df = df.copy()
    
    # ベクトル列が存在しない場合は追加
    if vector_column not in output_df.columns:
        output_df[vector_column] = None
        print_log(f"ベクトル列 '{vector_column}' を追加しました。")
    
    # 未処理の行を特定（ベクトル列がNaNまたは空の行）
    unprocessed_mask = output_df[vector_column].isna() | (output_df[vector_column] == '') | (output_df[vector_column] == 'None')
    unprocessed_indices = output_df[unprocessed_mask].index.tolist()
    
    if not unprocessed_indices:
        print_log(f"モデル '{model_name}' のベクトル化は既に完了しています。")
        return True
    
    print_log(f"未処理の行: {len(unprocessed_indices)}件 / 全体: {len(output_df)}件")
    
    # モデルのロード
    if model_type == 'sentence-transformers':
        print_log(f"Sentence-Transformersモデル '{huggingface_name}' をロードします...")
        print_log("--- 注意: 初回実行時はモデルのダウンロードが始まります。(数分〜数十分かかる場合があります) ---")
        model = SentenceTransformer(huggingface_name)
        print_log("モデルのロードが完了しました。")
    elif model_type == 'ollama':
        # Ollamaクライアントにタイムアウトを設定
        timeout = model_config.get('timeout', 30)
        client = ollama.Client(timeout=timeout)
        print_log("Ollamaクライアントを初期化しました。")
        try:
            print_log(f"モデル '{model_name}' のウォームアップを開始します... (初回は数分以上かかることがあります)")
            client.embeddings(model=model_name, prompt="warm-up")
            print_log("ウォームアップが完了しました。")
        except Exception as e:
            print_log(f"エラー: モデルのウォームアップ中にエラーが発生しました。詳細: {e}")
            return False
    
    # バッチ処理
    num_batches = math.ceil(len(unprocessed_indices) / batch_size)
    if debug:
        print_log(f"[DEBUG] 未処理テキスト総数: {len(unprocessed_indices)}")
        print_log(f"[DEBUG] モデル: {model_name}, バッチサイズ: {batch_size}, バッチ数: {num_batches}")
    
    save_interval = 10  # 10バッチごとに保存
    
    with tqdm(total=len(unprocessed_indices), desc=f"Vectorizing with {model_name}") as pbar:
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(unprocessed_indices))
            batch_indices = unprocessed_indices[batch_start:batch_end]
            batch_texts = [texts[idx] for idx in batch_indices]
            
            try:
                if model_type == 'ollama':
                    # 1件ずつ処理
                    for i, (idx, text) in enumerate(zip(batch_indices, batch_texts)):
                        # プログレスバーに現在処理中のテキストを表示
                        pbar.set_postfix_str(f"Now: {text[:40]}...")
                        
                        # 空白文字のみのテキストはスキップ
                        if not text.strip():
                            output_df.at[idx, vector_column] = '[]'
                            pbar.update(1)
                            continue
                        
                        response = client.embeddings(model=model_name, prompt=text)
                        vector = response['embedding']
                        # ベクトルをJSON文字列として保存
                        output_df.at[idx, vector_column] = json.dumps(vector)
                        pbar.update(1)
                
                elif model_type == 'sentence-transformers':
                    # バッチで処理
                    vectors = model.encode(batch_texts, convert_to_numpy=True)
                    for idx, vector in zip(batch_indices, vectors):
                        # ベクトルをJSON文字列として保存
                        output_df.at[idx, vector_column] = json.dumps(vector.tolist())
                    pbar.update(len(batch_texts))
            
            except Exception as e:
                print_log(f"\nエラー: バッチ処理中にエラーが発生しました。モデル: {model_name}")
                print_log(f"詳細: {e}")
                print_log("ここまでの進捗を保存して処理を中断します。")
                # エラー時も保存
                try:
                    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print_log(f"中間結果を保存しました: {output_path}")
                except Exception as save_error:
                    print_log(f"警告: CSVファイルの保存にも失敗しました。詳細: {save_error}")
                return False
            
            # 定期的に保存（save_intervalバッチごと、または最終バッチ）
            if (batch_num + 1) % save_interval == 0 or (batch_num + 1) == num_batches:
                try:
                    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    if debug:
                        print_log(f"\n[DEBUG] 中間保存完了: バッチ {batch_num + 1}/{num_batches}")
                except Exception as e:
                    print_log(f"\n警告: CSVファイルの保存に失敗しました。詳細: {e}")
    
    # 最終保存
    try:
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print_log(f"最終的なCSVファイルを保存しました: {output_path}")
        return True
    except Exception as e:
        print_log(f"エラー: 最終ファイルの保存に失敗しました。詳細: {e}")
        return False

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='設定ファイルに基づいてテキストをベクトル化し、CSVに書き込みます。')
    parser.add_argument('--model', type=str, help='実行するモデルのnameを個別に指定します。')
    parser.add_argument('--force', action='store_true', help='このフラグを立てると、既存のベクトル列を削除して強制的に再実行します。')
    parser.add_argument('--debug', action='store_true', help='デバッグログを有効にします。')
    args = parser.parse_args()

    print_log("ベクトル化処理を開始します。")
    config, config_dir = load_config()
    
    # config.yamlからの相対パスを、config.yamlの場所を基準にした絶対パスに変換
    input_file_path = os.path.join(config_dir, config['input_file'])
    input_file_path = os.path.normpath(input_file_path)

    df = load_data(input_file_path)
    if df is None: 
        return
        
    texts = get_texts_to_embed(df, config['text_column'])
    if texts is None: 
        return

    # 出力ディレクトリの作成
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 出力CSVファイル名（入力ファイル名に _vectors を付加）
    input_basename = os.path.basename(input_file_path)
    output_filename = input_basename.replace('.csv', '_vectors.csv')
    output_path = os.path.join(output_dir, output_filename)
    
    print_log(f"出力先: {output_path}")

    models_to_run = config['models']
    if args.model:
        models_to_run = [m for m in models_to_run if m['name'] == args.model]
        if not models_to_run:
            print_log(f"エラー: 指定されたモデル名 '{args.model}' がconfig.yamlに見つかりません。")
            return

    processing_times = {}

    print_log(f"{len(models_to_run)}件のモデル処理を開始します。")
    for model_config in models_to_run:
        model_name = model_config['name']
        vector_column = get_vector_column_name(model_name)

        print_log(f"\n========== モデル '{model_name}' の処理を開始します ==========")

        # 強制再実行の場合は、該当するベクトル列を削除
        if args.force and os.path.exists(output_path):
            print_log(f"--forceフラグが指定されたため、ベクトル列 '{vector_column}' を削除します...")
            temp_df = pd.read_csv(output_path, encoding='utf-8-sig')
            if vector_column in temp_df.columns:
                temp_df = temp_df.drop(columns=[vector_column])
                temp_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print_log(f"ベクトル列 '{vector_column}' を削除しました。")

        start_time = time.time()
        success = process_in_batches_csv(
            df=df,
            texts=texts,
            model_config=model_config,
            batch_size=config['batch_size'],
            output_path=output_path,
            text_column=config['text_column'],
            debug=args.debug
        )
        end_time = time.time()

        if success:
            duration = end_time - start_time
            processing_times[model_name] = f"{duration:.2f}秒"
            print_log(f"モデル '{model_name}' の処理が完了しました。(処理時間: {duration:.2f}秒)")
        else:
            print_log(f"モデル '{model_name}' の処理中にエラーが発生したため、中断されました。")

    print_log("\n========== すべてのモデル処理が完了しました ==========")
    print_log(f"出力ファイル: {output_path}")
    print_log("各モデルの処理時間:")
    for model, duration in processing_times.items():
        print_log(f"- {model}: {duration}")

if __name__ == '__main__':
    main()