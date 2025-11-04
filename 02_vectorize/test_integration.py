import unittest
import ollama
from ollama import ResponseError
import httpx
import time
import yaml
import os
import sys

class TestOllamaIntegration(unittest.TestCase):
    """Ollamaサーバーとの実際の通信をテストする結合テストクラス。"""
    
    ollama_models = []
    client = None
    class_start_time = None

    @classmethod
    def print_log(cls, message):
        """タイムスタンプ付きでログを出力する"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    @classmethod
    def setUpClass(cls):
        """テストクラスの最初に一度だけ実行される。設定ファイルの読み込みとOllamaクライアントの初期化を行う。"""
        cls.class_start_time = time.time()
        cls.print_log("Ollama 結合テストのセットアップを開始します。")

        script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # run_tests_with_clipboard.pyからの実行に対応
        config_path = os.path.join(script_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cls.ollama_models = [m for m in config.get('models', []) if m.get('type') == 'ollama']
        # テスト実行時専用の短いタイムアウトを設定する
        # これにより、Ctrl+Cでの中断がしやすくなり、応答がない場合にテストがハングするのを防ぐ
        cls.client = ollama.Client(timeout=60) 
        cls.print_log("セットアップが完了しました。")

    @classmethod
    def tearDownClass(cls):
        """テストクラスの最後に一度だけ実行される。"""
        class_end_time = time.time()
        duration = class_end_time - cls.class_start_time
        cls.print_log("-" * 50)
        cls.print_log(f"Ollama 結合テストが完了しました。(総実行時間: {duration:.2f}秒)")

    def test_01_ollama_server_is_running(self):
        """Ollamaサーバーが起動しており、応答を返すかを確認する。"""
        self.print_log("\n--- (1/2) Ollamaサーバーの起動確認テスト ---")
        try:
            # ollama.ps() は起動中のモデルリストを返す。サーバーが落ちていると例外が発生する。
            self.client.ps()
            self.print_log("Ollamaサーバーは正常に応答しました。")
        except Exception as e:
            self.fail(f"Ollamaサーバーに接続できませんでした。Ollamaアプリケーションが起動しているか確認してください。\n詳細: {e}")

    def test_02_each_model_can_embed(self):
        """
        config.yaml に記載されている各Ollamaモデルが、実際にベクトルを生成できるかテストする。
        このテストを実行する前に、Ollamaアプリケーションを起動し、
        テスト対象のモデル (例: ollama run mxbai-embed-large) をインストールしておく必要があります。
        """
        if not self.ollama_models:
            self.skipTest("config.yaml に Ollama モデルが設定されていません。")

        for model_config in self.ollama_models:
            model_name = model_config['name']
            loop_start_time = time.time()
            with self.subTest(model=model_name, msg=f"モデル '{model_name}' のベクトル化テスト"):
                self.print_log(f"\n--- (2/2) モデル '{model_name}' のテスト ---")
                try:
                    self.print_log(f"  [開始] Ollamaクライアントを通じて '{model_name}' のベクトル化をリクエストします。")
                    self.print_log(f"  [待機] モデルのロードとベクトル化の完了を待っています... (PCのスペックにより数分以上かかる場合があります)")
                    start_time = time.time()
                    # 簡単なテキストでベクトル化を試す
                    # タイムアウト値をconfig.yamlから取得し、新しいクライアントを作成
                    model_timeout = model_config.get('timeout', 60) # configにない場合はデフォルト60秒
                    current_client = ollama.Client(timeout=model_timeout)

                    response = current_client.embed(model=model_name, input="これはテストです。")
                    end_time = time.time()
                    self.print_log(f"  [完了] Ollamaサーバーから応答を受け取りました。")
                    current_client.close() # リソースを解放するためにクライアントを閉じる
                    embed_duration = end_time - start_time
                    
                    # レスポンスの形式をチェック
                    self.assertIn('embedding', response, f"モデル '{model_name}' のレスポンスに 'embedding' キーがありません。")
                    self.assertIsInstance(response['embedding'], list, f"モデル '{model_name}' の embedding がリストではありません。")
                    self.assertGreater(len(response['embedding']), 0, f"モデル '{model_name}' のベクトルが空です。")
                    
                    self.print_log(f"  - ベクトル化成功 (embed処理時間: {embed_duration:.2f}秒, 次元数: {len(response['embedding'])})")
                
                except httpx.RemoteProtocolError as e:
                    self.fail(f"モデル '{model_name}' の処理中にOllamaサーバーとの接続が切れました。PCのメモリ不足が原因の可能性があります。\n詳細: {e}")
                except ConnectionError as e:
                    self.fail(f"モデル '{model_name}' のテスト開始時にOllamaサーバーに接続できませんでした。Ollamaがクラッシュした可能性があります。\n詳細: {e}")
                except ResponseError as e:
                    # モデルが存在しない場合のエラーを分かりやすく表示
                    if "model not found" in str(e):
                        self.fail(f"モデル '{model_name}' がOllamaにインストールされていません。`ollama run {model_name}` を実行してください。\n詳細: {e}")
                    else:
                        self.fail(f"モデル '{model_name}' の処理中にAPIエラーが発生しました。\n詳細: {e}")
                except Exception as e:
                    self.fail(f"モデル '{model_name}' との通信中にエラーが発生しました。Ollamaが起動しているか、モデルがインストールされているか確認してください。\n詳細: {e}")
                finally:
                    loop_end_time = time.time()
                    loop_duration = loop_end_time - loop_start_time
                    self.print_log(f"モデル '{model_name}' のテストが完了しました。(このモデルの合計テスト時間: {loop_duration:.2f}秒)")

if __name__ == '__main__':
    unittest.main()