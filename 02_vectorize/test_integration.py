import unittest
import ollama
from ollama import ResponseError
import yaml
import os

def print_log(message):
    """簡易的なログ出力"""
    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

class TestOllamaIntegration(unittest.TestCase):
    """Ollamaサーバーとの実際の通信をテストする結合テストクラス。"""
    
    ollama_models = []
    client = None

    @classmethod
    def setUpClass(cls):
        """テストクラスの最初に一度だけ実行される。設定ファイルの読み込みとOllamaクライアントの初期化を行う。"""
        print_log("Ollama 結合テストのセットアップを開始します。")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cls.ollama_models = [m for m in config.get('models', []) if m.get('type') == 'ollama']
        cls.client = ollama.Client()
        print_log("セットアップが完了しました。")

    def test_01_ollama_server_is_running(self):
        """Ollamaサーバーが起動しており、応答を返すかを確認する。"""
        print_log("\n--- (1/2) Ollamaサーバーの起動確認テスト ---")
        try:
            # ollama.ps() は起動中のモデルリストを返す。サーバーが落ちていると例外が発生する。
            self.client.ps()
            print_log("Ollamaサーバーは正常に応答しました。")
        except Exception as e:
            self.fail(f"Ollamaサーバーに接続できませんでした。Ollamaアプリケーションが起動しているか確認してください。\n詳細: {e}")

    def test_02_each_model_can_embed(self):
        """
        config.yaml に記載されている各Ollamaモデルが、実際にベクトルを生成できるかテストする。
        このテストを実行する前に、Ollamaアプリケーションを起動し、
        テスト対象のモデル (例: ollama run mxbai-embed-large) をインストールしておく必要があります。
        """
        for model_config in self.ollama_models:
            model_name = model_config['name']
            with self.subTest(model=model_name):
                print_log(f"モデル '{model_name}' のテストを開始します...")
                try:
                    client = ollama.Client()
                    # 簡単なテキストでベクトル化を試す
                    response = client.embed(model=model_name, input="これはテストです。")
                    
                    # レスポンスの形式をチェック
                    self.assertIn('embedding', response, f"モデル '{model_name}' のレスポンスに 'embedding' キーがありません。")
                    self.assertIsInstance(response['embedding'], list, f"モデル '{model_name}' の embedding がリストではありません。")
                    self.assertGreater(len(response['embedding']), 0, f"モデル '{model_name}' のベクトルが空です。")
                    
                    print_log(f"モデル '{model_name}' は正常にベクトルを生成しました。ベクトル次元数: {len(response['embedding'])}")
                except Exception as e:
                    self.fail(f"モデル '{model_name}' との通信中にエラーが発生しました。Ollamaが起動しているか、モデルがインストールされているか確認してください。\n詳細: {e}")

if __name__ == '__main__':
    unittest.main()