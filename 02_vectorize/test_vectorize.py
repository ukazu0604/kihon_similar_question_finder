import unittest
import os
import shutil
import yaml
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト用のダミーデータと設定
DUMMY_CONFIG_CONTENT = """
input_file: "dummy_data.csv"
text_column: "問題名"
metadata_columns: ["問題番号", "中項目", "問題名", "リンク", "出典"]
output_dir: "test_output"
batch_size: 2
models:
  - name: mxbai-embed-large
    type: ollama
    timeout: 5 # テスト用に短く設定
  - name: intfloat/multilingual-e5-large
    type: sentence-transformers
    huggingface_name: intfloat/multilingual-e5-large
"""

DUMMY_CSV_CONTENT = """問題番号,中項目,問題名,リンク,出典
1,テクノロジ系,ダミー問題1のテキストです。,http://example.com/q1,情報処理推進機構
2,テクノロジ系,ダミー問題2のテキストです。,http://example.com/q2,情報処理推進機構
3,マネジメント系,ダミー問題3のテキストです。,http://example.com/q3,情報処理推進機構
4,ストラテジ系,ダミー問題4のテキストです。,http://example.com/q4,情報処理推進機構
"""

class TestVectorization(unittest.TestCase):
    TEST_DIR = "test_env"
    OUTPUT_DIR = os.path.join(TEST_DIR, "test_output")
    CONFIG_PATH = os.path.join(TEST_DIR, "config.yaml")
    DUMMY_CSV_PATH = os.path.join(TEST_DIR, "dummy_data.csv")

    @classmethod
    def setUpClass(cls):
        # テスト環境のセットアップ
        os.makedirs(cls.TEST_DIR, exist_ok=True)
        with open(cls.CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(DUMMY_CONFIG_CONTENT)
        with open(cls.DUMMY_CSV_PATH, "w", encoding="utf-8-sig") as f: # utf-8-sigで保存
            f.write(DUMMY_CSV_CONTENT)

        # main.py の関数をインポート
        # 相対インポートはテストスクリプトの実行方法に依存するため、今回はパスを追加して絶対インポートを試みます。
        import sys
        # 02_vectorize ディレクトリの親ディレクトリをパスに追加
        sys.path.insert(0, os.path.dirname(__file__)) # スクリプトのあるディレクトリを優先
        from main import load_config, load_data, get_texts_to_embed, process_in_batches, save_metadata
        cls.load_config = staticmethod(load_config)
        cls.load_data = staticmethod(load_data)
        cls.get_texts_to_embed = staticmethod(get_texts_to_embed)
        cls.process_in_batches = staticmethod(process_in_batches)
        cls.save_metadata = staticmethod(save_metadata)


    @classmethod
    def tearDownClass(cls):
        # テスト環境のクリーンアップ
        shutil.rmtree(cls.TEST_DIR)

    def setUp(self):
        # 各テストの前に出力ディレクトリをクリーンアップ
        if os.path.exists(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR)

    def test_ollama_vectorization(self):
        print("\n--- Running Ollama Vectorization Test ---")
        config = self.load_config(self.CONFIG_PATH)
        df = self.load_data(self.DUMMY_CSV_PATH)
        texts = self.get_texts_to_embed(df, config['text_column'])

        ollama_config = next(m for m in config['models'] if m['type'] == 'ollama')

        # ファイルI/O関連の関数をモック
        with patch('main.ollama.Client') as MockOllamaClient, \
             patch('main.np.save') as MockNpSave, \
             patch('main.os.rename') as MockOsRename:
            mock_client_instance = MockOllamaClient.return_value # clientインスタンスのモック
            # embedが呼ばれるたびに、正しい形式のレスポンスを返すように設定
            mock_client_instance.embed.return_value = {'embedding': [0.1, 0.2, 0.3]}

            result = self.process_in_batches(
                texts=texts,
                model_config=ollama_config,
                batch_size=config['batch_size'],
                output_dir=self.OUTPUT_DIR,
                debug=True # デバッグログを有効に
            )
            self.assertTrue(result, "Ollama vectorization should succeed")
            
            # np.save が呼ばれたことを確認
            self.assertTrue(MockNpSave.called)
            # os.rename が呼ばれたことを確認
            self.assertTrue(MockOsRename.called)


    def test_sentence_transformers_vectorization(self):
        print("\n--- Running Sentence-Transformers Vectorization Test ---")
        config = self.load_config(self.CONFIG_PATH)
        df = self.load_data(self.DUMMY_CSV_PATH)
        texts = self.get_texts_to_embed(df, config['text_column'])

        st_config = next(m for m in config['models'] if m['type'] == 'sentence-transformers')

        # SentenceTransformer をモック
        with patch('main.SentenceTransformer') as MockST, \
             patch('main.np.save') as MockNpSave, \
             patch('main.os.rename') as MockOsRename:
            mock_st_instance = MockST.return_value
            mock_st_instance.encode.return_value = np.array([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]) # バッチサイズ2に対応

            result = self.process_in_batches(
                texts=texts,
                model_config=st_config,
                batch_size=config['batch_size'],
                output_dir=self.OUTPUT_DIR,
                debug=True # デバッグログを有効に
            )
            self.assertTrue(result, "Sentence-Transformers vectorization should succeed")
            # np.save が呼ばれたことを確認
            self.assertTrue(MockNpSave.called)
            # os.rename が呼ばれたことを確認
            self.assertTrue(MockOsRename.called)

    def test_resume_functionality(self):
        print("\n--- Running Resume Functionality Test ---")
        config = self.load_config(self.CONFIG_PATH)
        df = self.load_data(self.DUMMY_CSV_PATH)
        texts = self.get_texts_to_embed(df, config['text_column'])

        ollama_config = next(m for m in config['models'] if m['type'] == 'ollama')

        # 最初のバッチだけ処理した中間ファイルを作成
        model_filename = str(ollama_config['name']).replace('/', '__').replace('.', '_')
        tmp_path = os.path.join(self.OUTPUT_DIR, f"vectors_{model_filename}.npy.tmp")

        # 最初のバッチ (2件) のダミーベクトル
        initial_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=object)
        np.save(tmp_path, initial_vectors, allow_pickle=True)

        # ollama.Client.embed をモック
        with patch('main.ollama.Client') as MockOllamaClient, \
             patch('main.np.save') as MockNpSave, \
             patch('main.os.rename') as MockOsRename:
            mock_client_instance = MockOllamaClient.return_value
            # 残りのテキスト (2件) 用のダミーベクトル
            mock_client_instance.embed.return_value = {'embedding': [0.7, 0.8, 0.9]}

            result = self.process_in_batches(
                texts=texts,
                model_config=ollama_config,
                batch_size=config['batch_size'],
                output_dir=self.OUTPUT_DIR,
                debug=True
            )
            self.assertTrue(result, "Resume functionality should succeed")

            # np.save が呼ばれたことを確認
            self.assertTrue(MockNpSave.called)
            # os.rename が呼ばれたことを確認
            self.assertTrue(MockOsRename.called)

    @patch('main.subprocess.run')
    @patch('main.sys.stdout', new_callable=io.StringIO)
    @patch('main.sys.stderr', new_callable=io.StringIO)
    def test_clipboard_on_error(self, mock_stderr, mock_stdout, mock_subprocess_run):
        print("\n--- Running Clipboard on Error Test ---")
        # main.load_config がエラーを発生させるようにモック
        with patch('main.load_config', side_effect=Exception("Test configuration loading error")):
            # main 関数を呼び出す
            # main 関数内で例外が捕捉され、subprocess.run が呼ばれることを期待
            try:
                from main import main as main_function
                main_function()
            except Exception as e:
                # main 関数内で例外が再raiseされることを確認
                self.assertIn("Test configuration loading error", str(e))
            
            # subprocess.run が呼ばれたことを確認
            mock_subprocess_run.assert_called_once()
            
            # clip.exe が呼ばれたことを確認
            self.assertEqual(mock_subprocess_run.call_args[0][0][0], 'clip.exe')
            
            # クリップボードにコピーされる内容を検証
            copied_content = mock_subprocess_run.call_args[1]['input'].decode('utf-8')
            self.assertIn("Test configuration loading error", copied_content)
            self.assertIn("エラーが発生しました。コンソール出力をクリップボードにコピーします。", copied_content)
            print("Clipboard on Error Test Passed.")

if __name__ == '__main__':
    unittest.main()