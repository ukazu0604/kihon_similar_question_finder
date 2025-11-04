import unittest
import ollama
from ollama import ResponseError
import httpx

class TestOllamaMxbaiEmbedLargeSimple(unittest.TestCase):
    """Ollamaの 'mxbai-embed-large' モデルがベクトルを生成できるか、最小限のテストを行う。"""

    def test_mxbai_embed_large_vectorization(self):
        """'mxbai-embed-large'モデルでベクトル化を試みる。"""
        model_name = "mxbai-embed-large"
        test_text = "これはテストです。"

        # httpxのイベントフックを使って、実際のリクエスト内容をログに出力する関数
        def log_request(request: httpx.Request):
            print("\n--- 実際に送信されたHTTPリクエスト ---")
            print(f"> Method: {request.method}")
            print(f"> URL: {request.url}")
            print("> Headers:")
            for key, value in request.headers.items():
                print(f">   {key}: {value}")
            if request.content:
                # ボディはbytes形式なのでデコードして表示
                print(f"> Body: {request.content.decode('utf-8')}")
            print("------------------------------------")

        try:
            # タイムアウトとイベントフックを設定してクライアントを初期化
            client = ollama.Client(
                timeout=300,
                event_hooks={'request': [log_request]}
            )

            print(f"\n--- モデル '{model_name}' のベクトル化テストを開始します ---")
            print(f"PCのスペックにより数分以上かかる場合があります...")

            response = client.embed(model=model_name, input=test_text)

            # レスポンスの基本的な検証
            self.assertIn('embedding', response)
            self.assertIsInstance(response['embedding'], list)
            self.assertGreater(len(response['embedding']), 0)

            print(f"--- モデル '{model_name}' のベクトル化テストに成功しました ---")
        except ResponseError as e:
            if "model not found" in str(e):
                self.fail(f"モデル '{model_name}' が見つかりません。`ollama run {model_name}` を実行してください。\n詳細: {e}")
            else:
                self.fail(f"APIエラーが発生しました: {e}")
        except Exception as e:
            self.fail(f"Ollamaサーバーへの接続中に予期せぬエラーが発生しました。Ollamaが起動しているか確認してください。\n詳細: {e}")

if __name__ == '__main__':
    unittest.main()