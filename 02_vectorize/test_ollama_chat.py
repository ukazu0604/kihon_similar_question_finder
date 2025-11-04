import ollama

def test_chat_connection():
    """
    Ollamaにインストール済みのチャットモデルを選択し、通信できるかを確認するテスト。
    """
    print("--- Ollama 接続テスト ---")
    model_name = "（未選択）" # エラー発生時に備えて初期化

    try:
        # 最初にサーバーが起動しているかを確認
        ollama.ps()
    except (ollama.RequestError, ConnectionError) as e:
        print(f"\n[失敗] Ollamaサーバーに接続できませんでした。")
        print("Ollamaアプリケーションが起動していることを確認してください。")
        print(f"詳細: {e}")
        return

    try:
        # 1. インストール済みのモデルリストを取得
        print("インストール済みのモデルリストを取得しています...")
        installed_models_info = ollama.list()
        models = installed_models_info.get('models', [])

        if not models:
            print("\n[エラー] Ollamaにモデルがインストールされていません。")
            print("コマンドプロンプトで `ollama run <モデル名>` を実行して、モデルをダウンロードしてください。")
            return

        # 2. モデルを選択させる
        print("\n利用可能なモデル:")
        for i, model in enumerate(models):
            print(f"  {i + 1}: {model.model}")

        while True:
            try:
                choice = input(f"\n使用するモデルの番号を選択してください (1-{len(models)}): ")
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(models):
                    model_name = models[choice_index].model
                    break
                else:
                    print("無効な番号です。もう一度入力してください。")
            except ValueError:
                print("数値を入力してください。")

        # 3. 選択されたモデルでチャットを実行
        print(f"\n--- モデル '{model_name}' とのチャットを開始します ---")
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'こんにちは、自己紹介をしてください。'}]
        )
        
        print("\n[成功] サーバーからの応答:")
        print(response.message.content if response.message else '応答が空です。')

    except (ollama.ResponseError, ConnectionError) as e:
        print(f"\n[失敗] エラーが発生しました。")
        print(f"Ollamaサーバーが起動しているか、モデル '{model_name}' がインストールされているか確認してください。")
        print(f"詳細: {e}")

if __name__ == '__main__':
    test_chat_connection()