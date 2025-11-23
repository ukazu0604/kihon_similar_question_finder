import subprocess
import sys
import time
import pytest
import requests
from requests.exceptions import ConnectionError

PORT = 8000
BASE_URL = f"http://localhost:{PORT}/"

def wait_for_server(timeout=15):
    """サーバーが起動して応答するまで待機する。"""
    print(f"サーバー ({BASE_URL}) の起動を待っています...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(BASE_URL)
            if response.status_code == 200:
                print("サーバーが起動しました。テストを開始します。")
                return True
        except ConnectionError:
            time.sleep(0.5)  # 0.5秒待ってから再試行
    print(f"エラー: {timeout}秒以内にサーバーが起動しませんでした。")
    return False

def main():
    """
    Webサーバーの起動、pytestの実行、サーバーの停止を順番に行う。
    サーバーが既に起動している場合は、それを利用してテストを実行する。
    """
    server_process = None
    server_already_running = False

    # 最初にサーバーが起動しているかチェック
    try:
        requests.get(BASE_URL, timeout=1)
        print("既存のサーバーが起動していることを検出しました。")
        print("このサーバーに対してテストを実行します。テスト後にサーバーは停止されません。")
        server_already_running = True
    except ConnectionError:
        # サーバーが起動していない場合は、この後で起動する
        pass

    try:
        if not server_already_running:
            # 1. Webサーバーをバックグラウンドプロセスとして起動
            # sys.executable を使うことで、現在実行中のPythonインタプリタを指定できる
            print("Webサーバーを起動します...")
            server_process = subprocess.Popen(
                [sys.executable, "start_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 2. サーバーが起動するのを待つ
            if not wait_for_server():
                sys.exit("テストを実行できませんでした。") # サーバーが起動しなかった場合は終了

        # 3. pytestを実行
        # pytest.main() は終了コードを返す (0は成功、それ以外は失敗)
        # 引数を指定しないことで、pytestがカレントディレクトリからテストを自動検出する
        print("pytestを実行します...")
        exit_code = pytest.main(["-v", "test_main_page.py", "test_browser_console.py", "test_ui_interactions.py"])

        # 4. スクリプト全体の終了コードをpytestの結果に合わせる
        if exit_code != 0:
            sys.exit(exit_code)

    finally:
        # 5. テストが成功しても失敗しても、必ずサーバーを停止する
        if server_process:
            print("\nWebサーバーを停止しています...")
            server_process.terminate() # プロセスを終了させる

if __name__ == "__main__":
    main()