import requests
import sys

def check_server(url="http://localhost:8000/"):
    print(f"サーバー ({url}) への接続を試行しています...")
    all_ok = True

    try:
        # メインページにアクセス
        response = requests.get(url)
        print(f"メインページ ({url}) のステータスコード: {response.status_code}")

        if response.status_code == 200:
            print("メインページは正常に読み込まれました。")
            # HTMLコンテンツの簡単なチェック（例: <title>タグの存在）
            if "<title>" in response.text and "</title>" in response.text:
                # print("HTMLコンテンツは正常なようです（<title>タグが見つかりました）。")
                pass # 成功時は簡潔にするため出力しない
            else:
                print("警告: HTMLコンテンツ内に<title>タグが見つかりませんでした。ページが空であるか、形式が正しくない可能性があります。")
                all_ok = False

            # 関連するリソースもチェック
            resources_to_check = [
                "app.js",
                "style.css",
                "03_html_output/similar_results.json"
            ]

            for resource in resources_to_check:
                resource_url = f"{url}{resource}"
                try:
                    resource_response = requests.get(resource_url)
                    if resource_response.status_code == 200:
                        # print(f"{resource} は正常に読み込まれました。")
                        pass # 成功時は簡潔にするため出力しない
                    else:
                        print(f"エラー: {resource} の読み込みに失敗しました。ステータスコード: {resource_response.status_code}")
                        all_ok = False
                except requests.exceptions.ConnectionError:
                    print(f"エラー: {resource_url} に接続できませんでした。サーバーは実行中ですか？")
                    all_ok = False
                except Exception as e:
                    print(f"エラー: {resource} のチェック中に予期せぬエラーが発生しました: {e}")
                    all_ok = False

        else:
            print(f"エラー: メインページの読み込みに失敗しました。ステータスコード: {response.status_code}")
            all_ok = False

    except requests.exceptions.ConnectionError:
        print(f"エラー: {url} に接続できませんでした。Pythonサーバーがポート8000で実行されていることを確認してください。")
        print("プロジェクトのルートディレクトリで 'py start_server.py' を実行してサーバーを起動できます。")
        all_ok = False
    except Exception as e:
        print(f"エラー: 予期せぬエラーが発生しました: {e}")
        all_ok = False

    if all_ok:
        print("\n--- サーバーチェック結果: 成功 ---")
        print("すべての主要なファイルが正常に配信されているようです。")
    else:
        print("\n--- サーバーチェック結果: 失敗 ---")
        print("上記のエラーメッセージを確認し、問題を修正してください。")

    print("\n--- 重要な注意 ---")
    print("このスクリプトは、サーバーがファイルを正しく配信しているか（HTTPステータスコード）をチェックします。")
    print("JavaScriptの構文エラーや実行時エラーは、Webブラウザ内で発生するため、このスクリプトでは検出できません。")
    print("JavaScriptのエラーを確認するには、http://localhost:8000/ を表示中にブラウザの開発者コンソール（F12キー -> Consoleタブ）を開いてください。")

if __name__ == "__main__":
    check_server()