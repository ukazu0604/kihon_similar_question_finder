import unittest
import sys
import io
import subprocess
import os

class Tee(object):
    """コンソールへのリアルタイム出力と、メモリへのキャプチャを両立させるためのクラス"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # リアルタイムで出力するためにフラッシュ
    def flush(self):
        for f in self.files:
            f.flush()

def run_tests_and_copy_on_failure():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_output = io.StringIO()

    # コンソールへの出力とキャプチャを両立させる
    sys.stdout = Tee(original_stdout, captured_output)
    sys.stderr = Tee(original_stderr, captured_output)

    try:
        # テストスイートをロードして実行
        # このスクリプト(run_tests_with_clipboard.py)があるディレクトリからテストを探す
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir, pattern="test_*.py")
        
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
        result = runner.run(suite)

        if not result.wasSuccessful():
            # 失敗した場合、キャプチャした出力をクリップボードにコピー
            output_content = captured_output.getvalue()
            
            # 元のstdout/stderrに戻す
            sys.stdout, sys.stderr = original_stdout, original_stderr

            print("\nテストが失敗しました。コンソール出力をクリップボードにコピーします。", file=original_stderr)
            try:
                subprocess.run(['clip.exe'], input=output_content.encode('cp932', errors='replace'), check=True)
                print("コンソール出力がクリップボードにコピーされました。", file=original_stderr)
            except Exception as clip_e:
                print(f"クリップボードへのコピーに失敗しました。詳細: {clip_e}", file=original_stderr)

    except Exception as e:
        # 予期せぬエラーが発生した場合
        sys.stdout, sys.stderr = original_stdout, original_stderr
        print(f"テスト実行中に予期せぬエラーが発生しました: {e}", file=original_stderr)
        # ここでもクリップボードにコピーする選択肢もあるが、今回はテスト失敗時のみとする

    finally:
        # 確実に元のstdout/stderrに戻す
        sys.stdout, sys.stderr = original_stdout, original_stderr
        # 成功した場合は、最後に成功メッセージを表示
        if 'result' in locals() and result.wasSuccessful():
            print("\nすべてのテストが成功しました。")

if __name__ == '__main__':
    run_tests_and_copy_on_failure()
