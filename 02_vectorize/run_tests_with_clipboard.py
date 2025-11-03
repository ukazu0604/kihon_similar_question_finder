import unittest
import sys
import io
import subprocess
import os

def run_tests_and_copy_on_failure():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_output = io.StringIO()

    # stdoutとstderrをキャプチャ
    sys.stdout = captured_output
    sys.stderr = captured_output

    try:
        # テストスイートをロードして実行
        # test_vectorize.py のテストを検出
        # このスクリプト(run_tests_with_clipboard.py)があるディレクトリからテストを探す
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir, pattern="test_*.py")
        
        runner = unittest.TextTestRunner(stream=captured_output, verbosity=2)
        result = runner.run(suite)

        if not result.wasSuccessful():
            # 失敗した場合、キャプチャした出力をクリップボードにコピー
            output_content = captured_output.getvalue()
            
            # 元のstdout/stderrに戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            print("\nテストが失敗しました。コンソール出力をクリップボードにコピーします。", file=original_stderr)
            try:
                subprocess.run(['clip.exe'], input=output_content.encode('utf-8'), check=True)
                print("コンソール出力がクリップボードにコピーされました。", file=original_stderr)
            except Exception as clip_e:
                print(f"クリップボードへのコピーに失敗しました。詳細: {clip_e}", file=original_stderr)
        else:
            # 成功した場合も元のstdout/stderrに戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("\nすべてのテストが成功しました。", file=original_stdout)

    except Exception as e:
        # 予期せぬエラーが発生した場合
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"テスト実行中に予期せぬエラーが発生しました: {e}", file=original_stderr)
        # ここでもクリップボードにコピーする選択肢もあるが、今回はテスト失敗時のみとする

    finally:
        # 確実に元のstdout/stderrに戻す
        sys.stdout = original_stdout
        sys.stderr = original_stderr

if __name__ == '__main__':
    run_tests_and_copy_on_failure()
