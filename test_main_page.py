import time
import pytest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

BASE_URL = "http://localhost:8000/"

def wait_for_page_load(driver):
    """
    app.jsによってコンテンツが描画されるのを待つヘルパー関数。
    """
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#category-list > div"))
    )



def test_content_is_rendered(driver):
    """JSONが読み込まれ、カテゴリリストが描画されるかテストする。"""
    driver.get(BASE_URL)
    wait_for_page_load(driver)
    
    categories = driver.find_elements(By.CLASS_NAME, "middle-category-item")
    assert len(categories) > 0, "カテゴリコンテナが1つ以上表示されるべきです。"

def test_no_severe_console_errors(driver):
    """ブラウザコンソールに深刻な(SEVERE)エラーがないかテストする。"""
    driver.get(BASE_URL)
    wait_for_page_load(driver)
    time.sleep(1) # 非同期処理が完了するのを少し待つ

    logs = driver.get_log('browser')
    severe_errors = [log for log in logs if log['level'] == 'SEVERE']
    
    if severe_errors:
        messages = [err['message'] for err in severe_errors]
        # pytest.fail を使うと、テストが失敗し、詳細なエラーメッセージが表示される
        pytest.fail(f"深刻なコンソールエラーが検出されました:\n{messages}")

    assert not severe_errors

def test_untouched_filter_does_not_hide_on_check(driver):
    """
    「未着手のみ表示」がオンの時、問題にチェックを入れても即座に非表示にならないことをテストする。
    リストが再描画されたタイミングで非表示になることを確認する。
    """
    driver.get(BASE_URL)
    wait_for_page_load(driver)

    # 1. 状態をリセットして最初のカテゴリに移動
    driver.execute_script("localStorage.clear();")
    driver.refresh()
    wait_for_page_load(driver)
    driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "detail-container")))

    # 2. 「未着手のみ表示」をオンにする
    untouched_checkbox = driver.find_element(By.ID, "show-untouched-only")
    untouched_checkbox.click()
    time.sleep(0.5) # フィルター適用のための短い待機

    # 3. 最初の問題にチェックを入れる前の問題数を記録
    problems_before_check = driver.find_elements(By.CLASS_NAME, "problem-card")
    initial_problem_count = len(problems_before_check)
    assert initial_problem_count > 0, "テスト対象の問題がありません。"

    # 4. 最初の問題の最初のチェックボックスをクリック
    first_problem_first_check = driver.find_element(By.CSS_SELECTOR, ".problem-card .check-box")
    first_problem_first_check.click()
    time.sleep(0.5) # DOM更新のための短い待機

    # 5.【検証】チェック直後、問題数が変わらないことを確認
    problems_after_check = driver.find_elements(By.CLASS_NAME, "problem-card")
    assert len(problems_after_check) == initial_problem_count, "チェックした問題が即座に消えてしまいました。"

    # 6. ソート順を変更してリストを再描画させる
    Select(driver.find_element(By.ID, "sort-order")).select_by_value("ref-desc")
    time.sleep(0.5) # 再描画を待つ

    # 7.【検証】再描画後、問題数が1つ減っていることを確認
    problems_after_sort = driver.find_elements(By.CLASS_NAME, "problem-card")
    assert len(problems_after_sort) == initial_problem_count - 1, "再描画後に着手済み問題が非表示になりませんでした。"