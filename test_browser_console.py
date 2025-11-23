import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:8000/"

def wait_for_page_load(driver):
    """
    app.jsによってコンテンツが描画されるのを待つヘルパー関数。
    """
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#category-list > div"))
    )

def test_no_console_errors(driver):
    """
    ページを開いてコンソールにエラーが出てないか確認する。
    SEVEREレベルのエラーがある場合はテストを失敗させる。
    """
    driver.get(BASE_URL)
    wait_for_page_load(driver)
    time.sleep(1)  # スクリプト実行完了待ち

    logs = driver.get_log('browser')
    severe_errors = [log for log in logs if log['level'] == 'SEVERE']
    
    if severe_errors:
        messages = [err['message'] for err in severe_errors]
        pytest.fail(f"深刻なコンソールエラーが検出されました:\n{messages}")
    
    # 警告レベルもログに出力しておく（失敗はさせない）
    warnings = [log for log in logs if log['level'] == 'WARNING']
    if warnings:
        print(f"\n[WARNING] コンソール警告があります: {[w['message'] for w in warnings]}")

    assert not severe_errors
