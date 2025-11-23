import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:8000/"

@pytest.fixture(scope="module")
def mobile_driver():
    """モバイル端末のUser-Agentを設定したWebDriver"""
    chrome_options = Options()
    # iPhone X の User-Agent
    mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1"
    chrome_options.add_argument(f"--user-agent={mobile_ua}")
    chrome_options.add_argument("--window-size=375,812") # iPhone X size
    chrome_options.add_argument("--log-level=3")
    
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    yield driver
    driver.quit()

def wait_for_page_load(driver):
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#category-list > div"))
    )

def open_first_category(driver):
    """最初の大項目を開いてカテゴリを表示する"""
    major_title = driver.find_element(By.CLASS_NAME, "major-title")
    list_el = major_title.find_element(By.XPATH, "following-sibling::div[contains(@class, 'middle-category-list')]")
    if not list_el.is_displayed():
        major_title.click()
        WebDriverWait(driver, 2).until(
            EC.visibility_of(list_el)
        )

def test_mobile_link_conversion(mobile_driver):
    """モバイル表示時にリンクがスマートフォン版URLに変換されるか確認"""
    driver = mobile_driver
    driver.get(BASE_URL)
    wait_for_page_load(driver)
    
    open_first_category(driver)
    
    # 詳細画面へ移動
    driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "detail-view"))
    )
    
    # 問題カードのリンクを確認
    # .problem-panel.main-problem は <a> タグ
    problem_link = driver.find_element(By.CSS_SELECTOR, ".problem-panel.main-problem").get_attribute("href")
    
    # PC版: https://www.fe-siken.com/kakomon/...
    # SP版: https://www.fe-siken.com/s/kakomon/...
    # 外部サイトへのリンクなので、href属性値そのものをチェックする
    assert "/s/" in problem_link, f"モバイル版URLに変換されていません: {problem_link}"

    # 類似問題のリンクも確認（もしあれば）
    similar_items = driver.find_elements(By.CLASS_NAME, "similar-item")
    if similar_items:
        sim_link = similar_items[0].get_attribute("href")
        assert "/s/" in sim_link, f"類似問題のリンクがモバイル版URLに変換されていません: {sim_link}"
