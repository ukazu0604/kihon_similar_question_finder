import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

@pytest.fixture(scope="module")
def driver():
    """
    Selenium WebDriverのインスタンスを提供するpytest fixture。
    テストモジュール実行中に一度だけブラウザを起動し、終了時に閉じる。
    """
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # CI/CD環境など、UIなしで実行する場合に有効化
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=3") # Seleniumの冗長なログを抑制
    # コンソールログを取得するための設定
    chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

    service = ChromeService(ChromeDriverManager().install())
    web_driver = webdriver.Chrome(service=service, options=chrome_options)
    
    yield web_driver # テスト関数にWebDriverインスタンスを渡す
    
    web_driver.quit() # テストモジュール終了後にブラウザを閉じる