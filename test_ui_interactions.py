import pytest
import time
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

BASE_URL = "http://localhost:8000/"

def wait_for_page_load(driver):
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#category-list > div"))
    )

def wait_for_detail_view(driver):
    WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.ID, "detail-view"))
    )

def reset_storage(driver):
    """ローカルストレージをリセットする"""
    driver.execute_script("localStorage.clear();")
    driver.refresh()
    wait_for_page_load(driver)

class TestUIInteractions:

    def _open_first_category(self, driver):
        """ヘルパー：最初の大項目を開く"""
        major_title = driver.find_element(By.CLASS_NAME, "major-title")
        list_el = major_title.find_element(By.XPATH, "following-sibling::div[contains(@class, 'middle-category-list')]")
        if not list_el.is_displayed():
            major_title.click()
            WebDriverWait(driver, 2).until(
                EC.visibility_of(list_el)
            )

    def test_navigation_to_detail_and_back(self, driver):
        """
        カテゴリをクリックして詳細画面に遷移し、戻るボタンで戻れることを確認
        """
        driver.get(BASE_URL)
        wait_for_page_load(driver)
        
        self._open_first_category(driver)

        # 最初のカテゴリをクリック
        category_link = driver.find_element(By.CSS_SELECTOR, ".middle-category-link")
        category_name = category_link.find_element(By.CLASS_NAME, "category-name").text
        category_link.click()
        
        # 詳細画面が表示されるのを待つ
        wait_for_detail_view(driver)
        
        # タイトルが正しいか確認
        detail_title = driver.find_element(By.ID, "detail-title").text
        assert detail_title == category_name, f"期待値: {category_name}, 実際: {detail_title}"
        
        # 戻るボタンをクリック
        driver.find_element(By.ID, "back-button").click()
        
        # 一覧画面に戻ったことを確認
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "index-view"))
        )

    def test_sort_order_persistence(self, driver):
        """
        ソート順を変更し、リロード後もその設定が維持されることを確認
        """
        driver.get(BASE_URL)
        wait_for_page_load(driver)
        
        self._open_first_category(driver)

        # 詳細画面へ移動
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        # ソート順を変更 (例: 恐怖カウント順)
        select = Select(driver.find_element(By.ID, "sort-order"))
        select.select_by_value("fear-desc")
        time.sleep(0.5) 
        
        # リロード
        driver.refresh()
        wait_for_detail_view(driver)
        
        # 選択状態が維持されているか確認
        select_after = Select(driver.find_element(By.ID, "sort-order"))
        assert select_after.first_selected_option.get_attribute("value") == "fear-desc"

    def test_reaction_buttons_persistence(self, driver):
        """
        リアクションボタン（いいね、推し、恐怖）のクリックと永続化の確認
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        self._open_first_category(driver)

        # 詳細画面へ移動
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        # 最初の問題の「推し」ボタンを取得
        oshi_btn = driver.find_element(By.CSS_SELECTOR, ".reaction-button[data-reaction-type='oshi']")
        count_span = oshi_btn.find_element(By.XPATH, "following-sibling::span")
        
        initial_count = int(count_span.text)
        
        # クリック
        oshi_btn.click()
        time.sleep(0.1)
        
        # カウントが増えているか
        new_count = int(count_span.text)
        assert new_count == initial_count + 1
        
        # リロードしても維持されているか
        driver.refresh()
        wait_for_detail_view(driver)
        
        oshi_btn_after = driver.find_element(By.CSS_SELECTOR, ".reaction-button[data-reaction-type='oshi']")
        count_span_after = oshi_btn_after.find_element(By.XPATH, "following-sibling::span")
        assert int(count_span_after.text) == new_count

    def test_checkbox_progress(self, driver):
        """
        チェックボックスをクリックして進捗バーが更新されるか確認
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        self._open_first_category(driver)

        # 詳細画面へ移動
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        # 最初のチェックボックスをクリック
        checkbox = driver.find_element(By.CSS_SELECTOR, ".check-box")
        checkbox.click()
        time.sleep(0.1)
        
        # チェック状態のクラスが付与されたか
        assert "checked" in checkbox.get_attribute("class")
        
        # 戻るボタンで一覧へ
        driver.find_element(By.ID, "back-button").click()
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, "index-view"))
        )
        
        # 全体の進捗バーが0%でないことを確認
        progress_bar = driver.find_element(By.CSS_SELECTOR, "#total-progress-container .progress-bar-inner")
        width_style = progress_bar.get_attribute("style")
        assert "width: 0.00%" not in width_style and "width: 0%" not in width_style

    def test_accordion_toggle(self, driver):
        """
        類似問題のアコーディオン開閉確認
        """
        driver.get(BASE_URL)
        wait_for_page_load(driver)
        
        self._open_first_category(driver)

        # 詳細画面へ移動
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        # 類似問題があるカードを探す
        toggles = driver.find_elements(By.CLASS_NAME, "similar-toggle")
        if not toggles:
            pytest.skip("類似問題があるカードが見つかりません")
            
        toggle = toggles[0]
        content = toggle.find_element(By.XPATH, "following-sibling::div[@class='similar-content']")
        
        # 初期状態は非表示
        assert not content.is_displayed()
        
        # クリックして表示
        toggle.click()
        time.sleep(0.5)
        assert content.is_displayed()
        
        # もう一度クリックして非表示
        toggle.click()
        time.sleep(0.5)
        assert not content.is_displayed()

    def test_large_category_accordion(self, driver):
        """
        大項目のアコーディオン開閉と永続化の確認
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        # 最初の大項目を取得
        major_title = driver.find_element(By.CLASS_NAME, "major-title")
        list_el = major_title.find_element(By.XPATH, "following-sibling::div[contains(@class, 'middle-category-list')]")
        
        # 初期状態は非表示（デフォルトで閉じるため）
        assert not list_el.is_displayed()
        
        # クリックして表示する
        major_title.click()
        WebDriverWait(driver, 2).until(EC.visibility_of(list_el))
        assert list_el.is_displayed()
        
        # リロードして状態が維持されているか確認
        driver.refresh()
        wait_for_page_load(driver)
        
        # 要素を再取得
        major_title_after = driver.find_element(By.CLASS_NAME, "major-title")
        list_el_after = major_title_after.find_element(By.XPATH, "following-sibling::div[contains(@class, 'middle-category-list')]")
        
        # 開いた状態が維持されているはず
        assert list_el_after.is_displayed()

        # もう一度クリックして閉じる
        major_title_after.click()
        WebDriverWait(driver, 2).until(EC.invisibility_of_element(list_el_after))
        assert not list_el_after.is_displayed()

    def test_untouched_filter(self, driver):
        """
        未着手のみ表示フィルターの動作確認
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        self._open_first_category(driver)

        # 詳細画面へ移動
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        # 全ての問題数を取得
        all_cards = driver.find_elements(By.CLASS_NAME, "problem-card")
        total_count = len(all_cards)
        
        if total_count == 0:
            pytest.skip("問題がありません")

        # 1つチェックする
        checkbox = all_cards[0].find_element(By.CSS_SELECTOR, ".check-box")
        checkbox.click()
        
        # フィルターをONにする
        filter_check = driver.find_element(By.ID, "show-untouched-only")
        filter_check.click()
        time.sleep(0.5)
        
        # 問題数が減っているか確認
        visible_cards = driver.find_elements(By.CLASS_NAME, "problem-card")
        assert len(visible_cards) == total_count - 1

    def test_review_highlight_logic(self, driver):
        """
        復習タイミングが来た問題がハイライトされるか確認
        localStorageを直接操作して過去のチェック状態を作り出す
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        self._open_first_category(driver)
        
        # ターゲットとなる問題IDを取得するために一度詳細を開く
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        
        target_card = driver.find_element(By.CLASS_NAME, "problem-card")
        # data-problem-idを取得 (check-boxから取得するのが手っ取り早い)
        checkbox = target_card.find_element(By.CSS_SELECTOR, ".check-box")
        problem_id = checkbox.get_attribute("data-problem-id")
        
        # 1時間以上前のタイムスタンプを作成 (1時間 + 1分前)
        past_timestamp = (time.time() * 1000) - (61 * 60 * 1000)
        
        # localStorageにデータを注入
        # problemChecks = { "出典-番号": [ {checked: true, timestamp: ...}, ... ] }
        check_data = {
            problem_id: [
                {"checked": True, "timestamp": past_timestamp}, # 1つ目: 1時間で復習
                {"checked": False, "timestamp": None},
                {"checked": False, "timestamp": None},
                {"checked": False, "timestamp": None}
            ]
        }
        
        driver.execute_script(f"localStorage.setItem('problemChecks', JSON.stringify({json.dumps(check_data)}));")
        
        # リロードして反映させる
        driver.refresh()
        wait_for_detail_view(driver)
        
        # 対象のカードが 'needs-review' クラスを持っているか確認
        # 要素を再取得
        target_card_after = driver.find_element(By.XPATH, f"//div[contains(@class, 'check-box') and @data-problem-id='{problem_id}']/ancestor::div[contains(@class, 'problem-card')]")
        
        assert "needs-review" in target_card_after.get_attribute("class")

    def test_reset_storage(self, driver):
        """
        リセットボタンの動作確認
        """
        driver.get(BASE_URL)
        reset_storage(driver)
        
        self._open_first_category(driver)

        # 何か操作して保存させる（例：推しボタン）
        driver.find_element(By.CSS_SELECTOR, ".middle-category-link").click()
        wait_for_detail_view(driver)
        driver.find_element(By.CSS_SELECTOR, ".reaction-button[data-reaction-type='oshi']").click()
        
        # トップに戻る
        driver.find_element(By.ID, "back-button").click()
        
        # リセットボタンをクリック
        reset_btn = driver.find_element(By.ID, "reset-storage-button")
        
        # confirmダイアログが出るのでOKを押す
        driver.execute_script("window.confirm = function(){return true;}")
        reset_btn.click()
        time.sleep(0.5)
        
        # アラートが出るのでOKを押す
        try:
            alert = driver.switch_to.alert
            alert.accept()
        except:
            pass
            
        # リアクション数がリセットされているか確認
        total_reactions = driver.find_element(By.ID, "total-reactions").text
        assert "❤️ 0" in total_reactions
