import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://www.fe-siken.com/"

def scrape_website(url):
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'lxml')


    scraped_data = []

    # 左側のメニューから大項目と中項目のマッピングを作成
    category_map = {}
    for dl in soup.select('#tree dl'):
        major_category_tag = dl.find('dt')
        if not major_category_tag:
            continue
        # dtタグのテキストを取得し、ddタグのテキストを削除
        major_category = major_category_tag.get_text(strip=True)
        dd_text = major_category_tag.find_next_sibling('dd').get_text(strip=True)
        major_category = major_category.replace(dd_text, '').strip()

        for a in dl.select('dd a'):
            href = a.get('href')
            if not href or not href.startswith('#'):
                continue
            table_id = href.replace('#', '')
            minor_category = a.get_text(strip=True).split('(')[0].strip()
            category_map[table_id] = {
                'major_category': major_category,
                'minor_category': minor_category
            }

    # 右側の問題テーブルをすべて取得
    tables = soup.find_all('table', class_='qtable')

    for table in tables:
        table_id = table.get('id')
        if not table_id or table_id not in category_map:
            continue

        major_category = category_map[table_id]['major_category']
        minor_category = category_map[table_id]['minor_category']

        # テーブルから直接tdタグを取得
        cols = table.find_all('td')
        # 3つずつにグループ化
        for i in range(0, len(cols), 3):
            if i + 2 < len(cols):
                question_num = cols[i].get_text(strip=True)
                question_title_tag = cols[i+1].find('a')
                question_title = question_title_tag.get_text(strip=True) if question_title_tag else ''
                question_link = BASE_URL + question_title_tag['href'] if question_title_tag and question_title_tag.has_attr('href') else ''
                source = cols[i+2].get_text(strip=True)

                scraped_data.append({
                    '大項目': major_category,
                    '中項目': minor_category,
                    '問題番号': question_num,
                    '問題名': question_title,
                    'リンク': question_link,
                    '出典': source
                })

    return pd.DataFrame(scraped_data)

if __name__ == "__main__":
    target_urls = [
        "https://www.fe-siken.com/index_te.html",
        "https://www.fe-siken.com/index_ma.html",
        "https://www.fe-siken.com/index_st.html",
    ]
    
    all_data = []

    for url in target_urls:
        print(f"Scraping {url}...")
        df = scrape_website(url)
        if df is not None and not df.empty:
            all_data.append(df)
        else:
            print(f"No data found for {url}.")
        time.sleep(1) # サーバーへの負荷を軽減

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        output_filename = "fe_siken_all_items.csv"
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"All data successfully scraped and saved to {output_filename}")
    else:
        print("Failed to scrape any data.")
