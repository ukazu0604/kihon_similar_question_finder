# Web Scraper

このプロジェクトは、指定されたWebサイトから情報をスクレイピングし、CSVファイルに保存するための汎用的なWebスクレイパーです。

## セットアップ

1.  このリポジトリをクローンします。
    ```bash
    git clone <repository_url>
    cd web-scraper
    ```
2.  仮想環境を作成し、アクティベートします（推奨）。
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## 使い方

1.  `main.py`を開き、`target_url`変数をスクレイピングしたいURLに設定します。
2.  スクレイピングしたい具体的な要素に合わせて、`scrape_website`関数内のスクレイピングロジックを修正します。
3.  スクリプトを実行します。
    ```bash
    python main.py
    ```
4.  スクレイピングされたデータは`scraped_data.csv`として保存されます。

## カスタマイズ

-   `scrape_website`関数内の`soup.find_all()`や`soup.select()`などを使用して、特定のHTML要素をターゲットにします。
-   抽出したデータを`pd.DataFrame`に追加する前に、必要に応じて整形します。
-   複数のページをスクレイピングする場合は、ループ処理を追加します。
