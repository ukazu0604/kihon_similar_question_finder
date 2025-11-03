
# 基本情報 類似問題検索プロジェクト (kihon_similar_question_finder)

このプロジェクトは、基本情報技術者試験の過去問データを収集し、テキストのベクトル化技術を用いて類似した問題を見つけ出すことを目的としています。

## ディレクトリ構成

プロジェクトは、処理の工程ごとにディレクトリが分かれています。

- `01_scraping/`: Webサイトから過去問データを収集し、CSVファイルとして保存する工程。
- `02_vectorize/`: 収集した問題データのテキストを、複数のAIモデルを用いてベクトル化する工程。
- `03_search/` (予定): 生成したベクトルデータを用いて、類似問題の検索を行う工程。

## 実行手順

### Step 1: データ収集 (Scraping)

まず、問題データをWebサイトから収集します。

1.  `01_scraping` ディレクトリに移動します。
    ```bash
    cd 01_scraping
    ```
2.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```
3.  スクリプトを実行します。
    ```bash
    py main.py
    ```
4.  成功すると、`fe_siken_all_items.csv` というファイルが生成されます。これがステップ2の入力データになります。

### Step 2: ベクトル化 (Vectorization)

次に、収集したデータの「問題名」をベクトルに変換します。

1.  `02_vectorize` ディレクトリに移動します。
    ```bash
    cd ../02_vectorize
    ```
2.  必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ollamaアプリケーションを起動します。** (Ollamaモデルを使用するために必要です)

4.  スクリプトを実行します。
    ```bash
    # すべてのモデルを差分実行（未実行のモデルのみ処理）
    py main.py
    ```
    - **補足**: `sentence-transformers`のモデルは、初回実行時にインターネットからモデルデータ（数GB）をダウンロードするため、時間がかかる場合があります。

5.  （オプション）特定のモデルだけを実行または再実行したい場合
    ```bash
    # 例: mxbai-embed-large モデルだけを強制的に再実行
    py main.py --model mxbai-embed-large --force
    ```

6.  成功すると、`output/` ディレクトリに以下のファイルが生成されます。
    - `metadata.json`: 問題のメタデータ（全モデル共通）
    - `vectors_xxxx.npy`: 各モデルのベクトルデータ

### 設定の変更

ベクトル化するモデルを追加・変更したい場合は、`02_vectorize/config.yaml` ファイルを編集してください。

### Step 2-1: テストの実行 (任意)

ベクトル化処理の動作を確認するために、テストを実行することができます。

1.  `02_vectorize/tests` ディレクトリに移動します。
    ```bash
    cd 02_vectorize/tests
    ```
2.  テスト実行用のスクリプトを実行します。
    ```bash
    py run_tests_with_clipboard.py
    ```
    - **単体テスト (`test_vectorize.py`)**: Ollamaの起動は不要です。
    - **結合テスト (`test_integration.py`)**: Ollamaアプリケーションを起動し、`config.yaml` に記載のモデルをインストールしておく必要があります。
    - テストが失敗した場合、エラー内容が自動でクリップボードにコピーされます。
