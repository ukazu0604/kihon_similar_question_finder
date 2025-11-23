import { storage } from './js/storage.js';

(() => {
  let data = {};
  let referenceCounts = {}; // 各問題の被参照回数を格納する
  let categoryList = document.getElementById('category-list');
  let indexView = document.getElementById('index-view');
  let detailView = document.getElementById('detail-view');
  let modelInfo = document.getElementById('model-info');
  let oshiCounts = {}; // 推しカウントを保持するオブジェクト
  let likeCounts = {}; // いいねカウントを保持するオブジェクト
  let fearCounts = {}; // 恐怖カウントを保持するオブジェクト
  let problemChecks = {}; // チェック状態を保持するオブジェクト
  let currentSortOrder = storage.loadSortOrder('default'); // 現在の並び順（storageから読み込む）
  let showUntouchedOnly = false; // 「未着手のみ表示」フィルターの状態。デフォルトはfalse。

  async function loadData() {
    try {
      const res = await fetch('03_html_output/similar_results.json');
      data = await res.json();

      // **類似問題を同じ中分類のものだけにフィルタリングする**
      for (const middleCat in data.categories) {
        data.categories[middleCat].forEach(item => {
          item.similar_problems = item.similar_problems.filter(sim => {
            return sim.data.中分類 === item.main_problem.中分類;
          });
        });
      }

      modelInfo.textContent = `使用モデル: ${data.model || 'N/A'}`;
      
      // storageから状態を読み込む
      oshiCounts = storage.loadOshiCounts();
      likeCounts = storage.loadLikeCounts();
      fearCounts = storage.loadFearCounts();
      
      const loadedChecks = storage.loadChecks();
      const { migratedChecks, needsSave } = migrateChecks(loadedChecks);
      problemChecks = migratedChecks;
      if (needsSave) {
        storage.saveChecks(problemChecks); // 移行が発生した場合のみ保存
      }

      calculateReferenceCounts(data.categories);
      renderIndex(data.categories);
      renderTotalReactions(); // 全体のリアクション数を表示
      renderTotalReviewCount(); // 全体の復習数を表示
      renderTotalProgress(); // 全体の進捗を表示
    } catch (e) {
      modelInfo.textContent = 'データ読み込みエラー';
      console.error(e);
    }
  }

  // スマートフォンデバイスかどうかを判定する
  function isMobileDevice() {
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  }

  // 古いデータ構造（ブール値の配列）からの移行処理
  function migrateChecks(checks) {
    let needsSave = false;
    for (const problemId in checks) {
      if (Array.isArray(checks[problemId]) && typeof checks[problemId][0] === 'boolean') {
        needsSave = true;
        checks[problemId] = checks[problemId].map(isChecked => ({
          checked: isChecked,
          timestamp: isChecked ? Date.now() : null // 古いデータはとりあえず今の時刻で
        }));
      }
    }
    return { migratedChecks: checks, needsSave };
  }

  // 全体のリアクション数を計算して表示する
  function renderTotalReactions() {
    const totalOshi = Object.values(oshiCounts).reduce((sum, count) => sum + count, 0);
    const totalLike = Object.values(likeCounts).reduce((sum, count) => sum + count, 0);
    const totalFear = Object.values(fearCounts).reduce((sum, count) => sum + count, 0);

    const totalReactionsEl = document.getElementById('total-reactions');
    if (totalReactionsEl) {
      totalReactionsEl.innerHTML = `
          <span>❤️ ${totalOshi}</span> | <span>👍 ${totalLike}</span> | <span>😱 ${totalFear}</span>
        `;
    }
  }

  // 全体の進捗を計算して表示する
  function renderTotalProgress() {
    if (!data.categories) return;

    let totalProblems = 0; // 総問題数
    let totalCheckedCount = 0; // チェックされた総数

    for (const middleCat in data.categories) {
      const problems = data.categories[middleCat];
      totalProblems += problems.length;
      for (const item of problems) {
        const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
        const checks = problemChecks[problemId];
        if (checks) {
          checks.forEach(c => {
            if (c && c.checked) {
              totalCheckedCount++;
            }
          });
        }
      }
    }
    const completedProblemsEquivalent = totalCheckedCount / 4; // 4チェックで1問完了と換算
    const progressPercentage = totalProblems > 0 ? (completedProblemsEquivalent / totalProblems) * 100 : 0;

    const container = document.getElementById('total-progress-container');
    if (container) {
      container.innerHTML = `
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-bar-inner" style="width: ${progressPercentage.toFixed(2)}%;"></div>
          </div>
          <div class="progress-text">${completedProblemsEquivalent.toFixed(2)} / ${totalProblems} 問</div>
        </div>
      `;
    }
  }

  // 全体の復習数を計算して表示する
  function renderTotalReviewCount() {
    if (!data.categories) return;

    let totalReviewCount = 0;
    for (const middleCat in data.categories) {
      for (const item of data.categories[middleCat]) {
        const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
        if (shouldHighlightProblem(problemId)) {
          totalReviewCount++;
        }
      }
    }

    const totalReviewEl = document.getElementById('total-review-summary');
    if (totalReviewEl) {
      if (totalReviewCount > 0) {
        totalReviewEl.innerHTML = `<span class="review-count">🔥 ${totalReviewCount}</span>`;
      } else {
        totalReviewEl.innerHTML = `<span class="review-count" style="background: none; color: inherit;">😊</span>`;
      }
    }
  }

  function calculateReferenceCounts(categories) {
    referenceCounts = {}; // カウント結果を格納するオブジェクトを初期化

    // 中分類ごとにループ
    for (const middleCat in categories) {
      const problemsInCat = categories[middleCat];
      const countsInCat = {}; // この中分類内でのカウント用

      // この中分類内の各問題が持つ「類似問題リスト」をチェック
      problemsInCat.forEach(item => {
        item.similar_problems.forEach(sim => {
          // 類似度が50%以上のものだけをカウント対象にする
          if (sim.similarity >= 0.5) {
            const problemId = sim.data.問題番号;
            countsInCat[problemId] = (countsInCat[problemId] || 0) + 1;
          }
        });
      });

      // この中分類のカウント結果を保存
      referenceCounts[middleCat] = countsInCat;
    }
  }

  function renderIndex(categories) {
    // 大項目でグループ化
    const groupedByLargeCategory = {};
    for (const [middleCat, problems] of Object.entries(categories)) {
      if (problems.length > 0) {
        const largeCat = problems[0].main_problem.大項目;
        if (!groupedByLargeCategory[largeCat]) {
          groupedByLargeCategory[largeCat] = [];
        }
        groupedByLargeCategory[largeCat].push({ middleCat, problems });
      }
    }

    categoryList.innerHTML = '';
    // 大項目のキーでソートして表示
    Object.keys(groupedByLargeCategory).sort((a, b) => {
      // "1.基礎理論"のような文字列から先頭の数字を抜き出して比較する
      const numA = parseInt(a.split('.')[0], 10);
      const numB = parseInt(b.split('.')[0], 10);
      return numA - numB;
    }).forEach(largeCat => {
      // --- START: New logic for large category summary ---
      let largeCatTotalProblems = 0;
      let largeCatTotalChecked = 0;
      let largeCatTotalReviewItems = 0;

      groupedByLargeCategory[largeCat].forEach(({ problems }) => {
        largeCatTotalProblems += problems.length;
        problems.forEach(item => {
          const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
          // Calculate checked count for progress
          const checks = problemChecks[problemId];
          if (checks) {
            checks.forEach(c => {
              if (c && c.checked) largeCatTotalChecked++;
            });
          }
          // Calculate review items
          if (shouldHighlightProblem(problemId)) {
            largeCatTotalReviewItems++;
          }
        });
      });

      const largeCatCompletedEquivalent = largeCatTotalChecked / 4;
      const largeCatProgress = largeCatTotalProblems > 0 ? (largeCatCompletedEquivalent / largeCatTotalProblems) * 100 : 0;
      // --- END: New logic ---

      const largeCategorySection = document.createElement('div');
      largeCategorySection.className = 'major-category';

      // -- Reworked majorTitle --
      const majorTitle = document.createElement('div');
      majorTitle.className = 'major-title';
      majorTitle.dataset.largeCat = largeCat;
      majorTitle.style.display = 'flex';
      majorTitle.style.justifyContent = 'space-between';
      majorTitle.style.alignItems = 'center';

      const titleTextEl = document.createElement('span');
      titleTextEl.className = 'large-category-title-text';
      // title innerHTML will be set by expand/collapse logic later

      const summaryEl = document.createElement('div');
      summaryEl.className = 'large-category-summary';
      summaryEl.innerHTML = `
        <span class="progress-percentage">${largeCatProgress.toFixed(0)}%</span>
        ${largeCatTotalReviewItems > 0 ? `<span class="review-count">🔥 ${largeCatTotalReviewItems}</span>` : ''}
        <span class="problem-count">${largeCatTotalProblems}問</span>
      `;

      majorTitle.appendChild(titleTextEl);
      majorTitle.appendChild(summaryEl);
      largeCategorySection.appendChild(majorTitle);
      // -- End rework --

      const middleCategoryList = document.createElement('div');
      middleCategoryList.className = 'middle-category-list'; // クラスを追加
      groupedByLargeCategory[largeCat].forEach(({ middleCat, problems }) => {
        // カテゴリごとのリアクション合計を計算
        let totalOshi = 0;
        let totalLike = 0;
        let totalFear = 0;
        problems.forEach(item => {
          const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
          totalOshi += oshiCounts[problemId] || 0;
          totalLike += likeCounts[problemId] || 0;
          totalFear += fearCounts[problemId] || 0;
        });

        // このカテゴリの進捗を計算
        let checkedInCategory = 0;
        problems.forEach(item => {
          const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
          const checks = problemChecks[problemId];
          if (checks) {
            checks.forEach(c => {
              if (c && c.checked) {
                checkedInCategory++;
              }
            });
          }
        });
        const completedInCategoryEquivalent = checkedInCategory / 4;
        const categoryProgress = problems.length > 0 ? (completedInCategoryEquivalent / problems.length) * 100 : 0;
        const progressHtml = `<span class="progress-percentage">${categoryProgress.toFixed(0)}%</span>`;


        // このカテゴリにハイライトすべき問題があるかチェック
        let reviewItemCount = 0;
        for (const item of problems) {
          const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
          if (shouldHighlightProblem(problemId)) {
            reviewItemCount++;
          }
        }
        const hasReviewItems = reviewItemCount > 0;

        // 復習カウントのHTMLを生成
        let reviewCountHtml = ''; // デフォルトは空文字列
        if (hasReviewItems) {
          reviewCountHtml = `<span class="review-count">🔥 ${reviewItemCount}</span>`;
        }

        // 表示用のHTMLを生成
        const reactionSummaryHtml = `
            <div class="reaction-summary">
              <span>❤️ ${totalOshi}</span>
              <span>👍 ${totalLike}</span>
              <span>😱 ${totalFear}</span>
            </div>`;

        const item = document.createElement('div');
        item.className = 'middle-category-item';
        item.innerHTML = `
            <a href="#" class="middle-category-link ${hasReviewItems ? 'has-review-items' : ''}" data-cat="${middleCat}">
              <span class="category-name">${middleCat}</span>
              <div class="category-meta">
                ${progressHtml}
                ${reviewCountHtml}
                ${reactionSummaryHtml}
                <span class="problem-count">${problems.length}問</span>
                <span class="arrow">›</span>
              </div>
            </a>`;
        middleCategoryList.appendChild(item);
      });
      largeCategorySection.appendChild(middleCategoryList);
      categoryList.appendChild(largeCategorySection);
    });

    // --- Reworked event listeners ---
    // 大項目の開閉機能を追加
    document.querySelectorAll('.major-title').forEach(titleEl => {
      const largeCat = titleEl.dataset.largeCat;
      const titleTextEl = titleEl.querySelector('.large-category-title-text');
      const listEl = titleEl.nextElementSibling; // middle-category-listを指す
      // storageから開閉状態を復元。指定がなければ閉じた状態がデフォルト
      let isCollapsed = storage.isMajorCatCollapsed(largeCat);
      if (isCollapsed) {
        listEl.style.display = 'none';
        titleTextEl.innerHTML = `▶ ${largeCat}`; // 閉じた状態の矢印
      } else {
        listEl.style.display = ''; // デフォルト表示
        titleTextEl.innerHTML = `▼ ${largeCat}`; // 開いた状態の矢印
      }

      titleEl.addEventListener('click', () => {
        const currentlyCollapsed = listEl.style.display === 'none';
        if (currentlyCollapsed) {
          listEl.style.display = ''; // 表示
          titleTextEl.innerHTML = `▼ ${largeCat}`;
          storage.setMajorCatCollapsed(largeCat, false);
        } else {
          listEl.style.display = 'none'; // 非表示
          titleTextEl.innerHTML = `▶ ${largeCat}`;
          storage.setMajorCatCollapsed(largeCat, true);
        }
      });
    });

    // イベント設定
    document.querySelectorAll('.middle-category-link').forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const cat = e.currentTarget.dataset.cat;
        console.log(`[カテゴリクリック] カテゴリ「${cat}」がクリックされました。`);

        // URLにハッシュを追加して履歴に記録
        console.log(`[履歴操作] history.pushStateを実行します。ハッシュ: #${encodeURIComponent(cat)}`);
        history.pushState({ category: cat }, `詳細: ${cat}`, `#${encodeURIComponent(cat)}`);

        console.log(`[画面遷移] showDetail('${cat}', false) を呼び出します。`);
        showDetail(cat, false); // ユーザー操作なのでisPopStateはfalse
      });
    });
  }

  // 問題をハイライトすべきか判定する関数
  function shouldHighlightProblem(problemId) {
    const checks = problemChecks[problemId];
    if (!checks) return false;

    const now = Date.now();
    const reviewIntervals = [
      1 * 60 * 60 * 1000,   // 1時間
      1 * 24 * 60 * 60 * 1000,  // 1日
      6 * 24 * 60 * 60 * 1000,  // 6日
      Infinity // 4つ目はハイライトしない
    ];

    // 最後のチェックがどの段階かを見つける
    let lastCheckedIndex = -1;
    for (let i = checks.length - 1; i >= 0; i--) {
      if (checks[i] && checks[i].checked) {
        lastCheckedIndex = i;
        break;
      }
    }

    // どのチェックもされていない場合はハイライトしない
    if (lastCheckedIndex === -1) {
      return false;
    }

    // 最後のチェックのタイムスタンプと経過時間を取得
    const lastCheck = checks[lastCheckedIndex];
    const elapsedTime = now - lastCheck.timestamp;
    const requiredInterval = reviewIntervals[lastCheckedIndex];
    const shouldHighlight = elapsedTime > requiredInterval;

    // デバッグ用に時刻情報をコンソールに出力
    if (lastCheck.timestamp) { // タイムスタンプがある場合のみログ出力
      console.log(`[Highlight Check] Problem: ${problemId}`, {
        lastCheckedIndex: lastCheckedIndex,
        lastCheckTimestamp: new Date(lastCheck.timestamp).toLocaleString(),
        elapsedHours: (elapsedTime / (1000 * 60 * 60)).toFixed(2),
        requiredHours: (requiredInterval / (1000 * 60 * 60)).toFixed(2),
        shouldHighlight: shouldHighlight
      });
    }

    // 経過時間が指定の間隔を超えていればハイライト対象
    return shouldHighlight;
  }

  function showDetail(middleCat, isPopState = false) {
    indexView.style.display = 'none';
    detailView.style.display = 'block';
    document.getElementById('detail-title').textContent = middleCat; // isPopState引数を追加

    // ページ上部にスクロール
    if (!isPopState) { // popstateからの呼び出しでない場合のみスクロール
      window.scrollTo(0, 0);
    }
    const container = document.getElementById('detail-container');
    container.innerHTML = '';

    // 表示中の中分類に対応するカウント結果を取得
    const countsForThisCat = referenceCounts[middleCat] || {};

    // 要件1-2: 復習項目があれば自動で「復習優先」にソート
    const problemsForCheck = data.categories[middleCat];
    const hasReviewItems = problemsForCheck.some(item => {
      const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
      return shouldHighlightProblem(problemId);
    });

    if (hasReviewItems) {
      currentSortOrder = 'review-first';
      // 要件1-3: デバッグログ出力
      console.log(`[自動並び順変更] カテゴリ「${middleCat}」に復習項目があるため、並び順を「復習優先」に変更しました。`);
    } else {
      // 復習項目がない場合は、storageに保存された（またはデフォルトの）並び順を適用
      currentSortOrder = storage.loadSortOrder('default');
      console.log(`[並び順適用] カテゴリ「${middleCat}」に復習項目がないため、保存された設定「${currentSortOrder}」を適用します。`);
    }
    // ドロップダウンの表示を現在の並び順に合わせる
    document.getElementById('sort-order').value = currentSortOrder;

    // isPopState（リロードやブラウザバック）の場合のみstorageから状態を復元
    // 通常の画面遷移ではリセットする
    if (isPopState) {
      showUntouchedOnly = storage.loadShowUntouchedOnly();
      console.log(`[状態復元] リロードのため、「未着手のみ表示」の状態(${showUntouchedOnly})をstorageから復元しました。`);
    } else {
      showUntouchedOnly = false;
      console.log(`[状態リセット] 画面遷移のため、「未着手のみ表示」の状態をリセットしました。`);
    }

    renderProblemList(middleCat);

    // 「未着手のみ表示」チェックボックスのイベントリスナーを（再）設定
    const untouchedCheckbox = document.getElementById('show-untouched-only');
    // 既存のリスナーを削除して重複を防ぐ
    untouchedCheckbox.replaceWith(untouchedCheckbox.cloneNode(true));
    document.getElementById('show-untouched-only').addEventListener('change', e => {
      showUntouchedOnly = e.target.checked;
      storage.saveShowUntouchedOnly(showUntouchedOnly); // 状態を保存
      console.log(`[フィルター変更] 未着手のみ表示: ${showUntouchedOnly}`);
      renderProblemList(middleCat);
    });
    // 状態を復元
    document.getElementById('show-untouched-only').checked = showUntouchedOnly;
  }

  function renderProblemList(middleCat) {
    let problems = [...data.categories[middleCat]]; // ソートやフィルタリングのためにコピーを作成
    const countsForThisCat = referenceCounts[middleCat] || {};

    // 「未着手のみ表示」フィルターを適用
    if (showUntouchedOnly) {
      problems = problems.filter(item => isProblemUntouched(item));
    }
    // 選択された並び順に応じてソート
    if (currentSortOrder === 'review-first') {
      problems.sort((a, b) => {
        const aId = `${a.main_problem.出典}-${a.main_problem.問題番号}`;
        const bId = `${b.main_problem.出典}-${b.main_problem.問題番号}`;
        const aNeedsReview = shouldHighlightProblem(aId);
        const bNeedsReview = shouldHighlightProblem(bId);

        if (aNeedsReview !== bNeedsReview) {
          return bNeedsReview - aNeedsReview; // true (1) が先に来るように降順ソート
        }
        return a.main_problem.問題番号 - b.main_problem.問題番号; // 復習ステータスが同じ場合は問題番号順
      });
    } else if (currentSortOrder === 'ref-desc') {
      problems.sort((a, b) => {
        const countA = countsForThisCat[a.main_problem.問題番号] || 0;
        const countB = countsForThisCat[b.main_problem.問題番号] || 0;
        return countB - countA; // 降順
      });
    } else if (currentSortOrder === 'oshi-desc') {
      problems.sort((a, b) => {
        const countA = oshiCounts[`${a.main_problem.出典}-${a.main_problem.問題番号}`] || 0;
        const countB = oshiCounts[`${b.main_problem.出典}-${b.main_problem.問題番号}`] || 0;
        return countB - countA; // 降順
      });
    } else if (currentSortOrder === 'like-desc') {
      problems.sort((a, b) => {
        const countA = likeCounts[`${a.main_problem.出典}-${a.main_problem.問題番号}`] || 0;
        const countB = likeCounts[`${b.main_problem.出典}-${b.main_problem.問題番号}`] || 0;
        return countB - countA; // 降順
      });
    } else if (currentSortOrder === 'fear-desc') {
      problems.sort((a, b) => {
        const countA = fearCounts[`${a.main_problem.出典}-${a.main_problem.問題番号}`] || 0;
        const countB = fearCounts[`${b.main_problem.出典}-${b.main_problem.問題番号}`] || 0;
        return countB - countA; // 降順
      });
    } else { // default
      problems.sort((a, b) => {
        // 問題番号が数値なので、数値として比較する
        return a.main_problem.問題番号 - b.main_problem.問題番号; // 昇順
      });
    }

    const container = document.getElementById('detail-container');
    container.innerHTML = '';
    problems.forEach(item => {
      const main = item.main_problem;
      let mainProblemLink = main.リンク;
      if (isMobileDevice()) {
        // スマートフォン版のURLに変換
        mainProblemLink = mainProblemLink.replace('https://www.fe-siken.com/', 'https://www.fe-siken.com/s/');
      }

      const card = document.createElement('div');

      const mainProblemUniqueId = `${main.出典}-${main.問題番号}`;

      // ハイライト判定
      const needsReview = shouldHighlightProblem(mainProblemUniqueId);

      // チェックボックスのHTMLを生成
      let checksHtml = '<div class="check-container">';
      for (let i = 0; i < 4; i++) {
        const checkData = problemChecks[mainProblemUniqueId]?.[i];
        const isChecked = checkData && checkData.checked;
        checksHtml += `<div class="check-box ${isChecked ? 'checked c' + i : ''}" data-problem-id="${mainProblemUniqueId}" data-check-index="${i}"></div>`;
      }
      checksHtml += '</div>';

      // リアクションボタンのHTMLを生成
      const mainOshiCount = oshiCounts[mainProblemUniqueId] || 0;
      const mainLikeCount = likeCounts[mainProblemUniqueId] || 0;
      const mainFearCount = fearCounts[mainProblemUniqueId] || 0;
      const reactionHtml = `
          <div class="reaction-container">
            <button class="reaction-button" data-problem-id="${mainProblemUniqueId}" data-reaction-type="oshi">❤️</button>
            <span class="reaction-count">${mainOshiCount}</span>
            <button class="reaction-button" data-problem-id="${mainProblemUniqueId}" data-reaction-type="like">👍</button>
            <span class="reaction-count">${mainLikeCount}</span>
            <button class="reaction-button" data-problem-id="${mainProblemUniqueId}" data-reaction-type="fear">😱</button>
            <span class="reaction-count">${mainFearCount}</span>
          </div>`;

      card.className = `problem-card ${needsReview ? 'needs-review' : ''}`;
      let html = `
          <a href="${mainProblemLink}" target="_blank" class="problem-panel main-problem">
            <div class="problem-number">問題: ${main.問題番号}</div>
            <div class="problem-title">${main.問題名}</div>
            <div class="problem-source">出典: ${main.出典} ${reactionHtml}</div>
            ${checksHtml}
          </a>
        `;
      // 類似度が50%以上のものだけをフィルタリング
      const filteredSimilars = item.similar_problems
        ? item.similar_problems.filter(sim => sim.similarity >= 0.5)
        : [];

      if (filteredSimilars.length > 0) {
        const similarCount = filteredSimilars.length;
        // 平均類似度を計算
        const totalSimilarity = filteredSimilars.reduce((sum, sim) => sum + sim.similarity, 0);
        const averageSimilarity = (totalSimilarity / similarCount) * 100;

        html += `
            <div class="similar-section">
              <div class="similar-toggle">
                <span class="similar-title">📊 類似問題 (${similarCount > 5 ? '上位5' : similarCount}件)</span>
                <span class="average-similarity">平均: ${averageSimilarity.toFixed(1)}%</span>
                <span class="toggle-arrow">▼</span> <!-- 矢印を右端に -->
              </div>
              <div class="similar-content" style="display: none;">
          `;
        filteredSimilars.slice(0, 5).forEach(sim => {
          const s = sim.data;
          let similarProblemLink = s.リンク;
          if (isMobileDevice()) {
            // スマートフォン版のURLに変換
            similarProblemLink = similarProblemLink.replace('https://www.fe-siken.com/', 'https://www.fe-siken.com/s/');
          }

          // 類似問題用のチェックボックスHTMLを生成
          const simProblemUniqueId = `${s.出典}-${s.問題番号}`;
          let simChecksHtml = '<div class="check-container">';
          for (let i = 0; i < 4; i++) {
            const isChecked = problemChecks[simProblemUniqueId]?.[i]?.checked;
            simChecksHtml += `<div class="check-box ${isChecked ? 'checked c' + i : ''}" data-problem-id="${simProblemUniqueId}" data-check-index="${i}"></div>`;
          }
          simChecksHtml += '</div>';

          // 類似問題用のリアクションボタンHTMLを生成
          const simOshiCount = oshiCounts[simProblemUniqueId] || 0;
          const simLikeCount = likeCounts[simProblemUniqueId] || 0;
          const simFearCount = fearCounts[simProblemUniqueId] || 0;
          const simReactionHtml = `
              <div class="reaction-container">
                <button class="reaction-button" data-problem-id="${simProblemUniqueId}" data-reaction-type="oshi">❤️</button>
                <span class="reaction-count">${simOshiCount}</span>
                <button class="reaction-button" data-problem-id="${simProblemUniqueId}" data-reaction-type="like">👍</button>
                <span class="reaction-count">${simLikeCount}</span>
                <button class="reaction-button" data-problem-id="${simProblemUniqueId}" data-reaction-type="fear">😱</button>
                <span class="reaction-count">${simFearCount}</span>
              </div>`;

          html += `
              <a href="${similarProblemLink}" target="_blank" class="problem-panel similar-item">
                <span class="similarity-badge">${(sim.similarity * 100).toFixed(1)}%</span>
                <div class="problem-number">問題: ${s.問題番号}</div>
                <div class="problem-title">${s.問題名}</div>
                <div class="problem-source">出典: ${s.出典} ${simReactionHtml}</div>
                <div class="problem-meta">被参照: ${countsForThisCat[s.問題番号] || 0}回</div>
                ${simChecksHtml}
              </a>
            `;
        });
        html += `
              </div>
            </div>
          `;
      }
      card.innerHTML = html;
      container.appendChild(card);
    });

    // 新しく生成したアコーディオン要素にイベントリスナーを設定
    document.querySelectorAll('.similar-toggle').forEach(toggle => {
      toggle.addEventListener('click', () => {
        const content = toggle.nextElementSibling;
        const arrow = toggle.querySelector('.toggle-arrow');
        if (content.style.display === 'none' || content.style.display === '') {
          content.style.display = 'block';
          arrow.textContent = '▲';
        } else {
          content.style.display = 'none';
          arrow.textContent = '▼';
        }
      });
    });

    // 新しく生成したチェックボックスにイベントリスナーを設定
    document.querySelectorAll('.check-box').forEach(box => {
      box.addEventListener('click', e => {
        e.preventDefault(); // aタグのリンク遷移を防止
        e.stopPropagation(); // 親要素へのイベント伝播を停止

        const problemId = e.target.dataset.problemId;
        const checkIndex = parseInt(e.target.dataset.checkIndex, 10);

        // チェック状態の初期化
        if (!problemChecks[problemId]) {
          problemChecks[problemId] = Array(4).fill(null).map(() => ({ checked: false, timestamp: null }));
        }

        // 状態をトグル
        const currentCheck = problemChecks[problemId][checkIndex];
        const newCheckedState = !currentCheck.checked;

        const newTimestamp = newCheckedState ? Date.now() : null;
        if (newTimestamp) {
          console.log(`[Check ON] Problem: ${problemId}, Index: ${checkIndex}, Timestamp: ${new Date(newTimestamp).toLocaleString()}`);
        }

        problemChecks[problemId][checkIndex] = {
          checked: newCheckedState,
          timestamp: newTimestamp
        };
        storage.saveChecks(problemChecks); // 変更を保存

        // 画面に表示されている同じ問題IDとインデックスを持つすべてのチェックボックスの表示を更新
        document.querySelectorAll(`.check-box[data-problem-id="${problemId}"][data-check-index="${checkIndex}"]`).forEach(boxToUpdate => {
          if (newCheckedState) {
            boxToUpdate.classList.add('checked', 'c' + checkIndex);
          } else {
            boxToUpdate.classList.remove('checked', 'c' + checkIndex);
          }
        });

        // ハイライト状態もリアルタイムで更新
        const needsReview = shouldHighlightProblem(problemId);
        document.querySelectorAll(`.problem-card`).forEach(card => {
          const panel = card.querySelector(`.problem-panel[data-problem-id="${problemId}"]`);
          if (!panel) return; // 関係ないカードはスキップ
          card.classList.toggle('needs-review', needsReview);
        });

        // 全体の復習数のみ更新（トップページに戻った時にカテゴリ一覧は再描画される）
        renderTotalReviewCount();
        renderTotalProgress();
        // 「未着手のみ表示」がONの場合、リストを再描画して着手済みの項目を消す
        // if (showUntouchedOnly) renderProblemList(document.getElementById('detail-title').textContent);
      });
    });

    // 新しく生成したリアクションボタンにイベントリスナーを設定
    document.querySelectorAll('.reaction-button').forEach(button => {
      button.addEventListener('click', e => {
        e.preventDefault(); // aタグのリンク遷移を防止
        e.stopPropagation(); // 親要素へのイベント伝播を停止

        const problemId = e.target.dataset.problemId;
        const reactionType = e.target.dataset.reactionType;

        if (reactionType === 'oshi') {
          oshiCounts[problemId] = (oshiCounts[problemId] || 0) + 1;
          storage.saveOshiCounts(oshiCounts);
        } else if (reactionType === 'like') {
          likeCounts[problemId] = (likeCounts[problemId] || 0) + 1;
          storage.saveLikeCounts(likeCounts);
        } else if (reactionType === 'fear') {
          fearCounts[problemId] = (fearCounts[problemId] || 0) + 1;
          storage.saveFearCounts(fearCounts);
        }

        // 画面に表示されている同じ問題IDを持つすべてのカウント表示を更新
        document.querySelectorAll(`.reaction-button[data-problem-id="${problemId}"][data-reaction-type="${reactionType}"]`).forEach(btnToUpdate => {
          const countElement = btnToUpdate.nextElementSibling;
          if (countElement && countElement.classList.contains('reaction-count')) {
            countElement.textContent = (reactionType === 'oshi' ? oshiCounts[problemId] : reactionType === 'like' ? likeCounts[problemId] : fearCounts[problemId]);
          }
        });

        // 全体の合計数も更新
        renderTotalReactions();
      });
    });
  }

  function showIndex(isPopState = false) { // isPopState引数を追加
    detailView.style.display = 'none';
    indexView.style.display = 'block';
    if (!isPopState) { // popstateからの呼び出しでない場合のみスクロール
      // トップページに戻る際は、必ず最新の状態でカテゴリ一覧を再描画する
      renderIndex(data.categories);
      renderTotalReviewCount();
      renderTotalProgress();
      window.scrollTo(0, 0);
    }

    // トップページに復習項目がある場合、最初の復習項目までスクロールする
    // 描画が完了するのを待つために少し遅延させる
    setTimeout(() => {
      const firstReviewCategory = document.querySelector('.middle-category-link.has-review-items');
      if (firstReviewCategory) {
        // isPopStateがtrue（ブラウザバックなど）の場合はスクロール位置が復元されるため、
        // ユーザーの明示的な操作がない場合のみスクロールする
        if (!isPopState) firstReviewCategory.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 100);
  }

  // 問題が未着手かどうかを判定するヘルパー関数
  function isProblemUntouched(item) {
    const problemId = `${item.main_problem.出典}-${item.main_problem.問題番号}`;
    const checks = problemChecks[problemId];

    // チェック情報が存在しない場合は未着手
    if (!checks) return true;

    // checks配列のいずれかのcheckedプロパティがtrueであれば着手済み
    const isTouched = checks.some(c => c && c.checked);

    // isTouchedがtrueなら着手済みなので、その逆（false）を返す
    return !isTouched;
  }

  // ブラウザの戻る/進むボタンが押されたときの処理
  window.addEventListener('popstate', e => {
    const hash = location.hash.substring(1);
    if (hash) {
      showDetail(decodeURIComponent(hash), true); // popstateからの呼び出しなのでtrue
    } else {
      // ブラウザの戻るボタンでトップに来た時も再描画
      renderIndex(data.categories);
      renderTotalReviewCount();
      renderTotalProgress();
      showIndex(true); // popstateからの呼び出しなのでtrue
    }
  });

  document.getElementById('back-button').addEventListener('click', e => {
    e.preventDefault();
    history.back(); // ブラウザの「戻る」と同じ動作
  });

  // 並び順の変更イベント
  document.getElementById('sort-order').addEventListener('change', e => {
    currentSortOrder = e.target.value;
    console.log(`[並び順変更] ユーザーが手動で「${currentSortOrder}」を選択しました。`);
    // 要件1-1: 並び順をstorageに保存
    storage.saveSortOrder(currentSortOrder);
    const currentMiddleCat = document.getElementById('detail-title').textContent;
    renderProblemList(currentMiddleCat);
  });

  // ローカルストレージリセットボタンのイベントリスナー
  document.getElementById('reset-storage-button').addEventListener('click', () => {
    // 誤操作防止のために確認ダイアログを表示
    if (confirm('すべてのチェック状態とリアクションをリセットします。よろしいですか？')) {
      storage.resetAll();
      
      // グローバル変数をリセット
      problemChecks = {};
      oshiCounts = {};
      likeCounts = {};
      fearCounts = {};

      renderTotalReactions(); // 表示を0に更新
      
      // 現在表示されているのが詳細ページならインデックスを再描画
      if (detailView.style.display === 'block') {
        const currentMiddleCat = document.getElementById('detail-title').textContent;
        showDetail(currentMiddleCat, false);
      } else {
        renderIndex(data.categories);
      }
      alert('すべての状態がリセットされました。');
    }
  });

  // ページの初期化処理
  async function initializePage() {
    await loadData(); // データの読み込みを待つ
    // 初期読み込み時にハッシュがあれば詳細ページを表示
    const initialHash = location.hash.substring(1);
    if (initialHash) {
      renderTotalProgress(); // 詳細ページ直アクセスでもプログレスバーは表示
      showDetail(decodeURIComponent(initialHash), true); // リロード時はスクロール位置を復元するため isPopState=true
    }
    // 初期表示がトップページの場合、復習項目までスクロール
    if (!initialHash) {
      setTimeout(() => {
        const firstReviewCategory = document.querySelector('.middle-category-link.has-review-items');
        if (firstReviewCategory) {
          firstReviewCategory.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 100);
    }
  }
  initializePage();
})();