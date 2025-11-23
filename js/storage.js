// js/storage.js

// Generic helper functions
function loadJSON(key, defaultValue = {}) {
  const stored = localStorage.getItem(key);
  return stored ? JSON.parse(stored) : defaultValue;
}

function saveJSON(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function load(key, defaultValue = null) {
  return localStorage.getItem(key) ?? defaultValue;
}

function save(key, value) {
  localStorage.setItem(key, value);
}

function remove(key) {
    localStorage.removeItem(key);
}

// Specific data accessors
export const storage = {
  // Reaction counts
  loadOshiCounts: () => loadJSON('oshiCounts'),
  saveOshiCounts: (counts) => saveJSON('oshiCounts', counts),
  loadLikeCounts: () => loadJSON('likeCounts'),
  saveLikeCounts: (counts) => saveJSON('likeCounts', counts),
  loadFearCounts: () => loadJSON('fearCounts'),
  saveFearCounts: (counts) => saveJSON('fearCounts', counts),

  // Problem check state
  loadChecks: () => loadJSON('problemChecks'),
  saveChecks: (checks) => saveJSON('problemChecks', checks),

  // UI state
  loadSortOrder: (defaultValue) => load('currentSortOrder', defaultValue),
  saveSortOrder: (order) => save('currentSortOrder', order),
  loadShowUntouchedOnly: () => load('showUntouchedOnly') === 'true',
  saveShowUntouchedOnly: (value) => save('showUntouchedOnly', value),
  
  // Accordion state for major categories
  isMajorCatCollapsed: (largeCat) => load(`majorCatCollapsed-${largeCat}`) !== 'false',
  setMajorCatCollapsed: (largeCat, isCollapsed) => save(`majorCatCollapsed-${largeCat}`, isCollapsed),

  // Reset
  resetAll: () => {
    remove('problemChecks');
    remove('oshiCounts');
    remove('likeCounts');
    remove('fearCounts');
  }
};
