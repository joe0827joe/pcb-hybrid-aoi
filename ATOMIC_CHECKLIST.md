# 🛠️ Atomic Commit Checklist (門禁標準)

## Phase 1: 代碼孤立性 (Isolation)
- [ ] **單一職責**：本次變更是否僅涉及單一功能/Bug/優化？ (如：僅修改了 ROI 邏輯)
- [ ] **無污染**：是否已移除所有 `print()`, `pdb.set_trace()` 或臨時測試路徑？

## Phase 2: 效能與規格 (Performance)
- [ ] **Latency 達標**：`benchmark.py` 顯示 Latency 是否穩定在 130ms 以內？
- [ ] **Schema 對齊**：產出的 JSON/Dict 格式是否符合 `detection_schema.md` 定義？

## Phase 3: 工程品質 (Quality)
- [ ] **測試覆蓋**：`test_cv_prototype.py` 是否包含 Normal, Edge, Error 三種 Case？
- [ ] **環境同步**：若有新增 library，是否已更新 `requirements.txt` 或 `Conda` 配置？

## Phase 4: Git 規範 (DevOps)
- [ ] **格式合規**：Commit Message 是否符合 `type(scope): subject` 格式？
- [ ] **文件更新**：`README.md` 的專案樹狀圖是否需要隨此變更更新？
