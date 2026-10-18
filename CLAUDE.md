# CLAUDE.md — Multi_SMPL_clean（準備開源的乾淨版）

從 `../Multi_SMPL_Temporal`（研究用、很雜）抽出來的開源候選 repo，2026-10-18 建立。對外文件用英文（README.md、
docs/），這份是內部開發筆記。**目前只搬了「推論 → 追蹤 + TTO → 評分」這條路徑**；訓練、資料前處理、demo 還沒搬。

## 規則

- **路徑一律走 `configs/paths.yaml`**（`multismpl/paths.py` 讀，`PATHS.<key>`）。程式裡不可出現 `/mnt/...`。
  `configs/paths.yaml` 是本機檔（git-ignored），範本是 `configs/paths.example.yaml`。
- 執行方式：在 repo 根目錄 `python -m multismpl.<模組>`（或 `pip install -e .`）。
- **官方評分協定不改**：`multismpl/eval/protocols/` 的函式本體是從各 evaluator 原樣搬來的（見各檔 docstring），
  改了就不能和論文 / 其他方法比。
- **模型屬性名稱不改**：`MultiSMPL` 的 `aggregator`、`camera_head`、`smpl_multi_query_trans_rot_head`、
  `person_mask_head` 對應 checkpoint 的 key，改名 strict load 會失敗。
- `multismpl/model/` 是 VGGT 衍生程式（保留 Meta 版權標頭，`LICENSE_VGGT.txt`）；`multismpl/data/`、
  `body/smplx_body.py` 是訓練程式原樣搬來（只改 import），之後搬訓練時會再整理。
- GPU：依使用者當下指示（2026-10-18 起不准用任何 GPU）；CPU 最多 2 個程序 × 10 執行緒、加 nice。

## 結構

| 位置 | 內容 | 來源 |
|---|---|---|
| `multismpl/model/multismpl.py` | `MultiSMPL`：只建 checkpoint 裡有的 4 個部分；`load_model()` strict load | 新寫（取代 `vggt/models/vggt.py` 的 16 個 head 開關） |
| `multismpl/model/{aggregator,layers,heads,utils}` | 用到的 VGGT 模組（6 個 head 檔、geometry / pose_enc / rotation / distortion） | `vggt/` 原樣 |
| `multismpl/body/` | `smplx_body.py`（SMPL-X decode）、`numerics.py`、`body_model.py`（decode_anchor、gauge→world、body_up） | `training/smpl_body.py`、`optimize_clean/body_model.py` |
| `multismpl/data/` | `base_dataset.py`（process_one_image）、`dataset_util.py`、`harmony4d.py`、`landmark_mask_gt.py` | `training/data/` 原樣 |
| `multismpl/inference/` | `network.py`、`video.py`、`mamma.py`、`chi3d.py`、`harmony4d.py` | `optimize_clean/infer_*.py` |
| `multismpl/tto/` | `silhouette.py`、`triangulation.py`、`tracking.py`、`fit.py`、`run.py` | `optimize_clean/` |
| `multismpl/eval/` | `mamma.py`（dance / singles）、`chi3d.py`、`harmony4d.py`、`common.py` | `optimize_clean/evaluate_*.py` |
| `multismpl/eval/protocols/` | `common.py`（top2_nms3d、Umeyama、greedy PVE）、`mamma.py`、`chi3d.py`、`harmony4d.py` | 官方 evaluator 抽出（eval_mamma_dance_comparison / eval_mamma_singles / eval_chi3d / eval_rich_nano） |
| `configs/model_nano4.yaml` | 釋出模型的架構參數 | `training/config/mamma_harmony4d_mask_dpt.yaml` 的 model 區塊 |
| `tests/` | CPU、合成資料（見下） | 2 個由舊 tests 改寫，其餘新寫 |

## 測試（`pytest`，mamma env 已裝 pytest 9.1.1）

| 檔案 | 測什麼 |
|---|---|
| `test_model_heads.py` | temporal SMPL head 形狀 / 梯度、DPT mask head 分塊一致（舊 tests 搬來） |
| `test_geometry.py` | 投影、Geman-McClure 飽和、畫面外距離、深度圖取最近 |
| `test_triangulation.py` | DLT、200 px 離群 view 會被剔除、30 px 正常雜訊不剔除、mask 圓盤三角化 |
| `test_tracking.py` | slot 對調被修回、單人 identity、假候選（無 mask）不會被選 |
| `test_metrics.py` | Umeyama、相同 mesh = 0、平移 / 旋轉不變性、top2_nms3d、500 mm 門檻、recall |
| `test_tto_regression.py` | 合成 SMPL-X 場景 4 相機、起點偏 34 cm → TTO 拉回 < 5 cm；與 `tests/fixtures/tto_synthetic_expected.npz` 比對（需要 SMPL-X，沒有就 skip） |

## 驗證（2026-10-18，CPU）

同樣輸入跑舊研究程式（`optimize/multi/` + 官方 evaluator）和本 repo，比對每個輸出陣列 / 數字：**20 組全部逐位元相同**。

| 部分 | 測試 | 結果 |
|---|---|---|
| 模型 | `load_model` strict load checkpoint_19 | 1463 個 tensor 全部一致 |
| ① 推論（每條 3 幀） | dance + 4 候選、singles `--select track`、CHI3D + 4 候選、Harmony4D（含 1 個評分幀的 GT） | 4/4 IDENTICAL |
| ② TTO（迭代 20/10/20） | dance joint / indep、singles 100 幀（跨分塊）、CHI3D、Harmony4D × {outlier+z, plain+body} | 6/6 IDENTICAL（只少 `b_matched_frac`） |
| ③ 評分 | dance / singles（+ 逐幀 npy）/ CHI3D（含 nogate）/ Harmony4D 的 infer 與 fit | 9/9 差 0 |
| pytest | 25 個 | 全過（約 15–55 秒，CPU） |

踩到的坑：`smplx_body.py` 的 SMPL（非 SMPL-X）模型路徑原本用檔案位置推算，搬家後 Harmony4D GT decode 找不到檔 →
改成 `configs/paths.yaml` 的 `smpl_root` + `set_smpl_model_root()`。

## 和舊 repo 的對應

| 舊（`Multi_SMPL_Temporal/`） | 新 |
|---|---|
| `optimize/multi/infer_dance.py` / `optimize_clean/infer_mamma.py` | `python -m multismpl.inference.mamma`（singles 用 `--dataset singles`） |
| `infer_chi3d.py`、`infer_h4d_dense.py` | `multismpl.inference.chi3d`、`multismpl.inference.harmony4d` |
| `optimize_multi.py --mode joint [--repair --tracker joint --tri-outlier --tri-up body]` | `multismpl.tto.run [--tracker candidates --triangulation outlier --up-axis body]` |
| `eval_dance_fit.py`、`singles_eval/tto/eval_singles_fit.py` | `multismpl.eval.mamma --dataset dance / singles` |
| `eval_chi3d_fit.py`、`eval_h4d_fit.py` | `multismpl.eval.chi3d`、`multismpl.eval.harmony4d` |
| hydra `instantiate(cfg.model)` | `multismpl.model.multismpl.load_model(configs/model_nano4.yaml, ckpt, device)` |

## 還沒搬（2026-10-18 使用者決定：先不做，範圍維持「推論 + TTO + 評分」）

1. 訓練：trainer、loss（`training/loss*.py`、`smpl_matching.py`）、資料集（`sys_smpl_multi.py`、composed、dataloader）、
   最終訓練 config。驗證方式：固定 seed 跑數十步，loss 逐步和舊版一致。
2. 資料前處理：`compose_script/`、`smpl_transfer/` 中真正用到的轉換。
3. demo（取代 186 KB 的 `demo_gradio_smpl_multi.py`）。
4. E 組跨腳微調（平衡取樣、鏡像、contact loss）。
5. LICENSE（未決定）；發布前跑 secret scan；checkpoint 上 Hugging Face（原始 VGGT-1B 權重非商用）。
