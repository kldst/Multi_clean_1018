# CLAUDE.md — Multi_SMPL_clean（準備開源的乾淨版）

從 `../Multi_SMPL_Temporal`（研究用、很雜）抽出來的開源候選 repo，2026-10-18 建立。對外文件用英文（README.md、
docs/），這份是內部開發筆記。已搬「推論 → 追蹤 + TTO → 評分」（逐位元驗證過）與 MAMMA + Harmony4D 訓練（未驗證，見「訓練」）；資料前處理、demo 還沒搬。

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

同樣輸入跑舊研究程式（`optimize/multi/` + 官方 evaluator）和本 repo，比對每個輸出陣列 / 數字：**22 組全部逐位元相同**。

| 部分 | 測試 | 結果 |
|---|---|---|
| 模型 | `load_model` strict load checkpoint_19 | 1463 個 tensor 全部一致 |
| ① 推論（每條 3 幀） | dance + 4 候選、singles `--select track`、CHI3D + 4 候選、Harmony4D（含 1 個評分幀的 GT） | 4/4 IDENTICAL |
| ② TTO（迭代 20/10/20） | dance joint / indep、singles 100 幀（跨分塊）、CHI3D、Harmony4D × {outlier+z, plain+body} | 6/6 IDENTICAL（只少 `b_matched_frac`） |
| ③ 評分 | dance / singles（+ 逐幀 npy）/ CHI3D（含 nogate）/ Harmony4D 的 infer 與 fit | 9/9 差 0 |
| markerless（3 人） | 推論 3 幀（`--people auto --candidates auto`）、candidates 追蹤 + TTO 80 幀 | 2/2 IDENTICAL |
| pytest | 27 個 | 全過（約 15–55 秒，CPU） |

踩到的坑：`smplx_body.py` 的 SMPL（非 SMPL-X）模型路徑原本用檔案位置推算，搬家後 Harmony4D GT decode 找不到檔 →
改成 `configs/paths.yaml` 的 `smpl_root` + `set_smpl_model_root()`。

## 和舊 repo 的對應

| 舊（`Multi_SMPL_Temporal/`） | 新 |
|---|---|
| `optimize/multi/infer_dance.py` / `optimize_clean/infer_mamma.py` | `python -m multismpl.inference.mamma`（singles 用 `--dataset singles`） |
| `run_markerless.sh <seq_dir> <people>`（`--meta-dir meta --video-dir videos_crf24 --candidates N+4 --cand-by-mask`） | `scripts/run_markerless.sh [序列名 ...]` = `--dataset markerless --people auto --candidates auto`；資料在 `paths.yaml` 的 `mamma_markerless`（`../data/mamma_markerless_multiple_people`，34 條、3–4 人、無 GT，3 GB，不進 repo） |
| `infer_chi3d.py`、`infer_h4d_dense.py` | `multismpl.inference.chi3d`、`multismpl.inference.harmony4d` |
| `optimize_multi.py --mode joint [--repair --tracker joint --tri-outlier --tri-up body]` | `multismpl.tto.run [--tracker candidates --triangulation outlier --up-axis body]` |
| `eval_dance_fit.py`、`singles_eval/tto/eval_singles_fit.py` | `multismpl.eval.mamma --dataset dance / singles` |
| `eval_chi3d_fit.py`、`eval_h4d_fit.py` | `multismpl.eval.chi3d`、`multismpl.eval.harmony4d` |
| hydra `instantiate(cfg.model)` | `multismpl.model.multismpl.load_model(configs/model_nano4.yaml, ckpt, device)` |

## 訓練（2026-10-21 加入，只有 MAMMA + Harmony4D）

使用者要求：只要 MAMMA + Harmony4D 的訓練；不要 3DHP、BEDLAM、跨腳實驗、overfit；**不做新舊比對驗證**。

- 入口：`python -m multismpl.train.launch --config train_mamma_harmony4d`（config 在 `configs/train/`，用
  `hydra.initialize_config_dir`）。
- config 來源：研究 repo git `3009567`（2026-08-19）的 `training/config/mamma_harmony4d_mask_dpt.yaml`，也就是最後一版只有
  MAMMA + Harmony4D 的設定（8 view、隨機取 view、aggregator 凍結）。改的地方：`_target_` 換成 `multismpl.*`、路徑換本機、
  model 區塊換成 `MultiSMPL` 參數、初始權重改成 `init_checkpoint`（預設 VGGT-1B `model.pt`；原本的
  `mamma_harmony4d_concat_mask_dpt_temporal_rel/ckpts/checkpoint_5.pt` 在這台找不到）。
  **注意**：這份 config 不一定就是訓練出 ckpt19 的那份（ckpt19 的資料夾名 `nano4_4v` 暗示 4 view）。
- 程式來源：研究 repo **目前**（10 月）的訓練程式，搬進 `multismpl/train/`（trainer、loss*、smpl_matching、temporal、
  utils/）和 `multismpl/data/`（dynamic_dataloader、composed_dataset、sys_smpl_multi、augmentation、worker_fn、
  track_util、assets/）。只改 import，另外拿掉：
  - `dynamic_dataloader.py` 的 `crossleg_balance` / `pose_family_dose` sampler（`crossleg_balance.py`、`pose_family_dose.py` 沒搬）
  - `composed_dataset.py` 的鏡像 clip hook
  - `loss_smpl.py` 的 cross-relation / cross-contact loss（及其 helper）
  - `trainer.py` 的 `mamma_dance_val`（依賴研究 repo 的 `eval/mamma_eval_dance`）
  以上在原程式裡都是預設關閉，拿掉不影響 MAMMA + Harmony4D 訓練。
- `MultiSMPL.forward` 補上訓練要的輸出：`pose_enc_list`（camera loss）、eval 模式回傳 `images`；`use_temporal_smpl_head = True`。
- `multismpl/data/assets/`（verts_512.pkl、smplx_512_body_parts.json）是 MAMMA 衍生資產，只有 landmark head 用（此 config 關閉），已 git-ignore。
- temporal head 的 mean-params（`tram/data/smpl/smpl_mean_params.npz`）新舊 repo 都不存在 → 兩邊都用 0 初始化，行為相同。

## 追蹤統一（2026-10-23）

使用者要求所有資料集統一成 **top-k → Joint Candidate Tracking (JCT，舊稱 v9) → TTO**（不用 presence 閾值）：
- `multismpl.tto.run` 預設改成 `--tracker candidates --triangulation outlier --up-axis body`；舊的 slot tracker 用 `--tracker slots`。
- 推論預設存 4 個候選（`mamma.py` / `chi3d.py` `--candidates 4`；harmony4d 本來就是 4）；singles 的 `--select track` 現在可以和候選一起用。
- `mamma.py` 找不到預設影片資料夾時改用 `videos/`（下載版的 dance / singles 是 `videos/`）。
- `scripts/run_benchmark.sh` 預設就是統一流程；`LEGACY=1` 重現舊數字（`UNIFIED=1` 已移除）。`run_markerless.sh` 改用預設（outlier + body）。
- Network 那列不受影響（仍是各 evaluator 的 top-k）；評分仍每幀用 GT 配對，追蹤不用 GT。
- MAMMA 上幾乎每幀的 TTO 起點被換成三角化位置（`triangulated_init` ≈ 全部）。原以為是 IOI_13 直式裁切讓 mask 中心偏掉，
  **用 GT 驗證後不是**：是網路骨盆水平偏 0.2–0.45 m，mask 三角化反而準（singles 4 條序列水平誤差 1–2 cm）。
- 新數字：`outputs/unified/{dance,singles,harmony4d}`（CHI3D 不在這台）。Harmony4D 只重跑 JCT + TTO（推論沿用
  `outputs/harmony4d_bothvis/infer`）。
- 2026-10-23 使用者要求：README / RESULTS 的 Harmony4D 改成 both-visible 視角的數字（`outputs/harmony4d_bothvis`，
  視角由 GT 挑「兩人都在畫面內」，`multismpl.inference.harmony4d_views`），文件中不說明挑法，只寫「每序列一組固定 4 視角、
  所有方法相同」。舊的官方隨機視角數字：network 66.35 / 44.79，TTO 59.56 / 39.75（H36M-14）。

## JCT 定案（2026-10-24）

`multismpl.tto.run` 預設 = 使用者採用的版本（README / RESULTS / 會議網頁 tto_explained_noE.html 都已換成這組數字）：
- `--up-shift 0`（原 0.15 m 往下移；singles 上 GT 檢查 mask 中心本來就在骨盆高度附近，改 0 分數幾乎不變）。
- `--prefer-consistent`（預設開）：去重（< 25 cm）時優先保留「網路骨盆與 mask 三角化 < 1 m」的候選。修 singles
  Stepper_50049_1（混雜候選：mask 在主角、身體在背景人，225 幀選錯 → 0）。
- `--keep-topk-pairs`（預設開）：網路 top-k 的 slot 彼此不去重。修 CHI3D 擁抱（兩人 mask 中心三角化 < 25 cm，
  Hug 1 有 42/95 幀被當同一人刪掉）。
- `--topk-bonus` / `--w-pose-link` 試過沒效果，預設 0。舊行為：`LEGACY=1 scripts/run_benchmark.sh`。
- 結果（outputs/unified/<ds>/eval_fit_C*.json；singles 是 eval_fit_B.json，單人時 C 與 B 相同）：
  dance 42.08（舊 41.72）、CHI3D 有門檻 H36M-14 44.44（舊 44.35，recall 100%）、singles 48.45（舊 49.44）、
  Harmony4D SMPL-22 45.27（舊 45.77）。network 列改用本機重跑（dance 54.84、singles 61.13、CHI3D 無門檻 58.13）。
- 資料：CHI3D train/s04 解壓到 `/mnt/train-data-2-hdd/yian/data/CHI3D`（paths.yaml 已設）；singles 22 條齊全
  （SubjectCalib_50050_1 的 IOI_05/13 重新下載過）。

## 多人規則（2026-10-24，預設開）

markerless 多人場景（telephone_chain 4–6 人）看到：軌跡被沒信心的候選騙走、追丟後要等 60 幀才找回、一人缺席時全員
平滑被關掉（全員抖動）。新增並設為預設：
- `--min-presence-boost 0.1`：presence < 0.1 的候選不再用 mask 比例把信心拉到 max(presence, support)。
- `--reacquire`：上一幀缺席的軌跡可接「清楚（support ≥ 0.75、presence ≥ 0.3）」的候選，不受距離限制，移動成本上限 4。
- `--per-person-smooth`：平滑 / beta 用每人自己的 ambiguous（缺席或 < 15 cm），不再整幀全關。
- `--dup-mask-iou`（mask 重疊判同一人）試過：排成一列的人在部分視角互相遮擋，會誤刪真人 → 預設 0（關）。
- 結果（fit_D，README / RESULTS / 網頁已更新）：dance 42.08、CHI3D 44.44 都不變；singles 48.45 → 48.49；Harmony4D SMPL-22 45.27 → 44.28（缺人 175 → 37）；
  telephone_chain 4 人缺人 264 → 94、5 人 300 → 39。舊行為：`--min-presence-boost 0 --no-reacquire --no-per-person-smooth`。
- 影片：`outputs/dance_video/render_mesh_video.py --hide-dropped` 不畫追蹤找不到的人（只影響影片，不影響評分）。

## 還沒搬（使用者決定先不做）

1. 資料前處理：`compose_script/`、`smpl_transfer/` 中真正用到的轉換（產生 compose 格式的 MAMMA / Harmony4D 訓練資料）。
2. demo（取代 186 KB 的 `demo_gradio_smpl_multi.py`）。
3. E 組跨腳微調、3DHP、BEDLAM。
4. LICENSE（未決定）；發布前跑 secret scan；checkpoint 上 Hugging Face（原始 VGGT-1B 權重非商用）。
