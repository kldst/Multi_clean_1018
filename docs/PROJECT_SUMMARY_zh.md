# MultiSMPL 專案摘要

多視角、多人的 SMPL-X 人體重建：前饋網路一次預測每個人的 3D 身體和剪影，再用剪影做 test-time optimisation（TTO）把身體修準。
本 repo（`Multi_SMPL_clean`）是從研究用的 `Multi_SMPL_Temporal` 整理出來的開源候選版。

更新：2026-10-22

---

## 1. 方法

```
多台校正相機的同步影片
  │
  ▼
① 網路（每幀一次）            VGGT aggregator + 時序 SMPL head + DPT 人物 mask head
  │   每幀 20 個 slot：存在機率、SMPL-X 姿勢 / 體型、骨盆位置（以第一台相機為座標），以及每人每台相機的剪影
  ▼
② 追蹤                        決定每一幀哪個 slot 是哪個人
  │   slots：已選好的 slot 之間做 keep/swap Viterbi（dance、CHI3D、singles）
  │   candidates（v9）：mask 候選 + 剪影重心三角化 + 每幀所有人一起做 Hungarian（Harmony4D、markerless）
  ▼
③ TTO                         整段影片一起優化，不用 GT 姿勢、不用 2D 關鍵點偵測器、不用可微分渲染器
  │   loss：inside（身體點要在剪影內）+ cover（剪影要被身體蓋滿），Geman-McClure；
  │         姿勢 / 朝向往網路答案拉、體型整段一組、關節加速度平滑
  │   三階段：只動骨盆 → + 朝向、體型 → + 21 關節姿勢；多人時被別人擋住的點不罰
  ▼
每個人的 SMPL-X（世界座標）
```

- **模型**：ckpt19（`nano4`），12 億參數；只在 MAMMA 合成資料 + Harmony4D 上訓練（4 view、T=3），沒看過 CHI3D。
- **相機**：網路本身不需要相機；TTO 需要校正好的相機（其他比較方法也都用 GT 相機）。另有實驗顯示用預訓練 VGGT 估相機時，姿勢幾乎不受影響，只差整體尺度約 6%（不在本 repo）。

## 2. 程式架構

| 位置 | 內容 | 規模 |
|---|---|---|
| `multismpl/model/` | 網路：VGGT aggregator + 3 個用到的 head（+ camera head），可嚴格載入 ckpt19 | 26 檔、4,057 行（多為 VGGT 原碼） |
| `multismpl/body/` | SMPL-X / SMPL 解碼、座標轉換 | 4 檔、780 行 |
| `multismpl/data/` | 影像前處理、Harmony4D loader、訓練用 dataset / dataloader | 11 檔、5,754 行 |
| `multismpl/inference/` | ① 推論：`mamma`（dance / singles / markerless）、`chi3d`、`harmony4d` | 6 檔、625 行 |
| `multismpl/tto/` | ②③ 追蹤、三角化、剪影 loss、TTO、CLI（`run.py`） | 6 檔、703 行 |
| `multismpl/eval/` | 各 benchmark 評分；`protocols/` 是從官方評分程式原樣抽出的協定 | 10 檔、710 行 |
| `multismpl/train/` | MAMMA + Harmony4D 訓練（trainer、loss、optimizer…） | 20 檔、5,352 行 |
| `configs/` | 模型架構、路徑（`paths.yaml`，不進 git）、訓練 config | |
| `scripts/` | `run_benchmark.sh`（一鍵跑一個 benchmark）、`run_markerless.sh` | |
| `tests/` | 27 個 CPU 測試，用合成資料（含端到端 TTO 回歸測試） | 8 檔 |

所有路徑都從 `configs/paths.yaml` 讀，程式中沒有寫死的伺服器路徑。

## 3. 使用方式

```bash
# 一鍵：推論 → 網路分數 → 追蹤 + TTO → TTO 分數
OUT=outputs/dance scripts/run_benchmark.sh dance            # dance | singles | chi3d | harmony4d

# 分步
python -m multismpl.inference.mamma --out out/infer                          # ①
python -m multismpl.tto.run --infer out/infer --out out/fit                  # ②③
python -m multismpl.eval.mamma --dataset dance --kind fit --pred-dir out/fit --out eval.json

# markerless 多人（無 GT，人數從序列名讀）
OUT=outputs/markerless scripts/run_markerless.sh [序列名 ...]

# 訓練（MAMMA + Harmony4D）
python -m multismpl.train.launch --config train_mamma_harmony4d
```

## 4. 結果（MPJPE / PA-MPJPE，mm，越低越好）

| Benchmark | 指標 | 網路 | + TTO |
|---|---|---|---|
| MAMMA Dance（18 條，2 人，4 view） | SMPL-22 | 54.94 / 39.28 | **41.72 / 30.60** |
| CHI3D s04（MV4DHMR `test.py` 協定，500 mm 門檻） | H36M-14 | 57.82 / 34.13 | **44.37 / 29.31** |
| MAMMA singles（22 條，單人，top-1 選人） | SMPL-22 | 64.98 / 44.67 | **55.28 / 39.50** |
| Harmony4D test（近距離互動，無門檻） | H36M-14 | 66.35 / 44.79 | **59.56 / 39.75** |

TTO 在四個資料集都有效：Dance、CHI3D 降約 13 mm，singles 約 10 mm，Harmony4D 約 7 mm；CHI3D 的平移誤差從 234 降到 28 mm。

**和其他方法比（同一評分程式、同一批幀與相機）**

| | 我們 + TTO | MAMMA 官方 | HeatFormer | MV4DHMR |
|---|---|---|---|---|
| Dance SMPL-22 | 41.72 | **37.99**（同 4 view，慢約 7 倍） | 67.96（用 GT 框） | 75.00 |
| CHI3D H36M-14 | 44.37 | **32.97**（用全部 50 fps 幀） | 54.16（GT 框，同協定）* | 62.0（論文數字） |
| singles SMPL-22 | 55.28 | – | **53.78**（用 GT 框） | 122.34 |

\* HeatFormer 輸出 SMPL：依官方 `test.py` 以各自模型的 joint 0 置中（54.16）；若兩邊都以 H36M 骨盆置中則為 51.79。詳見第 7 節。

完整表格與每個數字的計算方式：`docs/RESULTS.md`；評分協定的設計理由見第 5 節。

## 5. 評分協定：為什麼用「匈牙利骨盆配對、無門檻」

MAMMA Dance、MAMMA singles 的主要數字，以及我們、HeatFormer、MV4DHMR、MAMMA 官方在這兩個資料集上的比較，都用同一套協定：

1. 預測與 GT 的 mesh 都先轉成同一個 6890 頂點的 SMPL 拓撲（SMPL-X 用 `smplx2smpl`，HeatFormer 本來就是 SMPL）。
2. 每一幀用**匈牙利演算法**把預測配給 GT，依據是兩者 **H36M regressor 算出的骨盆**在世界座標的距離，**不設距離門檻**。
3. 配到的每一對：SMPL-22 以 SMPL regressor 的 joint 0 為根、H36M-14 以 H36M 骨盆為根，兩邊用同一個 regressor、各自減掉自己的根。

**為什麼這樣設計**

| 選擇 | 原因 |
|---|---|
| 用 H36M 骨盆，不用各模型自己的骨盆關節 | 各方法輸出 SMPL 或 SMPL-X，兩者的骨盆定義在同一個人身上就差約 18.5 mm（CHI3D GT 上量得）。H36M 骨盆是從同一份 SMPL 拓撲、用同一個 regressor 算的，對每個方法都是同一個定義；它也正是 H36M-14 的根。 |
| 用骨盆位置配對，不用 PVE | 配對只該回答「誰是誰」。PVE 包含姿勢誤差，用要評的量來配對，預測比 GT 多時會偏向挑出姿勢最好的那個。 |
| 匈牙利，不用貪婪 | 保證一對一並讓總距離最小；兩人貼近時（雙人舞、擁抱）貪婪先鎖定最近的一對，容易把身分配錯。 |
| 不設門檻 | 每個 GT 人次都計分。設門檻（如 500 mm）會把失敗的幀排除在 MPJPE 之外，分數只代表「成功的那部分」，不同 recall 的方法無法比較。 |

**用 MV4DHMR 自己的預測驗證**（同一批預測，只換配對方式，SMPL-22 / recall）：

| 配對方式 | Dance 4 view | singles |
|---|---|---|
| **匈牙利骨盆，無門檻（本協定）** | **75.00 / 100%** | **122.34 / 100%** |
| 貪婪 PVE，無門檻 | 80.89 / 100% | 122.69 / 100% |
| 貪婪 PVE + 500 mm（`test.py` 的機制，修正 bug 後） | 82.49 / 14.0% | 75.51 / 5.3% |
| MV4DHMR 官方 `match_greedy` 原樣 + 500 mm | 90.91 / 15.1% | 75.64 / 5.3% |

- 每個 GT 都計分時，本協定不比貪婪 PVE 不利，Dance 上還低 5.9 mm。也就是說，用它評 MV4DHMR 並沒有低估對方。
- 加 500 mm 門檻後 MV4DHMR 只剩 5–15% 的人被計分：它在 MAMMA 4 view 的骨盆平均偏 677 mm（8 view 為 110 mm），大多超過門檻。門檻版的 MPJPE 看起來較低，只是因為大部分人被排除了。
- MV4DHMR 官方的 `utils/training.py` `match_greedy` 用在 PVE 時有兩個 bug：判斷式的運算子優先順序讓它不檢查預測或 GT 是否已配過（`training.py:89-90`）；以及把被拒絕的「配對」當成 false positive 計數，導致迴圈提早結束（`training.py:62-63, 103`）。每幀 2 預測對 2 GT（CHI3D）時影響小；在 Dance 上造成重複配對，MPJPE 多出約 8 mm。
- 重算程式與結果：`analysis_outputs/crossed_legs_15s/e2e3/mv4dhmr_testpy_match/`；協定原文：研究 repo `eval/mamma_eval_dance/AGENTS.md`。

**例外：**
- **CHI3D**：為了能和論文數字（MV4DHMR、Multi-HMR 等）直接比較，沿用 MV4DHMR `test.py` 協定：貪婪 PVE 配對、500 mm 門檻、以各模型自己的 joint 0 為根。同時另外報告不設門檻的版本。
- **Harmony4D**：沿用官方 `eval_h4d_nano.py` 指標：以絕對 PVE 做匈牙利配對、以 H36M 骨盆為根；主要數字不設門檻，另附 500 mm 門檻版。這個資料集上只有我們自己的方法。

## 6. 驗證

| 項目 | 方法 | 結果 |
|---|---|---|
| 推論 / TTO / 評分 | 和研究版程式用同樣輸入跑，逐一比對每個輸出陣列與數字 | **22 組全部逐位元相同**（4 個資料集 + markerless 3 人） |
| 模型 | ckpt19 strict load | 1,463 個 tensor 全部一致 |
| 單元 / 回歸測試 | `pytest`（CPU、合成資料） | 27 個全過；合成場景中 TTO 把偏 33.5 cm 的骨盆拉回 0.5–1.1 cm |
| 訓練 | 啟動檢查（建模型 / loss / optimizer、載入權重、讀一個 batch） | 可啟動；**數值未驗證**（依使用者決定不做新舊比對） |

## 7. 公平性與已知限制（2026-10-22 與 Codex 共同審核）

評分程式本身一致（同一份 GT、同樣關節定義、同樣配對、涵蓋率完整），但比較表要分組呈現：

1. **singles 的 track 選人參數是在測試集上調的** → 主要數字改用未調參的 top-1（55.28），track（49.00）只放消融。
2. **HeatFormer 用了 GT 框、GT 人數、GT 跨 view 對應與 GT 關節可見度**（CHI3D 的 YOLO 版本也用 GT 身分對應並退回 GT 框）→ 標為 oracle，不和端到端方法排名。
3. **CHI3D 根關節**：官方 `test.py` 以「各自模型的 joint 0」置中；SMPL 輸出的方法因此是 SMPL 骨盆對 SMPL-X 骨盆（同一人身上相差約 18.5 mm），對 HeatFormer 這類方法有約 2 mm 的系統性不利。
4. **MV4DHMR 會輸出多於 GT 人數的預測**（dance 多 824 個、singles precision 87.7%），評分時等於由 GT 挑出最接近者 → 需和 recall / precision 一起報告。
5. **線上與離線方法混在同一表**：我們的網路是因果 T=3；TTO、MAMMA、MV4DHMR 都使用整段影片 → 建議分組。
6. CHI3D 上 MAMMA 官方用全部 50 fps 影格擬合，我們的 TTO 只用評分的幀（每 3 幀取 1 幀）。

## 8. 尚未完成

- LICENSE 未決定；模型由原始 VGGT-1B 權重初始化（該權重為非商用授權）；SMPL / SMPL-X 與各資料集不可隨 repo 散布。
- 資料前處理（產生訓練用 compose 格式）、demo、E 組跨腳微調、3DHP / BEDLAM 訓練：依決定暫不納入。
- 統一用 v9 追蹤跑全部 benchmark：程式已備（`UNIFIED=1`），尚未跑；singles 用 v9 可解決在測試集調參的問題。
- singles 改用「presence 門檻 + 匈牙利 PVE 配對」的偵測式評分（和 MAMMA 上評 MV4DHMR 的方式相同）：已重跑推論並存下所有 slot
  （`--save-all-slots`），可離線重選。全部 22 條、只有網路（無 TTO）的結果：

  | 選人方式 | SMPL-22 | PA | H36M-14 | PA | 每幀預測數 | recall | precision |
  |---|---|---|---|---|---|---|---|
  | top-1（未調參） | 64.98 | 44.67 | 71.71 | 46.13 | 1.00 | 100% | 100% |
  | track（在測試集調參） | 60.91 | 42.80 | 66.51 | 44.03 | 1.00 | 100% | 100% |
  | presence ≥ 0.35 + 匈牙利 PVE | 61.12 | 43.04 | 66.90 | 44.38 | 2.40 | 99.9% | 41.7% |
  | presence ≥ 0.5 + 匈牙利 PVE | 62.45 | 43.46 | 68.43 | 44.80 | 1.51 | 95.1% | 63.1% |

  門檻 0.35 讓 GT 從每幀約 2.4 個預測中挑人，結果和 track 幾乎相同；precision 低主要是因為 singles 只標註受試者，
  路過的工作人員被偵測到也算 false positive。門檻版沒有對應的 TTO（每幀人數不固定）。
