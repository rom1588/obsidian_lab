# BioVid Heat Pain Database オンライン前処理＆HDF5抽出パイプラインの構築

BioVid Heat Pain Database（148GBのZipファイル）を完全に解凍することなく、Pythonの `zipfile` ライブラリを使用して親Zip内の子Zipから生理信号データをストリーミング抽出し、前処理（EDAのSCL/SCR分離、ECGのR波ピーク検出、HRV特徴量抽出）を行った上で、軽量なHDF5ファイル（Gzip圧縮適用）に一括保存するパイプラインを構築します。

## User Review Required

IMPORTANT

**5.5秒窓におけるHRV周波数領域特徴量（LF, HF）の扱いについて** 1試行あたりのデータ長が2817サンプル（約5.5秒）と非常に短いため、通常のFFTやWelch法を用いた周波数領域の解析（特に下限が0.04HzのLF成分）は統計的・信号処理的な限界があります。本パイプラインでは補間とゼロパディングによる周波数特徴量算出を実装しますが、極端に短い時系列であるため、LF/HF値の信頼性には工学的な限界があることをご留意ください（機械学習モデルの特徴量の一つとしては動作します）。 ※BioVid Part Bの公式特徴量セット（`Table_Step2_159Features-85Subs-5Levels-z.xlsx` など）も同梱されているため、必要に応じてそちらから特徴量を直接取り出すことも可能です。今回は自前パイプラインの構築を進めます。

## Open Questions

IMPORTANT

**Q1. データのソースとして `biosignals_raw.zip` と `biosignals_filtered.zip` のどちらをメインに使用しますか？** `biosignals_filtered.zip` はすでにベースラインノイズや高周波ノイズが除去された信号が格納されていますが、自前でバタワースフィルタ等の設計を検証・カスタマイズしたい場合は `biosignals_raw.zip` を使用します。基本的には、ノイズ除去がすでに施されている `biosignals_filtered.zip` を用いた方がR波検出などの精度が安定するため、こちらをデフォルトとして実装することを提案します。

## Proposed Changes

### データ前処理・特徴量抽出コンポーネント

#### [NEW] feature_extraction.py

- **EDA前処理**:
    - 3次バタワースローパスフィルタ（カットオフ: 1Hz）による平滑化。
    - SCR（皮膚電気反応）とSCL（皮膚電気水準）への分離（ローパスフィルタによる基底トレンド抽出および差分）。
    - 特徴量: SCR最大振幅（SCR Amplitude）、立ち上がり時間（Rise time）。
- **ECG前処理**:
    - バンドパスフィルタ（5 - 15 Hz）および `scipy.signal.find_peaks` を用いたR波ピーク検出。
    - R-R間隔（RRI）時系列の算出。
    - 時間領域特徴量: Mean RRI, SDNN, RMSSD。
    - 周波数領域特徴量: RRIの3次スプライン補間（4Hzリサンプリング）、Welch法によるパワースペクトル密度（PSD）推定、LF（0.04 - 0.15 Hz）パワー、HF（0.15 - 0.40 Hz）パワー、LF/HF比の算出。

#### [NEW] preprocess_pipeline.py

- **Zipストリーミング機構**:
    - 親Zip `BioVid.zip` をオープン。
    - `BioVid/PartB/starting_point.zip` 内の `samples.csv` をタブ区切りで読み込み、被験者・試行のリストを作成。
    - `BioVid/PartB/biosignals_filtered.zip` （または `raw`）をメモリ上で `zipfile.ZipFile` として開き、各CSVファイルを順次ストリーム読み込み。
- **HDF5保存機構**:
    - `h5py` ライブラリを使用して `f:/研究用（大学）/My_Research_Scratch(AI)/data/biovid_preprocessed.h5` を作成。
    - メタデータ（`subject_ids`, `labels`, `temperatures`, `self_reports`）および抽出したEDA/ECG特徴量（`eda_features`, `ecg_features`）をフラットな2D Datasetとして格納。
    - 時系列データ（前処理済みEDA/ECG生信号）は被験者・試行ごとの階層構造（`/raw_signals/subject_XX/trial_YY/...`）に格納。
    - データセット全体に対して Gzip 圧縮（`compression="gzip", compression_opts=4`）を適用。
- **メモリ・処理効率最適化**:
    - 被験者ごとのバッチ処理と明示的なガベージコレクション（`gc.collect()`）。
    - 進行状況をプログレスバー（`tqdm`）で可視化。

## Verification Plan

### 自動テスト / 動作確認

- `uv run --with pandas --with scipy --with h5py --with tqdm --with openpyxl` 環境で実行し、少数のサンプル（例: 最初の1名の被験者のデータのみ）に対してテスト実行を行います。
- テスト書き込みされた HDF5 ファイルを読み込み、以下の項目を確認します：
    1. HDF5の構造が設計通りか（`h5py.File.keys()` 等で検証）。
    2. 抽出された特徴量に `NaN` や無限大が含まれていないか。
    3. R波検出の可視化ログまたは検出数を確認し、正しくピークが捉えられているか。

### 手動確認

- テスト実行完了後、処理速度（1被験者あたりの処理時間）を測定し、全87名の処理にかかる想定時間を算出します。