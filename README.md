# whisper-tpu-colab-longform

> **A TPU-accelerated pipeline to transcribe long audio (up to 1h)**  
> Using OpenAI Whisper (PyTorch/XLA) on Google Colab, splitting into overlapping 30-second chunks, inferring on TPU, and stitching Chinese transcripts.

> 可將長音檔（最長 1 小時）切片成 30 秒段，並透過重疊（overlap）策略推論，最終合併成一份中文逐字稿。

## 1. Background

OpenAI’s Whisper model is trained to process **up to 30s** of audio at a time (`max_source_positions=1500`).  
- Whenever you input longer audio, the feature extractor will automatically truncate or pad to 30s.  
- To fully transcribe a longer file (e.g., 1h), you must split it into consecutive 30s segments（或使用 longform transcription API），然後再將各段結果拼接。  

本專案展示如何在 **Google Colab 的 TPU v2/v3** 上，利用 **PyTorch/XLA**：
1. 將 **最長 1 小時** 的音檔切成 30 秒 chunk（左右重疊 5 秒）。  
2. 一段一段地把每 30 秒送到 TPU 做推論 (Whisper-small)。  
3. 最後把所有 chunk 的中文文字串接成完整逐字稿。  

> 不廢話 [點我前往 Colab Notebook](https://drive.google.com/file/d/1whpeuVN_HTxuwa11LV3cNhanz4wbSs6W/view?usp=sharing)


## 2. Features

- **TPU 加速**：利用免費 Colab TPU，每個 30s chunk 只需約 2–5 秒推論。  
- **流式讀取**：不需一次把整個音檔載入記憶體，使用 `SoundFile` 逐 chunk 讀取。  
- **自動 Overlap 切片**：左右各延伸 5 秒，通常能減少在切點處「句子斷裂」的問題。  
- **XLA Cached Compile**：只在第一個 Dummy 推論時做完整編譯，後續 chunk 都走快取，進而大幅加快推論速度。  
- **可處理最長 1 小時音檔**：適合長時間會議、講座、Podcast 等錄音檔批次轉錄。  
- **可輕易轉為 任何長度**：只要 `n_chunks = ceil(total_samples/480000)`，就能擴充支援更長音訊。  


## 3. Repo Structure

```
whisper-tpu-stream-transcription/
├── README.md
├── LICENSE
├── requirements.txt
└── whisper-tpu-colab-longform.ipynb
```

- **`README.md`**：本檔案，提供安裝、使用、技術說明。  
- **`LICENSE`**：建議採用 MIT 或 Apache 2.0。  
- **`requirements.txt`**：列出必要套件。  
- **`colab_notebooks/`**：提供三個 Colab Notebook 範例，供直接在 Colab 執行、Demo。    


## 4. Quick Start

### 4.1 Clone & Open Colab

```bash
!git clone https://github.com/你的帳號/whisper-tpu-colab-longform.git
%cd whisper-tpu-colab-longform/colab_notebooks
````

然後在 Colab 直接開啟 `01_install_and_dummy.ipynb`。

### 4.2 Run Step 1: Install & Dummy TPU Test

1. 開啟 `01_install_and_dummy.ipynb`。
2. 執行第一格：

   ```python
   import os
   os.environ["PT_XLA_DEBUG"] = "1"
   # 安裝 torch, torch_xla, transformers, ffmpeg…
   !pip uninstall -y torch torch_xla...
   !pip install torch==2.6.0+cpu.cxx11.abi ...
   !pip install "transformers>=4.39.0,<4.40.0" sentencepiece librosa soundfile
   !apt update && apt install -y ffmpeg
   # 匯入 torch / torch_xla & 檢查 TPU
   import torch
   import torch_xla.core.xla_model as xm
   print("XLA devices:", xm.get_xla_supported_devices())
   ```
3. 執行第二格：

   ```python
   # Dummy Generate 測試
   import numpy as np, time
   silence = np.zeros(16000*30, dtype=np.float32)
   feats = processor.feature_extractor(
       silence, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
   )
   input_feats = feats.input_features.to(device)
   attn_mask   = feats.attention_mask.to(device)
   decoder_prompt = torch.tensor([processor.get_decoder_prompt_ids("chinese","transcribe")[0]], device=device)
   t0 = time.time()
   _ = model.generate(input_feats, attention_mask=attn_mask, decoder_input_ids=decoder_prompt, max_length=model.config.max_target_positions)
   print(f"Dummy 推理耗時: {time.time()-t0:.2f} 秒")
   ```
4. **檢查 XLA Metrics**（可選）：

   * 在 Dummy 推理結束後呼叫：

     ```python
     import torch_xla.debug.metrics as met
     import torch_xla.core.xla_model as xm
     met.clear_all()
     xm.mark_step()
     _ = model.generate(input_feats, attention_mask=attn_mask, decoder_input_ids=decoder_prompt, max_length=model.config.max_target_positions)
     xm.mark_step()
     print(met.short_metrics_report())
     ```
   * 確保 `ExecuteTime>0`、`UncachedCompile=1`、`CachedCompile=1`，表示該 graph 已編譯成功、實際運算落在 TPU。

> **注意**：若上述 `get_xla_supported_devices()` 回傳 `[]`，就表示 Runtime 沒有分配 TPU，需要在 Colab menu (`Runtime` → `Change runtime type`) 重新選擇「TPU」。

### 4.3 Run Step 2: 1h Streaming Transcription

1. 開啟 `02_streaming_transcription.ipynb`。
2. 先執行前置斷言，確認 `processor` 與 `model` 都已在 Cell 1 載入。

   ```python
   assert "processor" in globals() and "model" in globals()
   ```
3. 上傳一支**最長可達 1 小時**的音檔（Colab 可上傳檔案大小有上限，若 >100MB 建議先 copy 到 Google Drive，再以路徑載入）。
4. 直接按「執行」整顆 Cell，Notebook 會自動：

   1. 計算音檔總 sample、分段數（ n\_chunks ）。
   2. 針對每段做重疊切片、特徵抽取、Whisper 推論。
   3. 拼接文字並輸出到 `*.txt`。
5. 執行結束後，左側「檔案」面板會看到新的 `*_1h_transcript.txt`，點擊即可下載。

---

### 4.4 Optional: Inspect XLA Metrics (Step 3)

1. 開啟 `03_metrics_inspection.ipynb`。
2. 按步驟執行，確認每 N 塊 chunk 完成後呼叫 `short_metrics_report()`，觀察 `ExecuteTime>0`、`TransferTime` 等指標。
3. 若發現 `ExecuteTime=0` 或多次 `Op(s) not lowered` → 代表該運算 fallback CPU，須檢查是否安裝或升級到支持該 op 的 PyTorch/XLA 版本，或改為 Kaggle TPU v4/v5 VM。

---

## 5. How It Works

### 5.1 Whisper’s 30s Input Limit

* **原理**：Whisper 是 Seq2Seq 架構，**encoder** 一次能處理 **1500** 個 log-mel frames，對應 ≈ 30 秒 音訊。
* **影響**：若輸入聲音長度 > 30 秒，`processor.feature_extractor` 會自動截斷至前 30 秒；若 < 30 秒 → 自動填充（padding）到 30 秒。
* **參考**：

  > > “`max_source_positions (int, defaults to 1500)`: The maximum sequence length of log-mel features that the model can process (≈ 30 秒). If your audio is longer, it will be truncated.”

### 5.2 Overlap Sliding Window Strategy

* **為何重疊**：

  * 直接把音檔切成連續的 30 秒集會產生「句子中段斷裂」問題，導致拼接時斷字不連續。
  * **左右各重疊 5 秒**：第 i 段實際讀的是 `[i*30 s−5 s, i*30 s+30 s+5 s]` = `[i*30−5, i*30+35]`，但特徵抽取僅保留該區段「前 30 秒」＝ `[i*30−5, i*30+25]`。
  * 第 (i+1) 段保留 `[ (i+1)*30−5 , (i+1)*30+25 ]` = `[i*30+25, i*30+55]`→ 前 5 秒與上段交疊 (`[i*30+25, i*30+30]`)，有重疊即可對齊句子，減少斷裂。
* **實作重點**：

  1. `seg_start = max(0, mid_start−STRIDE_SMP)`
  2. `seg_end = min(mid_start+CHUNK_SMP+STRIDE_SMP, total_samples)`
  3. `processor.feature_extractor(audio)` → 先做 **truncation**（若超過 1500 frames）或 **padding**（若不足），最終傳給 model 的永遠是 30 秒。
* **參考**：

  > > “Whisper feature extractor first pads/truncates any audio to 30 秒, then converts to log-Mel spectrogram.”

### 5.3 TPU Caching & Metrics

* **Dummy 推論**：

  1. 在 Cell 1 執行一次 `model.generate(...)` → 真正觸發 XLA 編譯（CompileTime 約 20–60 秒）。
  2. 清空 metrics 並 `xm.mark_step()`，確保後續 chunk 使用同一 cached graph。
  3. 第二次以後的 `generate` 只走 `ExecuteTime`，且 `CachedCompile > 0`、`UncachedCompile=1`、`ExecuteTime>0`。
* **關鍵指標**：

  * **`ExecuteTime`**：每次 chunk 實際執行在 TPU 上的累計時間；非零代表真的跑在 TPU。
  * **`TransferToDeviceTime`** / **`TransferFromDeviceTime`**：Host↔TPU 傳輸時間，通常是數十 μs 到數十 ms。
  * **`aten::xxx` Counter**：若看到大量 `aten::` 前綴的 ops，代表那些小操作 fallback 回 CPU，需要留意是否有瓶頸。

---

## 6. Customization

### 6.1 調整 Chunk/Stride 大小

* 若覺得 `STRIDE_SEC=5 秒` 過多/過少，可自行修改。**重疊時間要保證 i 段 (i+1) 段文字能在該範圍內有足夠語意銜接**。

  * 若過短（如 `STRIDE_SEC=2`），重疊區段語意連貫可能不足 → 句子仍會斷裂。
  * 若過長（如 `STRIDE_SEC=10`），雖然連貫度高，但重疊段落冗餘度增加、總推論次數不變，但每次讀取的 audio 更長（42 秒 vs 40 秒），對 I/O 有輕微影響。

### 6.2 後處理 / 去重疊（Deduplication）

* **最簡版**：直接 `full_transcript = "".join(segments)`，保留所有重疊文字 → 較快。
* **進階做法**：

  1. 在 `batch_decode(..., return_timestamps=True)` 時取得每個 token 的 `(start_time, end_time)`。
  2. 按 chunk 的實際「時間範圍」(mid−5s → mid+25s)，刪除早於 `prev_chunk_end` 的 token。
  3. 串接剩餘文字，即可避免 5 秒重疊區段的重複輸出。
* **示例**：可以參考 `scripts/postprocess.py` 內的時間戳對齊範例。

### 6.3 使用 GPU 而非 TPU

* 若 Colab 沒有取得 TPU，想在 Colab GPU T4 上執行：

  1. 在 Colab 選擇「Runtime → Change runtime type → Hardware accelerator: GPU」。
  2. 將 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
  3. 不需安裝 `torch_xla`，而改安裝 `torch` 對應 GPU 版本即可。

> [點我前往 Colab Notebook](https://colab.research.google.com/drive/1ljb2EBTgmzr3QoJ61M4QNHivFgyMGx4H?usp=sharing)

---

## 7. Requirements

在 `requirements.txt` 中列出：

```
torch==2.6.0+cpu.cxx11.abi
torch_xla==2.6.0        # 僅在 TPU 模式需要
transformers>=4.39.0,<4.40.0
sentencepiece
librosa
soundfile
numpy
```

* 若只使用 GPU，可以移除 `torch_xla==2.6.0`。
* 建議同時安裝 `xla[tpu]` 以確保所有依賴符合 PyTorch/XLA 在 TPU VM 上的需求。



## 8. License

本專案建議採用 **MIT License**，在 `LICENSE` 放入以下內容：

```text
MIT License

Copyright (c) 2025 蔡秀吉

Permission is hereby granted, free of charge, to any person obtaining a copy
...


## 9. Acknowledgements

* 秀吉的肝還有國立陽明交通大學博雅書苑的學生自主學習計畫
* 感謝 OpenAI Whisper 團隊提供的強大語音轉文字模型和 ChatGPT 團隊。
* 感謝 PyTorch/XLA 開發者社群提供完整 TPU 支援與 Metrics 工具。
* 特別致謝 Colab 團隊持續提供免費 TPU 資源，讓我們能以低成本完成推論試驗。
