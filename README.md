# High-Performance Whisper Transcription on Google Colab TPU

> 點我使用：[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FUrfNPB7mOfCz-DNYwtkNMXV0J1gwfma)

> 本專案示範如何在 Google Colab 上，利用 TPU 加速 OpenAI Whisper 模型，對長音檔（可達數小時）進行分段轉錄，並自動輸出文字檔與效能統計 (RTF)。

## 專案介紹
在現代工作或研究裡，會議錄音、訪談錄音或上課錄影常常要轉成文字，但用自己電腦跑一整天都還沒跑完，或者只能去用那些要錢的雲端服務。這個專案就利用 Google Colab 提供的免費 TPU，再搭配 PyTorch/XLA 的高效能編譯，讓你能在短時間內把「長音檔拆段轉錄」，還自動幫你算出 Real-Time Factor (RTF) 來看速度到底快多少。整個流程只要在 Colab 裡面跑三個儲存格 (Cell)，就能體驗到超順暢、快如閃電的 Whisper 轉錄。

## 主要特色
- 🚀 **TPU 加速**：以 PyTorch/XLA 將 Whisper 模型部署到 Colab TPU，遠快於一般 CPU/GPU。  
- 🧠 **支援多種 Whisper 模型**：從 `tiny`、`base`、`small`、`medium` 到 `large-v2`/`large-v3`，任意切換。  
- 🪡 **長音檔分段處理**：自動將音檔每 30 秒切成一小段，逐段推論，避免一次載入過大導致 OOM。  
- 🌐 **多語言與翻譯**：Whisper 原生支援 100 多種語言，並可選擇「transcribe」(原語言轉錄) 或「translate」(直接譯成英文)。  
- ⚡ **BF16 混合精度**：預設使用 TPU 原生 `bfloat16` 精度，大幅提升運算效能且幾乎不影響準確度。  
- 📊 **自動計算 RTF**：轉錄完成後，程式會自動計算「Real-Time Factor」，讓您知道處理速度。  
- 🎛️ **一鍵安裝與熱機**：Cell 1 自動安裝所有 Dependency；Cell 2 進行 XLA 編譯 Warm-up，避免首次推理太慢。  
- 📂 **多格式音訊**：支援 `.mp3`、`.wav`、`.flac`、`.ogg`、`.m4a` 等常見音檔格式。  
- 💾 **輸出結果自動儲存**：每個音檔會在 Colab Notebook 左側 **檔案** 目錄下產生 `[原始檔名]_transcript.txt`，方便下載。  


## 先決條件
1. **Google 帳號**：能夠登入並使用 Google Colab。
2. **Chrome / Firefox 最新版瀏覽器**：避免瀏覽器相容問題。
4. **音訊檔**：準備要轉錄的音檔，格式建議為 PCM WAV、MP3、FLAC、M4A 等。
5. **GitHub 帳號（選用）**：如果要 clone 本儲存庫或在本地執行 Notebook。

## 專案目錄結構

```
whisper-tpu-stream-transcription/
├── README.md
├── LICENSE
├── whisper-tpu-colab-longform.ipynb
├── whisper-gpu-colab-longform.ipynb
└── whisper_tpu_v2-8_longform_optimized.ipynb
```

* **`README.md`**：本檔案，包含完整使用指南與常見問題。
* **`LICENSE_MIT.md`**：MIT 授權條款全文。
* **`requirements.txt`**：列出 Python 套件依賴，例如：

### Requirements
```
torch==2.7.0
torch-xla==2.7.0
transformers==4.35.0
accelerate==0.21.0
librosa==0.10.0
soundfile==0.12.1
numpy==1.24.4
```

## 快速開始
1. **Clone 本儲存庫（可選，本地端預覽）**  

```bash
git clone https://github.com/thc1006/whisper-tpu-colab-longform.git
cd whisper-tpu-colab-longform
```

2. **在瀏覽器打開 Colab Notebook**
   點擊此徽章開啟 Notebook：[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FUrfNPB7mOfCz-DNYwtkNMXV0J1gwfma)

3. **切換 TPU 硬體加速器**
在 Colab 中，依次點擊：

     ```
     執行階段 (Runtime) → 變更執行階段類型 (Change runtime type) 
     → Hardware accelerator → TPU → 儲存 (Save)
     ```
     > 此時左上方應顯示 `TPU`，代表已成功切換。

4. **執行 Cell 1：安裝所有 Dependency**

   * 按下 Cell 1 左側的 ▶️ 按鈕，開始安裝 `torch-xla`、`transformers`、`librosa` 等。
   * Cell 1 成功執行完畢後，**一定要手動點選**：**執行階段 (Runtime) → 重新啟動工作階段 (Restart runtime)**
   * 重啟後，輸出畫面會清空，表示新的 kernel 已載入安裝好的套件。

5. **重新執行 Cell 1**

   * 再次按 ▶️ 執行 Cell 1，此步驟主要確認安裝與初始化（此次不需重啟）。
   * 輸出訊息應顯示各套件版本與安裝成功。

6. **執行 Cell 2：載入模型與 TPU Warm-up**

   * 按 ▶️ 執行 Cell 2，完成以下動作：

     1. `import torch_xla`、`import transformers`、`import numpy` 等。
     2. 下載並載入 `openai/whisper-medium`（預設）。
     3. 模型搬到 TPU；
     4. 以 5 秒全靜音片段進行一次 dummy 推論，觸發 XLA 編譯。
   * 輸出訊息示例：

     ```
     ✅ torch_xla 相關模組匯入成功。
     📥 正在下載並載入模型：openai/whisper-medium…
     ✅ ASR Pipeline 初始化完成！
     🔁 開始執行 XLA 暖機，請稍候…
     ✅ XLA 暖機完成，TPU 編譯已就緒！
     ```

7. **執行 Cell 3：上傳音檔 & 開始轉錄**

   * 按 ▶️ 執行 Cell 3，程式會提示：

     ```
     📤 請上傳音檔 (mp3 / wav / m4a / ogg / flac …)
     您可以一次選擇多個檔案。
     ```
   * 點擊「Choose Files」按鈕，選擇本地音檔（可同時選多個）。
   * 上傳完成後，Notebook 會自動依序載入、重採樣、切片、送入 Whisper Pipeline 推論，並把結果寫入 `[原始檔名]_transcript.txt`。同時會顯示每段耗時與最終 RTF，例如：

     ```
     ▶️ 正在轉錄第 1/4 段 (時刻 0.00–30.00 秒)…  耗時 12.90 秒  
     ...  
     🏁🏁🏁 檔案 sample_audio.mp3 全部分段處理完畢！🏁🏁🏁  
       總音訊時長: 120.00 秒  
       總轉錄耗時: 50.00 秒  
       整體即時率 (RTF): 0.417
     ```
   * 轉錄完後，點擊左側檔案視窗，可看到檔案 `sample_audio_transcript.txt`，右鍵即可下載。

8. **下載轉錄文字檔**

   * 完成後，Colab 左側「檔案」面板中會顯示所有 `.txt` 輸出檔。
   * 點擊檔案右鍵 → 選「下載」，即可取得純文字轉錄結果。

## 常見問題
### Q1：為何會出現 `ImportError: No module named 'torch_xla'`？

* **原因**：Cell 1 執行後，尚未「重啟工作階段 (Restart runtime)」。
* **解決方法**：

  1. 確認執行完 Cell 1 之後，手動點選 `執行階段 (Runtime) → 重新啟動工作階段 (Restart runtime)`。
  2. 重新啟動後，再從 Cell 1 開始依序執行所有儲存格。

### Q2：為何 Cell 2 的「XLA 暖機 (Warm-up)」要花非常久？

* **原因**：Whisper Medium 以上模型較複雜，初次在 TPU 上編譯（XLA JIT）可能需要 5～10 分鐘甚至更久。
* **解決方法**：

  1. 如果不急，可耐心等待；
  2. 若時間不足，可改選用體積較小的 Whisper 小模型 (`tiny`、`base`、`small`)，Warm-up 會更加快速。

### Q3：為何在執行 Cell 3 時出現 OOM（Out Of Memory）？

* **原因**：Whisper Medium/Large 模型載入在 TPU 上的記憶體已滿。
* **解決方法**：

  1. 改用體積較小的模型，例如：`openai/whisper-small` 或 `openai/whisper-base`；
  2. 減少分段長度，例如把 `segment_length_s` 從 30 改成 15；
  3. 如掛載 Drive 讀取更大檔案，避免一次上傳大量資料進行處理。

### Q4：如何解釋 Real-Time Factor (RTF)？如何判斷速度？

* **定義**：RTF = 總轉錄耗時(秒) ÷ 總音訊時長(秒)

* **意義**：

  * RTF < 1：代表「提前」，即處理 1 秒音訊所需時間 < 1 秒，例如：RTF=0.2 表示轉錄 1 小時音訊只需 12 分鐘。
  * RTF > 1：代表「延遲」，處理速度慢於實際播放時長。
* **評估**：越小表示效能越好。使用大模型時 RTF 會較高，但準確度也更高；使用小模型時 RTF 低，但可能犧牲部分辨識率。

### Q5：如果要處理多個檔案，能否一次上傳多個？

* **答案**：可以。在 Cell 3 中，`files.upload()` 支援多選檔案，上傳完成後 `uploaded_files` 會是一個字典，包含多個音檔。程式會迴圈依序執行每個檔案的轉錄流程，最終各自輸出對應文字檔。


## 其他進階功能

1. **修改分段長度 (Segment Length)**

   * 預設為 `segment_length_s = 30.0`（單位：秒）。
   * 若要改為 60 秒一段，編輯 Cell 3 中：

```python
segment_length_s = 60.0
```
   
   * 早期段數減少，推論次數少，但每次推論佔用記憶體更高，可能導致 OOM。

2. **更換 Whisper 模型**

   * Cell 2 預設：

```python
MODEL_NAME = "openai/whisper-medium"
```
   * 若想更快速度，可試：

     * `"openai/whisper-small"` (精度略低但速度提升)
     * `"openai/whisper-base"`
     * `"openai/whisper-tiny"` (最快但最不準確)
   * 若追求最高準確度，可改成：

     * `"openai/whisper-large-v2"` 或 `"openai/whisper-large-v3"` (每段推論耗時明顯增加，但文字品質最佳)。

3. **改用 CPU 或 GPU（非 TPU）**

   * 若 Colab 沒分配到 TPU，可在 Cell 2 中插入：

     ```python
     try:
         device = xm.xla_device()
         tpu_device = True
     except Exception:
         print("⚠️ 無法取得 TPU，改用 GPU 或 CPU")
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         tpu_device = False
     ```
   * 接著在建立 pipeline 時：

     ```python
     pipeline(
         "automatic-speech-recognition",
         model=model,
         tokenizer=processor.tokenizer,
         device=  0 if tpu_device else (-1 if device.type=="cpu" else 0),
     )
     ```
   * 必要時移除所有 `xm.mark_step()`、`xm.collect_metrics()` 呼叫。

4. **逐塊載入 (Streaming) 讀取 (進階)**

   * 若音檔超過 2 小時，建議別用 `librosa.load` 一次全部讀進記憶體，改用 `soundfile` 逐塊載入。
   * 範例程式放在 `examples/streaming_sample.py`，主要思路：

     ```python
     import soundfile as sf
     CHUNK_SEC = 30
     with sf.SoundFile("long_audio.wav") as f:
         sr = f.samplerate
         chunk_frames = CHUNK_SEC * sr
         segment_idx = 0
         while True:
             data = f.read(frames=chunk_frames, dtype="float32")
             if len(data)==0:
                 break
             # 如果音訊非 16k，先做 resample
             # 然後呼叫 asr_pipeline(data, sampling_rate=sr)
             segment_idx += 1
     ```
   * 這樣每次只載入 30 秒的框架資料到記憶體，RAM 使用大幅降低。

---

## License
```text
MIT License

Copyright (c) 2025 Hsiu-Chi Tsai (蔡秀吉)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
```

## 9. Acknowledgements

* 秀吉的肝還有國立陽明交通大學博雅書苑的學生自主學習計畫
* 感謝 OpenAI Whisper 團隊提供的強大語音轉文字模型和 ChatGPT 團隊。
* 感謝 PyTorch/XLA 開發者社群提供完整 TPU 支援與 Metrics 工具。
* 特別致謝 Colab 團隊持續提供免費 TPU 資源，讓我們能以低成本完成推論試驗。

---

> **最後更新日期**：2025-06-05
> **作者**：蔡秀吉 (Tsai Hsiu-Chi)
> **電子郵件**：\[[hctsai@linux.com](mailto:your_email@example.com)] (可選)
> **GitHub**：[@thc1006](https://github.com/thc1006)
