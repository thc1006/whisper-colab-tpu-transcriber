# Whisper Speech Transcription Colab Notebook (TPU + PyTorch/XLA Accelerated)

> **Press here into**：[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VxyV0rTyYpJ4CClTMNpatgVAOag-JrV4) Colab Notebook.

🚀 This Google Colab Notebook provides an efficient and accurate solution for audio transcription and translation using OpenAI's Whisper model, accelerated on TPUs with PyTorch/XLA. It features an interactive user interface for easy customization of models, languages, and long-form audio processing parameters.

## ✨ Key Features

  * **High-Performance Transcription:** Significantly speeds up transcription using Google Colab's TPUs and PyTorch/XLA.
  * **Multiple Whisper Models:** Supports various Whisper model sizes, from `tiny` to `large-v3`, allowing a trade-off between speed and accuracy.
  * **Flexible Language Options:** Offers automatic language detection and manual selection for dozens of languages (including English, Chinese, Japanese, Korean, Spanish, French, German, etc.), plus custom ISO code input.
  * **Transcription & Translation:** Perform speech-to-text in the original language (`transcribe`) or translate speech into English (`translate`).
  * **Optimized Compute Precision:** Recommends `BF16` for optimal performance on TPUs, with `FP32` support.
  * **Long-Form Audio Processing:** Implements chunking and striding mechanisms to effectively handle audio files longer than 30 seconds.
  * **Interactive User Interface:** User-friendly GUI powered by `ipywidgets` for easy configuration of all transcription parameters.
  * **XLA Warm-up:** Automatically performs an XLA warm-up step to compile the computation graph and optimize subsequent transcription performance.
  * **Automated Environment Setup:** Handles the installation of necessary Python package dependencies, including `torch_xla` and `ffmpeg`.

## 📋 Prerequisites

  * A Google Account (for accessing Google Colab).
  * Basic understanding of Google Colab operations.
  * (Optional) A GitHub account if you wish to save modified versions of the notebook to your own repository.

## 🚀 Getting Started

### 1\. Open the Notebook

Click the [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VxyV0rTyYpJ4CClTMNpatgVAOag-JrV4) badge above to open this notebook in Google Colab.

> **Recommended:** Save a copy to your Google Drive by clicking **File** at the upper-left corner of the screen and selecting **Save a copy in Drive**. Then, rename the copied file (`FILENAME.ipynb`) to any name you prefer.


### 2\. Configure the Runtime

For optimal performance, it's recommended to use a TPU hardware accelerator:

1.  In the Colab menu, select **Runtime** -\> **Change runtime type**.
2.  Under "Hardware accelerator," choose **TPU**.
3.  Click **Save**.

### 3\. Run Cell 1: Install Packages & UI Setup

This cell performs the following actions:

1.  Installs all necessary Python packages, including `torch`, `torch_xla`, `transformers`, `ffmpeg`, etc.
2.  **‼️ IMPORTANT:** After this cell finishes its first execution, Colab will prompt you to **Restart session**. You MUST click the button in the prompt or manually go to **Session -\> Restart session** (or **Runtime -\> Restart session**).
3.  After restarting the session, **run Cell 1 again**. This time, it will skip the lengthy installation and display the interactive UI for configuring transcription parameters.
4.  Adjust the settings in the UI according to your needs:
      * **Whisper Model:** Choose the model size (e.g., `small`, `medium`, `large-v3`).
      * **Transcription Language:** Select the language of your audio, or leave it as `auto` for automatic detection. If "Other" is selected, enter the [ISO 639-1 format](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) (e.g., `de` for German) in the text box below.
      * **Task:** `transcribe` (speech-to-text in original language) or `translate` (translate speech to English).
      * **Compute Precision:** If using TPU, `bf16` is recommended. For CPU/GPU, use `fp32`.
      * **Long Audio Processing (Advanced):**
          * **Audio Chunk Length (s):** Duration of chunks for processing long audio (default 28-30s).
          * **Left/Right Overlap (s):** Overlap between chunks to maintain context.

### 4\. Run Cell 2: Load Model & XLA Warm-up

This cell will:

1.  Load the specified Whisper model and its corresponding Processor from Hugging Face Hub based on your selections in Cell 1.
2.  Move the model to the TPU (if available).
3.  Initialize the Automatic Speech Recognition (ASR) Pipeline.
4.  If using a TPU, it will perform a "warm-up" step. This involves compiling the XLA computation graph with a short dummy audio. This might take a few minutes but significantly speeds up subsequent processing of actual audio.

**Please be patient while this cell executes, especially the warm-up step.**

### 5\. Run Cell 3: Upload Audio & Transcribe

This cell will:

1.  Prompt you to upload one or more audio files (supports common formats like mp3, wav, m4a, ogg, flac).
2.  Process each uploaded file for transcription.
3.  During processing, it will display the audio duration, transcription time, and Real-Time Factor (RTF) for each file. A lower RTF indicates faster processing (RTF \< 1 means faster than real-time).
4.  After transcription, a preview of the result will be shown.
5.  The full transcript will be saved as a `.txt` file, named like `[original_filename]_transcript_[model_size]_[language].txt`.
6.  You can find and download these `.txt` files from the "Files" panel (folder icon) on the left side of Colab.
7.  Once all files are processed, overall statistics and final TPU memory usage will be displayed.

## 🛠️ Technical Details

### PyTorch/XLA and TPUs

This notebook leverages PyTorch/XLA (Accelerated Linear Algebra) to enable PyTorch models to run efficiently on Google's Tensor Processing Units (TPUs). TPUs are specialized hardware designed for large-scale machine learning computations. Using them with XLA can significantly accelerate inference for large models like Whisper. We recommend using `bfloat16` (BF16) mixed precision on TPUs to maximize performance and reduce memory footprint while maintaining acceptable accuracy.

### Long-Form Audio Processing

OpenAI Whisper models have an input audio length limit (typically around 30 seconds). To handle longer audio files, the Hugging Face `transformers` pipeline implements a chunking and striding strategy:

  * **`chunk_length_s`:** Long audio is divided into shorter chunks (e.g., 28 seconds).
  * **`stride_length_s`:** An overlap is set between consecutive chunks (e.g., 5 seconds on each side). This overlap helps the model maintain contextual coherence at chunk boundaries, reducing information loss or transcription errors due to segmentation.

You can adjust these parameters in the Cell 1 UI to suit different types of audio.

### ipywidgets Interface

For a more user-friendly experience, this notebook uses the `ipywidgets` library to create interactive controls. This allows users to easily adjust various transcription parameters—such as model selection, language, task type, and long-form audio settings—without directly modifying the code.

## 🔍 Troubleshooting

  * **`ModuleNotFoundError: No module named 'torch_xla'` or related XLA errors:**

      * **Solution:** Ensure you have **correctly restarted the Colab session** after Cell 1's first execution, then re-run Cell 1 and Cell 2. This is the most common cause.
      * Verify that the Colab runtime type is set to TPU.

  * **Out Of Memory (OOM) errors:**

      * **Solution:**
        1.  Try selecting a smaller Whisper model (e.g., `medium`, `small`, `base`, or `tiny`). `large` series models require more memory.
        2.  Ensure compute precision is set to `bf16` when on TPU.
        3.  "Restart session" to free all allocated resources, then run all cells from the beginning.

  * **XLA warm-up takes too long or fails:**

      * **Solution:**
        1.  Be patient; initial compilation, especially for larger models, can take a few minutes.
        2.  Check the package installation logs in Cell 1 for errors.
        3.  Try testing with a smaller model.
        4.  Ensure a stable internet connection for downloading model files.

  * **File upload issues:**

      * **Solution:** Ensure your internet connection is stable. If uploading large files causes issues, try uploading them in smaller batches or check Colab's file size limits.

  * **Poor transcription results:**

      * **Solution:**
        1.  Try specifying the correct audio language instead of relying on auto-detection.
        2.  For audio with multiple languages or heavy accents, try a larger model (e.g., `large-v3`) for better accuracy.
        3.  Check audio quality; excessive background noise or poor recording quality will affect results.
        4.  Adjust the `chunk_length_s` and `stride_length_s` parameters for long-form audio processing.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## 🙏 Acknowledgements

* My liver and the Student Community Active Learning Program of the Liberal Arts College at National Yang Ming Chiao Tung University.
* The OpenAI Whisper team for providing the powerful speech-to-text model, and the ChatGPT team
* The PyTorch/XLA developer community for full TPU support and metrics tools
* Special thanks to the Google Colab team for continuously providing free TPU resources, allowing us to run inference experiments at low cost
* [Hugging Face](https://huggingface.co/) for the `transformers` library and model hosting.

---

> **最後更新日期**：2025-06-05
> **作者**：蔡秀吉 (Tsai Hsiu-Chi)
> **電子郵件**：\[[hctsai@linux.com](mailto:your_email@example.com)] (可選)
> **GitHub**：[@thc1006](https://github.com/thc1006)
