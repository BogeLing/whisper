# Whisper.cpp Python Wrapper for macOS (Apple Silicon)

A high-performance Python wrapper for [whisper.cpp](https://github.com/ggerganov/whisper.cpp), optimized for Apple Silicon (M1/M2/M3/M4) with Metal GPU acceleration.

## Features

- 🚀 **Metal Acceleration**: Fully utilizes Apple Neural Engine & GPU via `whisper.cpp`.
- 🧠 **Smart Segmentation**: Custom logic to prevent broken sentences and ensuring optimal subtitle length (90 chars).
- 🔄 **Smart Caching**: Caches JSON results to allow instant parameter tuning (subtitle length, gap thresholds) without re-running the heavy inference.
- 📂 **Organized Output**: Automatically archives `.srt`, `.txt`, and `.json` files into a dedicated `_output` folder.
- ⏳ **Accuracy**: Fixes standard Whisper timestamp issues where punctuation could extend segment duration excessively.

## Prerequisites

### 1. System Requirements (macOS)
You need to install `whisper-cpp` and `ffmpeg` via Homebrew:

```bash
brew install whisper-cpp
brew install ffmpeg
```

### 2. Python Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the example script:

```bash
python 使用示例.py
```

### Configuration
You can modify `whisper_cpp_wrapper.py` to tune parameters:
- `MAX_CHARS`: Maximum characters per subtitle line (Default: 90).
- `threads`: CPU threads (Default: 10, optimized for M4).

## Project Structure
- `whisper_cpp_wrapper.py`: Core logic class.
- `使用示例.py`: Example usage script.
- `models/`: Stores downloaded GGML models (e.g., `ggml-large-v3.bin`).
- `*_output/`: Generated results.
