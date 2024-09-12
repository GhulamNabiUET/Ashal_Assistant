# Ashal 🌟✨ The Light of Knowledge 💬 Talking Assistant 🎙️🤖

This repository contains the implementation of a voice assistant that transcribes audio inputs, processes queries using a Retrieval-Augmented Generation (RAG) pipeline, and provides responses in audio form.

## Features
- Record audio inputs using Streamlit
- Transcribe audio to text using Whisper model
- Process queries with a Retrieval-Augmented Generation pipeline
- Convert text responses to audio using gTTS

## Installation
1. Clone the repository:
    ```bash
    https://github.com/GhulamNabiUET/Talking-Assistant-.git
    ```
2. Navigate to the project directory:
    ```bash
    cd voice_assistant
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Install `ffmpeg`:
    - **Ubuntu/Debian**:
        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```
    - **MacOS** (using Homebrew):
        ```bash
        brew install ffmpeg
        ```
    - **Windows**:
        - Download `ffmpeg` from the [official website](https://ffmpeg.org/download.html).
        - Extract the contents of the ZIP file.
        - Add the `bin` directory to your system’s PATH environment variable.
    - **Docker**:
        - Add the following to your Dockerfile:
            ```Dockerfile
            RUN apt-get update && apt-get install -y ffmpeg
            ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app/main.py
    ```

2. Open the app in your web browser, record your audio query, and get the response in audio form.

## License
This project is licensed under the MIT License.
