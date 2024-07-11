# AI GENERATED STORIES

<span style="font-size:2em;">A tool that creates stories written by an LLM</span>

## How to use

1. Download this repo to a local directory
2. Make sure you have python ^3.10 active
3. Install required python libraries, they can be found in **pyproject.toml**. I suggest you use [poetry](https://python-poetry.org/) for easy dependency management
4. Download a gguf of your favorit LLM and put it in the **models** folder. Tested and working models are:
    - [gemma-2-9b-it-IQ4_XS.gguf](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/blob/main/gemma-2-9b-it-IQ4_XS.gguf)
    - [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
    - [Phi-3-mini-128k-instruct.IQ4_XS.gguf](https://huggingface.co/PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed/blob/main/Phi-3-mini-128k-instruct.IQ4_XS.gguf)
5. From a terminal, cd to the local directory of this repo and run either
    - ```bash
        python create_story.py # model keep generating paragraphs until you say stop
        ```
    - ```bash
        python create_story_w_feedback.py # model creates 1 paragraph at a time with an option for human feedback
        ```

