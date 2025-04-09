# OpenShift Forward - AI Companion

OpenShift Forward (OSF) is an AI-first application to manage an OpenShift Container Platform (OCP).  Rather than "shift left" or "shift right", we want to "shift forward"!

Log into your cluster and give it prompts.  It will create YAMLs for you.  You can edit the YAML to your liking and then apply the changes.  &#x26a0; Beware: This is a prototype only.  **DO NOT USE IN PRODUCTION!**

## Demo

Try it live at: [https://kubecloud.site](https://kubecloud.site)

## Install and Setup

Environment: This should work on any Linux-based, MacOS, or WSL environment.

Prerequisites:
- Install Python, pip, npm, and ffmpeg.
- OpenAI key

1. (Optional) Set up virtual environment

    1.1 Use either Python's venv (or Anaconda).

    ```
    # venv
    python3 -m venv pyenv
    ```

    1.2 Activate the virtual environment.

    ```
    # venv
    source pyenv/bin/activate
    ```

2. Install packages

    ```
    cd backend
    pip install -r requirements.txt
    ```

3. Set up OpenAI key and Flask key

    Create a file named ".env" in the **backend** directory.  Replace with your keys.  It should contain:

    ```
    OPENAI_API_KEY='<your_openai_key>'
    FLASK_SECRET_KEY='<some_random_key>'
    ```

4. Set up frontend

    ```
    npm create vite@latest frontend -- --template react
    cd frontend
    npm install

    rm -rf .gitignore README.md eslint.config.js public src/App.css src/App.css src/assets
    ```

## Run application

1. (Optional) Switch to your virtual environment

    ```
    source pyenv/bin/activate
    ```

2. Run app in development mode

    ```
    cd backend
    python app.py
    ```

    In another session,

    ```
    cd frontend
    npm run dev
    ```

    Point your browser at: [http://localhost:5173](http://localhost:5173)

    Note: If you want to change the ports, take a look at _frontend/vite.config.js_ and _backend/app.py_.
