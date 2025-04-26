# OpenShift Forward - AI Companion

OpenShift Forward (OSF) is an AI-first application to manage an OpenShift cluster alongside OpenShift Container Platform (OCP) web console.  Rather than "shift left" or "shift right", we want to "shift forward"!  Note: Support for Kubernetes cluster is limited.

&#x26a0; Beware: This is a prototype only.  **DO NOT USE IN PRODUCTION!**

Log into your cluster and give it prompts.  It will create YAMLs for you.  You can edit the YAML to your liking and then apply the changes.  **WARNING:** YAML creation is a work in progress!

## Demo

Try it live at: [https://kubecloud.site](https://kubecloud.site)

## Install and Setup

Environment: This should work on any Linux-based, MacOS, or WSL environment.

Prerequisites:
- OpenAI key

- Binary copy of [`oc`](https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/html/cli_tools/openshift-cli-oc) (OpenShift/Kubernetes) or [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) (Kubernetes only)

1. Install packages
    ```
    # Debian environment
    sudo apt update
    sudo apt install python3 python3-pip npm ffmpeg docker.io
    ```

2. (Optional) Set up virtual environment

    1.1 Use either Python's venv (or Anaconda).

    ```
    # venv
    sudo apt install python3.12
    python3.12 -m venv pyenv
    ```

    1.2 Activate the virtual environment.

    ```
    source pyenv/bin/activate
    ```

3. Set up Docker

    - The user running the Flask app must have access to the `docker` command without requiring sudo privileges.
    ```
    sudo usermod -aG docker user  # replace user with the actual user
    # Log out and log back in for it to take effect.
    ```

    - Copy the binary [`oc`](https://docs.redhat.com/en/documentation/openshift_container_platform/4.18/html/cli_tools/openshift-cli-oc) (OpenShift/Kubernetes) or [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) (Kubernetes only) into this directory.  If you are using `kubectl`, change the **Dockerfile** and replace the `COPY` and `RUN` lines with `COPY kubectl /usr/local/bin/kubectl`.

    - Create the Docker container.
    ```
    cd docker
    ./run.sh
    docker images  # verify container exists
    ```

4. Install Python packages

    ```
    cd ../backend
    pip install -r requirements.txt

    # if running Python 3.13+
    pip install audioop-lts
    ```

5. Set up OpenAI key and Flask key

    Create a file named ".env" in the **backend** directory.  Replace with your keys.  It should contain:

    ```
    OPENAI_API_KEY='<your_openai_key>'
    FLASK_SECRET_KEY='<some_random_key>'
    ```

    FLASK_SECRET_KEY can be any arbitrary value.

6. Set up frontend

    ```
    cd ..
    npm create vite@latest frontend -- --template react
      > Choose "Ignore files and continue".
    cd frontend
    npm install
    npm install lucide-react

    rm -rf .gitignore README.md eslint.config.js public src/App.css src/assets
    git restore index.html src/App.jsx src/index.css src/main.jsx
    ```

7. (Optional) Set up Ollama

    - Follow instructions to [install Ollama](https://github.com/ollama/ollama).
    - Get some models using `ollama pull <model_library>`.  In OSF, you should see the models by clicking the Settings icon in the upper right corner.

    Note: In OSF, your LLM selection is not preserved and will be reset on a new session.


## Run application

1. (Optional) Switch to your virtual environment

    ```
    cd ..
    # if not done already
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

    Note: If you want to change the port, take a look at _frontend/vite.config.js_ and _backend/app.py_.

    Happy shifting! &#x1f600;
