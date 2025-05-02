import atexit
import io
import os
import re
import requests # Needed for checking Ollama server reachability
import shlex
import shutil
import stat
import sys
import tempfile
import traceback
import uuid
import yaml # Still needed for YAML parsing in /submit route

from urllib.parse import urlparse, urlunparse # Needed in /login route

import docker
from docker.errors import APIError, NotFound, ImageNotFound

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

# --- Kubernetes client imports ---
from kubernetes import config
from kubernetes import client
from kubernetes.config.config_exception import ConfigException
from kubernetes.client.exceptions import ApiException

from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment

# --- LangChain/LLM Imports ---
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama # Assuming you updated this
# from langchain_ibm import ChatWatsonx # Placeholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# --- End LangChain/LLM Imports ---

# --- Local Imports ---
from audio import speech_to_text
from rag_setup import load_or_build_vector_store
from prompt import SYSTEM_PROMPT, RAG_TASK_SYSTEM_INSTRUCTION, SUBMIT_SCRIPT_PROMPT
from helper_functions import run_in_container, cleanup_kube_configs, cleanup_exec_dir
from helper_functions import CLI_DOCKER_IMAGE, OSF_KUBECONFIG_DIR, OSF_EXEC_DIR # Import constants
# --- End Local Imports ---


load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'mysecret')
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://127.0.0.1:5173"]) # Adjust port if needed
API_PREFIX = '/api'

# --- Default LLM Configuration ---
DEFAULT_PROVIDER = 'openai'
DEFAULT_OPENAI_MODEL = 'gpt-4o-mini' # Changed default
# DEFAULT_OLLAMA_MODEL = 'llama3:8b' # Example if Ollama is default
# DEFAULT_IBM_MODEL = 'ibm/granite-13b-chat-v2' # Example

# --- Environment Variable Configuration ---
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
# IBM_API_KEY = os.environ.get('IBM_API_KEY') # Add if using IBM
# IBM_ENDPOINT = os.environ.get('IBM_ENDPOINT', 'https://us-south.ml.cloud.ibm.com') # Add if using IBM

temperature = 0
enable_rag = True

ACTION_MARKER = "/OSF_ACTION:" # Define the marker
CONTAINER_EXEC_DIR = "/app_temp" # Directory containing scripts in container; sync with prompt.py


# --- Initialize OpenAI Client and LLM ---
try:
    openai_client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

# --- RAG Setup (Conditional & uses default LLM initially) ---
vector_store = None
retriever = None
if enable_rag:
    print("Initializing RAG...")
    vector_store = load_or_build_vector_store()
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 docs
        print("RAG retriever initialized.")
    else:
        print("WARNING: RAG vector store failed to initialize. RAG disabled.")
        enable_rag = False # Disable RAG if store fails
else:
    print("RAG is DISABLED by configuration.")
# --- End RAG Setup ---

# --- Initialize Docker Client ---
docker_client = None
try:
    docker_client = docker.from_env()
    docker_client.ping() # Test connection
    print(f"Docker client initialized. Verifying image '{CLI_DOCKER_IMAGE}'...")
    try:
        docker_client.images.get(CLI_DOCKER_IMAGE)
        print(f"Docker image '{CLI_DOCKER_IMAGE}' found.")
        os.makedirs(OSF_KUBECONFIG_DIR, exist_ok=True) # Ensure temp dir exists
    except ImageNotFound:
        print(f"CRITICAL ERROR: Docker image '{CLI_DOCKER_IMAGE}' not found.")
        sys.exit(1) # Exit the application
    except APIError as e:
        print(f"CRITICAL ERROR: Docker API error during image check: {e}. Is Docker running?")
        sys.exit(1) # Exit the application
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Docker client: {e}")
    sys.exit(1) # Exit the application


# --- Cleanup ---
# Register cleanup functions imported from helper_functions
atexit.register(cleanup_kube_configs)
atexit.register(cleanup_exec_dir)


# --- Helper Functions ---

def extract_bash(text):
    """Extracts the first Bash script block enclosed in ```bash ... ```"""
    # Find the bash block
    match = re.search(r"```bash\s*([\s\S]*?)\s*```", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def extract_yaml(text):
    """Extracts the first YAML block enclosed in ```yaml ... ```"""
    match = re.search(r"```(yaml|yml)?\s*([\s\S]*?)\s*```", text, re.MULTILINE)
    if match:
        return match.group(2).strip()
    return None

def write_temp_script(bash_content):
    """
    Writes Bash content to a temporary script file in a unique subdir under OSF_EXEC_DIR.
    Returns a dict: {'bash_path': path, 'run_subdir': path}
    Returns None if bash_content is empty or writing fails.
    """
    if not bash_content:
        print("Warning: No bash content provided to write_temp_script.")
        return None

    run_id = uuid.uuid4().hex[:12]
    run_subdir = os.path.join(OSF_EXEC_DIR, run_id)
    os.makedirs(run_subdir, exist_ok=True)
    print(f"Created temporary execution directory: {run_subdir}")

    script_filename = "run_script.sh"
    script_path = os.path.join(run_subdir, script_filename)
    try:
        with open(script_path, 'w') as f:
            f.write(bash_content)
        # Make the script executable
        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC | stat.S_IRUSR | stat.S_IWUSR)
        print(f"DEBUG: Wrote and made executable Bash script: {script_path}")
        return {'bash_path': script_path, 'run_subdir': run_subdir}
    except Exception as e:
        print(f"ERROR: Failed to write temporary Bash script {script_path}: {e}")
        # Clean up the directory if script writing failed
        try:
            shutil.rmtree(run_subdir, ignore_errors=True)
        except Exception as cleanup_err:
             print(f"Error during cleanup after failed script write: {cleanup_err}")
        return None


def cleanup_run_subdir(subdir_path):
    """Removes a specific temporary run subdirectory."""
    if subdir_path and os.path.exists(subdir_path):
        print(f"Cleaning up temporary run directory: {subdir_path}")
        try:
            shutil.rmtree(subdir_path, ignore_errors=True)
        except OSError as e:
            print(f"Error removing run temp directory {subdir_path}: {e}")


def parse_osf_action(text):
    """
    Parses the ACTION_MARKER line from the end of the text.
    Returns the action_type string or None.
    """
    print('RAW TEXT:\n', text)
    action_line = None
    lines = text.strip().split('\n')
    for line in reversed(lines):
        stripped_line = line.strip()
        if stripped_line:
            if stripped_line.startswith(ACTION_MARKER):
                action_line = stripped_line
            break

    if not action_line:
        return None

    action_content_raw = action_line[len(ACTION_MARKER):].strip()

    KNOWN_ACTIONS = [
        "display_yaml",
        "login",
        "logout",
        "submit",
        "run_script"
    ]

    if action_content_raw in KNOWN_ACTIONS:
        print(f"Action found: {action_content_raw}")
        return action_content_raw
    else:
        print(f"Warning: Unrecognized OSF Action content: '{action_content_raw}'")
        return None

def remove_osf_action_line(text):
    """Removes the ACTION_MARKER line if it's the last line."""
    lines = text.strip().split('\n')
    if lines and lines[-1].strip().startswith(ACTION_MARKER):
        return "\n".join(lines[:-1]).strip()
    return text.strip()


def get_llm_instance(provider, model_name):
    """Initializes and returns the Langchain LLM instance."""
    print(f"Attempting to initialize LLM: Provider='{provider}', Model='{model_name}'")
    try:
        if provider == 'openai':
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif provider == 'ollama':
            print(f"Initializing ChatOllama with base_url='{OLLAMA_BASE_URL}'")
            OLLAMA_CLIENT_TIMEOUT = 300
            return ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                timeout=OLLAMA_CLIENT_TIMEOUT
            )
        elif provider == 'ibm_granite':
            print("WARNING: IBM Granite provider selected but not fully implemented.")
            raise NotImplementedError("IBM Granite provider not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ImportError as e:
        print(f"ERROR: Missing dependency for provider '{provider}'. {e}")
        raise ValueError(f"Dependencies missing for provider '{provider}'. Please install.") from e
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM for Provider='{provider}', Model='{model_name}'. Error: {e}")
        raise ValueError(f"Failed to initialize LLM {provider}:{model_name}: {e}") from e

# --- END Helper Functions ---


# --- Routes ---

@app.route('/')
def api_root():
    return jsonify({"message": "OpenShift Forward - AI Companion"}), 200

@app.route(f'{API_PREFIX}/clear_session', methods=['GET', 'POST'])
def clear_user_session():
    """Clears the entire session for the current user."""
    session.clear()
    return redirect(url_for('api_root'))


# Check initial login status based on session
@app.route(f'{API_PREFIX}/check_login', methods=['GET'])
def check_login():
    if session.get('cluster_logged_in'):
        return jsonify({
            "isLoggedIn": True,
            "clusterType": session.get('cluster_type'),
            "clusterInfo": session.get('cluster_info')
        })
    else:
        return jsonify({"isLoggedIn": False})

@app.route(f'/{API_PREFIX}/available_models', methods=['GET'])
def get_available_models():
    ollama_models = []
    ollama_error = None
    try:
        # Check if Ollama server is reachable first
        response = requests.get(OLLAMA_BASE_URL, timeout=3) # Quick check
        response.raise_for_status() # Raise exception for bad status codes

        # Use the Ollama library if installed, or fallback to requests
        try:
            import ollama
            client = ollama.Client(host=OLLAMA_BASE_URL)
            models_info = client.list().get('models', [])
            ollama_models = sorted([m['model'] for m in models_info])
            print(f"Fetched {len(ollama_models)} models from Ollama.")
        except ImportError:
            print("Ollama library not installed, using direct HTTP request to /api/tags.")
            api_tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            response_tags = requests.get(api_tags_url, timeout=10)
            response_tags.raise_for_status()
            models_info = response_tags.json().get('models', [])
            ollama_models = sorted([m['name'] for m in models_info])
            print(f"Fetched {len(ollama_models)} models via HTTP.")
        except Exception as e:
            print(f"Error fetching models from Ollama API: {e}")
            ollama_error = f"Error connecting to Ollama API: {e}"

    except requests.exceptions.RequestException as e:
        print(f"Ollama server at {OLLAMA_BASE_URL} not reachable: {e}")
        ollama_error = f"Ollama server ({OLLAMA_BASE_URL}) not reachable."
    except Exception as e:
        print(f"Unexpected error checking Ollama: {e}")
        ollama_error = f"Unexpected error checking Ollama: {e}"


    # Predefined lists (customize as needed)
    openai_models = ['o4-mini', 'gpt-4.1-mini', 'gpt-4o-mini', 'gpt-3.5-turbo'] # Add more if desired
    ibm_granite_models = ['ibm/granite-13b-chat-v2'] # Placeholder

    return jsonify({
        'openai': openai_models,
        'ollama': {'models': ollama_models, 'error': ollama_error},
        'ibm_granite': ibm_granite_models # Keep placeholder structure
    })

@app.route(f'{API_PREFIX}/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('prompt')
    current_yaml_from_user = data.get('current_yaml')
    provider = data.get('provider', DEFAULT_PROVIDER)
    model_name = data.get('model') # Get specific model

    # Set default model if none provided for the selected provider
    if not model_name:
        if provider == 'openai':
            model_name = DEFAULT_OPENAI_MODEL
        # Add defaults for other providers if needed
        # elif provider == 'ollama': model_name = DEFAULT_OLLAMA_MODEL
        else:
            # Attempt to get first available if none specified (e.g., for Ollama)
            # This part can be complex, stick to explicit selection or predefined default for now
            return jsonify({"error": f"No model specified for provider '{provider}' and no default set."}), 400

    print(f"Chat request received: Provider='{provider}', Model='{model_name}'")
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # --- Initialize the selected LLM ---
    try:
        llm = get_llm_instance(provider, model_name)
    except (ValueError, NotImplementedError) as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e: # Catch other init errors
        print(f"Unhandled error during LLM initialization: {e}")
        return jsonify({"error": f"Failed to initialize LLM: {e}"}), 500

    # --- Session History (Optional with RAG, but can provide context for follow-ups) ---
    # Decide if you want to include chat history in the RAG chain input.
    # For simplicity here, we'll primarily use the current prompt for retrieval,
    # but the session history is still stored for non-RAG fallback or potential future use.
    if 'conversation' not in session:
        # Use original system prompt for conversation history storage
        session['conversation'] = [{"role": "system", "content": SYSTEM_PROMPT}]
        session.modified = True

    # --- Add User Message & Manage History ---
    # Only add the text part of the user's prompt to the persistent history
    session['conversation'].append({"role": "user", "content": user_prompt})
    max_history = 20 # Define max user/assistant pairs
    if len(session['conversation']) > max_history * 2 + 1:
        # Keep system prompt, remove oldest user/assistant pair(s) to meet limit
        # Calculate how many items to keep from the end (most recent pairs)
        items_to_keep = max_history * 2
        session['conversation'] = [session['conversation'][0]] + session['conversation'][-items_to_keep:]
        print(f"History truncated (after user msg). Length: {len(session['conversation'])}")
        session.modified = True
    # --- End User History Management ---

    combined_input = user_prompt # Start with user text
    if current_yaml_from_user and current_yaml_from_user.strip():
        combined_input += f"\n\n[User's Current YAML Editor Content]:\n```yaml\n{current_yaml_from_user}\n```"

    bot_reply_full = "Sorry, something went wrong."

    try:
        # --- Use RAG only if enabled rag_chain was successfully created ---
        if enable_rag and retriever and llm:
            print(f"Invoking RAG chain with: Provider='{provider}', Model='{model_name}'...")
            # Create the RAG chain *dynamically* using the selected LLM
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", RAG_TASK_SYSTEM_INSTRUCTION),
                ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:")
            ])
            question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            response = rag_chain.invoke({"input": combined_input})
            bot_reply_full = response.get("answer", "Sorry, I couldn't generate an answer using the available documents.").strip()
            print(f"RAG chain response received.")

        # --- Fallback to Non-RAG (if RAG failed or isn't setup) ---
        else:
            print(f"Using direct LLM call with: Provider='{provider}', Model='{model_name}'...")
            # Construct messages for direct call
            direct_messages = session['conversation'] # includes system + user text prompt
            if current_yaml_from_user and current_yaml_from_user.strip():
                direct_messages.append({"role": "user", "content": f"[User is currently viewing/editing this YAML]:\n```yaml\n{current_yaml_from_user}\n```"})
            if len(direct_messages) > 0 and direct_messages[0]["role"] != "system":
                direct_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            # Use the Langchain LLM instance directly
            ai_message = llm.invoke(direct_messages)
            bot_reply_full = ai_message.content.strip()
            print(f"Direct LLM response received.")
        # --- End RAG / LLM Call ---

        # --- Process LLM Output ---
        action_type = parse_osf_action(bot_reply_full)
        user_visible_text_raw = remove_osf_action_line(bot_reply_full)

        # --- Initialize final_reply_for_chat with the base text ---
        final_reply_for_chat = user_visible_text_raw
        yaml_for_panel = None

        # --- Check for extracted YAML ---
        extracted_yaml_content = extract_yaml(user_visible_text_raw)
        if extracted_yaml_content:
            yaml_for_panel = extracted_yaml_content
            user_is_logged_in = session.get('cluster_logged_in', False)

            # Try to remove the full YAML block (including ```) from the raw text
            # Use a regex similar to extract_yaml but capture the whole block
            yaml_block_pattern = r"```(yaml|yml)?\s*([\s\S]*?)\s*```"
            match = re.search(yaml_block_pattern, user_visible_text_raw, re.MULTILINE)
            base_chat_message = "I've placed the generated YAML in the panel." # Default when YAML found

            if match:
                full_yaml_block = match.group(0)
                text_without_yaml = user_visible_text_raw.replace(full_yaml_block, '', 1).strip()
                if text_without_yaml:
                    # Use remaining text + standard note if text exists besides YAML
                    base_chat_message = text_without_yaml + "\n\n(YAML placed in the panel.)"

            # Re-assign final_reply_for_chat only if YAML was found
            final_reply_for_chat = base_chat_message # Start with base message

            # Conditionally add submit/login instructions
            if action_type == 'display_yaml':
                if user_is_logged_in:
                    final_reply_for_chat += f" Review it before clicking 'Submit to Cluster'."
                else: # Not logged in
                    final_reply_for_chat += " Please log in to a cluster if you want to apply this change."

        # Add the assistant's textual reply (without YAML block) to persistent history
        session['conversation'].append({"role": "assistant", "content": final_reply_for_chat})

        # --- Store Action Details ---
        session['last_osf_action_type'] = action_type
        session.modified = True

        # --- Prepare JSON Response ---
        response_payload = {
            "reply": final_reply_for_chat,
            "yaml": yaml_for_panel,
            "osf_action": action_type,
        }

        print(f"--- Backend /chat Response ---")
        print(f"Reply Text: {f'{final_reply_for_chat[:100]}...' if final_reply_for_chat else 'None'}")
        print(f"Extracted YAML: {'Present' if yaml_for_panel else 'None'}")
        print(f"OSF Action: {action_type}")

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during chat processing: {e}")
        traceback.print_exc()
        error_msg = f"LLM ({provider}:{model_name}) or RAG processing error: {e}"
        return jsonify({"error": error_msg}), 500


@app.route(f'{API_PREFIX}/transcribe', methods=['POST'])
def transcribe_audio():
    if not openai_client:
        return jsonify({"error": "OpenAI client not initialized. Check API key."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    try:
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        try:
            audio_segment = AudioSegment.from_file(audio_buffer) # Try auto-detect first
        except Exception:
            try:
                audio_buffer.seek(0) # Reset buffer pointer
                audio_segment = AudioSegment.from_file(audio_buffer, format="webm") # Try webm explicitly
            except Exception as pydub_err_fmt:
                print(f"Pydub error: {pydub_err_fmt}")
                return jsonify({"error": f"Could not process audio format. Ensure ffmpeg is installed. Error: {pydub_err_fmt}"}), 400

        transcribed_text = speech_to_text(openai_client, audio_segment)
        return jsonify({"text": transcribed_text})

    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": f"Transcription failed: {e}"}), 500


@app.route(f'{API_PREFIX}/login', methods=['POST'])
def login():
    data = request.json
    cluster_type = data.get('cluster_type')
    display_info = ""
    success = False
    output = "Login failed." # Default error message
    login_details_for_session = {} # Dict to hold details before putting in session

    if cluster_type == 'openshift':
        url = data.get('url')
        username = data.get('username')
        password = data.get('password')

        if not all([url, username, password]):
            return jsonify({"success": False, "error": "Missing OpenShift URL, username, or password"}), 400

        processed_url_for_login = url.strip() # Start with the input URL

        try:
            parsed_url = urlparse(processed_url_for_login)
            hostname = parsed_url.hostname

            if hostname:
                # Use the regex to check if it looks like a console hostname
                console_hostname_match = re.match(r'(?:console-)?openshift-console\.apps[\.\-](.+)', hostname)

                print('HOSTNAME:', hostname)
                print('MATCH:', console_hostname_match)
                if console_hostname_match:
                    cluster_base_domain = console_hostname_match.group(1)
                    api_hostname = f"api.{cluster_base_domain}"
                    # Use the original scheme (defaulting to https) but hardcode port 6443
                    scheme = parsed_url.scheme if parsed_url.scheme else 'https'
                    processed_url_for_login = f"{scheme}://{api_hostname}:6443"
                    print(f"Converted console URL format in '{url}' to API URL: '{processed_url_for_login}'")
        except Exception:
            processed_url_for_login = url.strip() # Fallback

        # --- Determine Display Info (based on the URL *used for login*) ---
        display_info = processed_url_for_login
        try:
            parsed_login_url = urlparse(processed_url_for_login)
            display_hostname = parsed_login_url.hostname or parsed_login_url.path # Handle cases without schema
            display_port = parsed_login_url.port # Use port if explicitly in URL
            if display_port:
                display_info = f"{display_hostname}:{display_port}"
            elif display_hostname:
                # Display hostname:6443 if no port was specified but hostname exists
                display_info = f"{display_hostname}:6443"
        except Exception:
            print(f"Warning: Could not parse login URL '{processed_url_for_login}' for display info. Using raw URL.")

        print(f"URL used for login: {processed_url_for_login}, Username: {username}")
        # Command: Use environment variable $LOGIN_PASSWORD
        login_and_token_cmd_str = (
            f"oc login "
            f"-u {shlex.quote(username)} -p \"$LOGIN_PASSWORD\" "
            f"--insecure-skip-tls-verify=true "
            f"{shlex.quote(processed_url_for_login)} && "
            f"oc whoami --show-token"
        )

        # Pass password via environment variable to the container
        environment_vars = {"LOGIN_PASSWORD": password}

        login_token_success, result_output = run_in_container(
            docker_client=docker_client,
            command_input=login_and_token_cmd_str,
            environment_vars=environment_vars,
            timeout=45
        )

        # --- Process Execution Result ---
        extracted_token = None
        final_status_determined = False # Flag to track if we've set success/output

        if login_token_success: # Container exit code 0
            raw_output = result_output
            # Check for 401 even with exit 0
            has_401_error = "401 Unauthorized" in raw_output.lower() # Case-insensitive check

            # Try to extract the token (finds last sha256 line)
            lines = raw_output.strip().splitlines()
            for line in reversed(lines):
                stripped_line = line.strip()
                # Simplified check (as per recent discussion)
                if stripped_line.startswith("sha256~"):
                    extracted_token = stripped_line
                    break # Found potential token

            # Determine final success based on token presence and absence of 401
            if extracted_token and not has_401_error:
                # ---- SUCCESS PATH ----
                print(f"Token extracted successfully (exit 0): {extracted_token}")
                success = True
                final_status_determined = True
                # Prepare session data using extracted token and PROCESSED URL/DISPLAY INFO
                login_details_for_session = {
                    'cluster_type': 'openshift',
                    'oc_token': extracted_token, # Use the extracted token
                    'oc_server': processed_url_for_login, # <<< Store the URL used for login
                    'oc_skip_tls': True, # Because we used the flag
                    'cluster_display': display_info # <<< Store the derived display_info
                }
                output = f"Successfully logged in via container and obtained token for server {display_info}."
                # ---- END SUCCESS PATH ----
            else:
                # Exit 0, but bad output (no token found or 401 present)
                success = False
                final_status_determined = True
                print(f"Login command exited 0, but output indicates failure or no valid token found: {raw_output[:250]}...")
                if has_401_error:
                    output = "Login failed: Invalid credentials (401 Unauthorized)."
                else: # Other non-token output from exit 0 (e.g., help text, other errors)
                    error_detail = raw_output[:200] + '...' if len(raw_output) > 200 else raw_output
                    output = f"OpenShift login failed: Unexpected server response. Detail: {error_detail}"

        # Handle container non-zero exit code OR if status wasn't determined above
        if not final_status_determined:
            success = False # Explicitly set failure
            raw_output = result_output # Use the output from the failed container run

            # ---- FAILURE PATH (Non-Zero Exit or Undetermined Status) ----
            print(f"Login command failed (non-zero exit or prior issue). Output: {raw_output[:250]}...")
            # Check specifically for 401 Unauthorized in the output of the failed run
            if "401 Unauthorized" in raw_output.lower(): # Case-insensitive check
                output = "Login failed: Invalid credentials (401 Unauthorized)."
            else:
                # Generic failure message for non-zero exit without 401
                error_detail = raw_output[:200] + '...' if len(raw_output) > 200 else raw_output
                output = f"{re.sub(r'^WARNING: Using insecure TLS.*?supported! ?', '', error_detail, 1, re.DOTALL)}"

    elif cluster_type == 'kubernetes':
        context_name = data.get('context')
        if not context_name:
            return jsonify({"success": False, "error": "Missing Kubernetes context name"}), 400

        # --- Validate K8s context using client library ---
        try:
            print(f"Validating Kubernetes context '{context_name}'...")
            # Create a temporary Configuration object to load into
            # This avoids modifying the global client state
            temp_kube_config = client.Configuration()
            # Attempt to load config for the specific context
            # This implicitly uses the default kubeconfig location(s)
            config.load_kube_config(context=context_name, client_configuration=temp_kube_config)

            # Create an ApiClient using this temporary, context-specific configuration
            api_client = client.ApiClient(configuration=temp_kube_config)
            # Instantiate a client for a common API group (e.g., CoreV1)
            v1 = client.CoreV1Api(api_client=api_client)
            # Make a simple, low-impact API call to verify connectivity and auth
            print(f"Attempting 'list_namespace' API call for context '{context_name}'...")
            v1.list_namespace(limit=1, _request_timeout=10) # Short timeout for validation
            # If the above call doesn't raise ApiException, context is valid & reachable

            print(f"Context '{context_name}' validation successful.")
            success = True
            display_info = context_name
            login_details_for_session = {
                 'cluster_type': 'kubernetes',
                 'cluster_context': context_name, # Store context name
                 'cluster_display': display_info   # Store display name (same as context)
            }
            output = f"Successfully validated Kubernetes context '{context_name}'."

        except ConfigException as e:
            # Error finding/parsing kubeconfig or the specified context
            print(f"Error loading kube config for context '{context_name}': {e}")
            output = f"Error: Context '{context_name}' not found or kubeconfig is invalid."
            success = False
        except ApiException as e:
            # Error connecting to the cluster API server using the context
            print(f"Error connecting to cluster for context '{context_name}': Status={e.status}, Reason={e.reason}")
            # Provide more specific feedback if possible
            if e.status == 401: # Unauthorized
                output = f"Error: Authentication failed for context '{context_name}'. Check credentials/token validity."
            elif e.status == 403: # Forbidden
                output = f"Error: Insufficient permissions for context '{context_name}' to list namespaces."
            else:
                output = f"Error: Failed to connect to cluster using context '{context_name}'. Check cluster status and kubeconfig. (Status: {e.status})"
            success = False
        except Exception as e:
            # Catch other potential errors (e.g., network issues not raising ApiException)
            print(f"Unexpected error validating context '{context_name}': {e}")
            output = f"Unexpected error validating context '{context_name}': {e}"
            success = False
        # --- End K8s Validation ---

    else:
        return jsonify({"success": False, "error": "Invalid cluster type specified"}), 400

    # --- Update Session State ---
    if success:
        print("--- Updating Session (Success) ---")
        session['cluster_logged_in'] = True
        session['cluster_type'] = login_details_for_session['cluster_type']
        session['cluster_info'] = login_details_for_session['cluster_display']

        if login_details_for_session['cluster_type'] == 'kubernetes':
            session['cluster_context'] = login_details_for_session['cluster_context']
            # Clear any stale OpenShift keys
            session.pop('oc_token', None);
            session.pop('oc_server', None);
            session.pop('oc_skip_tls', None)
            print(f"Stored K8s context: {session['cluster_context']}")
        elif login_details_for_session['cluster_type'] == 'openshift':
            session['oc_token'] = login_details_for_session['oc_token']
            session['oc_server'] = login_details_for_session['oc_server']
            session['oc_skip_tls'] = login_details_for_session['oc_skip_tls']
            # Clear any stale Kubernetes keys
            session.pop('cluster_context', None)
            print(f"Stored OC Server: {session['oc_server']}") # Don't log token

        # Clear previous action state regardless of type
        session.pop('last_osf_action_type', None)
        session.modified = True
        print(f"Login successful. Session updated for {session['cluster_type']}.")

        return jsonify({
            "success": True,
            "cluster_display": session['cluster_info'],
            "cluster_type": session['cluster_type']
        })
    else:
        # --- Clear Session on Failure ---
        print("--- Clearing Session (Failure) ---")
        session.pop('cluster_logged_in', None)
        session.pop('cluster_type', None)
        session.pop('cluster_info', None)
        session.pop('cluster_context', None)
        session.pop('oc_token', None)
        session.pop('oc_server', None)
        session.pop('oc_skip_tls', None)
        session.pop('last_osf_action_type', None)
        session.modified = True
        print(f"Login failed. Session cleared. Reason: {output}")
        return jsonify({"success": False, "error": output})


@app.route(f'{API_PREFIX}/logout', methods=['POST'])
def logout():
    session.pop('cluster_logged_in', None)
    session.pop('cluster_type', None)
    session.pop('cluster_info', None)
    session.modified = True
    return jsonify({"success": True})


@app.route(f'{API_PREFIX}/submit', methods=['POST'])
def submit_yaml():
    if not session.get('cluster_logged_in'):
        return jsonify({"success": False, "error": "Not logged into a cluster."}), 403

    data = request.json
    yaml_content = data.get('yaml')
    provider = data.get('provider', DEFAULT_PROVIDER)
    model_name = data.get('model')
    if not model_name:
        if provider == 'openai':
            model_name = DEFAULT_OPENAI_MODEL
        else:
            return jsonify({"success": False, "error": f"Internal Error: No model specified for script generation ({provider})."}), 500

    if not yaml_content:
        return jsonify({"success": False, "error": "YAML content is empty."}), 400

    # --- Step 1: Make internal LLM call to get the Bash script ---
    generated_bash_script = None
    try:
        print(f"Making internal LLM call to generate script for submission using {provider}:{model_name}...")
        llm_for_script = get_llm_instance(provider, model_name)

        auth_args = ''
        cluster_type = session.get('cluster_type')
        tool = 'kubectl'

        if cluster_type == 'kubernetes':
            # Get Kubernetes context from session
            context = session.get('cluster_context')
            if not context:
                 return jsonify({"success": False, "error": "Kubernetes context missing from session."}), 401
        else:
            tool = 'oc'
            # Get OpenShift details from session
            oc_token = session.get('oc_token')
            oc_server = session.get('oc_server')
            oc_skip_tls = session.get('oc_skip_tls', False)
            if not oc_token or not oc_server:
                return jsonify({"success": False, "error": "OpenShift token and/or server missing."}), 401

            auth_flags = []
            # Use shlex.quote just in case server URL or token has special chars, although unlikely for tokens
            auth_flags.append(f"--token={shlex.quote(oc_token)}")
            auth_flags.append(f"--server={shlex.quote(oc_server)}")
            if oc_skip_tls:
                auth_flags.append("--insecure-skip-tls-verify=true")
            auth_args_str = " ".join(auth_flags) # Create space-separated string

        # Format the UPDATED prompt
        script_generation_prompt = SUBMIT_SCRIPT_PROMPT.format(
            yaml_content=yaml_content,
            auth_args=auth_args_str,
            oc_cmd=tool
        )

        script_ai_message = llm_for_script.invoke(script_generation_prompt)
        raw_llm_response = script_ai_message.content.strip()
        print("Internal LLM call finished.")

        generated_bash_script = extract_bash(raw_llm_response)
        print(f'BASH SCRIPT:\n{generated_bash_script}')

        if not generated_bash_script:
            print("ERROR: Internal LLM call did not generate a Bash script block.")
            print(f"LLM Response (preview):\n{raw_llm_response[:500]}...")
            return jsonify({"success": False, "error": "Failed to generate execution script. LLM response did not contain a ```bash block."}), 500

        print("Bash script extracted from internal LLM response.")

    except Exception as e:
        print(f"Error during internal LLM call for script generation: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to generate execution script: {e}"}), 500


    # --- Step 2: Write Temporary Script File ---
    temp_script_info = None
    try:
        # Use the simplified function, passing only the script content
        temp_script_info = write_temp_script(generated_bash_script)

        if not temp_script_info or not temp_script_info.get('bash_path'):
            print("ERROR: Failed to write temporary script file for execution.")
            return jsonify({"success": False, "error": "Failed to write temporary script for execution."}), 500

    except Exception as e:
        print(f"Error during temporary script writing for submission: {e}")
        traceback.print_exc()
        # Attempt cleanup even if info object wasn't fully formed
        if temp_script_info and temp_script_info.get('run_subdir'):
            cleanup_run_subdir(temp_script_info['run_subdir'])
        elif 'run_subdir' in locals() and run_subdir: # If subdir was created but function failed later
             cleanup_run_subdir(run_subdir)
        return jsonify({"success": False, "error": f"Failed to write temporary script file: {e}"}), 500


    # --- Step 3: Execute the Generated Bash Script in Container ---
    script_path_host = temp_script_info['bash_path']
    run_subdir_host = temp_script_info['run_subdir']
    print('SCRIPT:', script_path_host, run_subdir_host)
    script_filename = os.path.basename(script_path_host)
    script_path_container = os.path.join(CONTAINER_EXEC_DIR, script_filename)

    print(f"Executing generated script '{script_filename}' ({script_path_host}) in container...")

    success, script_output = False, "Script execution did not return output." # Initialize defaults
    try:
        success, script_output = run_in_container(
            docker_client=docker_client,
            command_input=script_path_container,
            session_auth=session.get('cluster_type') and { # Pass auth details for run_in_container to handle kubeconfig/env vars
                 'cluster_type': session.get('cluster_type'),
                 'oc_token': session.get('oc_token'),
                 'oc_server': session.get('oc_server'),
                 'oc_skip_tls': session.get('oc_skip_tls', False),
                 'cluster_context': session.get('cluster_context')
            },
            environment_vars=None,
            # Mount the directory containing the script
            temp_dir_mapping={run_subdir_host: CONTAINER_EXEC_DIR}
        )

        print(f"Script execution finished: Success={success}, Output (preview): {script_output[:200]}...")

    except Exception as e:
        print(f"Error calling run_in_container for script execution: {e}")
        traceback.print_exc()
        success = False
        script_output = f"Error calling execution container: {e}"


    # --- Step 4: Report Results and Cleanup ---
    if temp_script_info and temp_script_info.get('run_subdir'):
        cleanup_run_subdir(temp_script_info['run_subdir'])

    if success:
        return jsonify({"success": True, "output": script_output, "tool_used": tool})
    else:
        return jsonify({"success": False, "error": script_output, "tool_used": tool})


# --- Main Execution ---
if __name__ == '__main__':
    if not openai_client:
        print("WARNING: OpenAI client failed to initialize. LLM features will not work.")
    if enable_rag and not vector_store:
        print("WARNING: RAG vector store failed to initialize. Chatbot will not use retrieved context.")
    # Production mode only if FLASK_DEBUG=0
    enable_debug = False if os.environ.get('FLASK_DEBUG', '') == '0' else True
    app.run(debug=enable_debug, host='0.0.0.0', port=5001)
