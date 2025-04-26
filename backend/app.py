import io
import os
import re
import requests # Needed for checking Ollama server reachability
import shlex
import sys
import yaml # Still needed for YAML parsing in /submit route

from urllib.parse import urlparse, urlunparse # Needed in /login route

import docker
from docker.errors import APIError, NotFound, ImageNotFound

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS

# --- Kubernetes client imports ---
from kubernetes import config
from kubernetes import client

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
from prompt import SYSTEM_PROMPT, RAG_TASK_SYSTEM_INSTRUCTION
from helper_functions import run_in_container
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
CLI_DOCKER_IMAGE = 'oc-image:latest'
TEMP_KUBECONFIG_DIR = "/tmp/temp_kube_configs"


# --- Initialize OpenAI Client and LLM ---
try:
    openai_client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

# --- RAG Setup (Conditional & uses default LLM initially) ---
# Note: RAG chain creation now happens *inside* /chat based on selected LLM
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
         os.makedirs(TEMP_KUBECONFIG_DIR, exist_ok=True) # Ensure temp dir exists
    except ImageNotFound:
        print(f"CRITICAL ERROR: Docker image '{CLI_DOCKER_IMAGE}' not found.")
        sys.exit(1) # Exit the application
    except APIError as e:
        print(f"CRITICAL ERROR: Docker API error during image check: {e}. Is Docker running?")
        sys.exit(1) # Exit the application
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize Docker client: {e}")
    sys.exit(1) # Exit the application


# --- Helper Functions ---

def extract_yaml(text):
    """Extracts the first YAML block enclosed in ```yaml ... ```"""
    # Allow optional language specifier like ```yaml or ```yml
    match = re.search(r"```(yaml|yml)?\s*([\s\S]*?)\s*```", text, re.MULTILINE)
    if match:
        # Return the content inside the backticks
        return match.group(2).strip()
    # Check if the text *itself* looks like YAML (starts with apiVersion/kind)
    # Be careful not to grab the OSF_ACTION line if it's the only thing left
    lines = text.strip().split('\n')
    # Filter out the action line before checking
    non_action_lines = [line for line in lines if not line.strip().startswith("*OSF_ACTION:")]
    if non_action_lines and (non_action_lines[0].startswith("apiVersion:") or non_action_lines[0].startswith("kind:")):
         return "\n".join(non_action_lines).strip() # Return the likely YAML part
    return None

def parse_osf_action(text):
    """
    Parses the ~OSF_ACTION: line from the end of the text.
    Strips potential trailing non-alphanumeric chars from keywords.
    Returns a tuple: (action_type, action_data)
    Example return values:
      ('oc_apply', None)
      ('cmd', 'oc get pods; kubectl get nodes')
      ('submit', None)
      (None, None) if no valid action line is found.
    """
    action_line = None
    # Find the last non-empty line
    print('RAW TEXT:\n', text)
    lines = text.strip().split('\n')
    for line in reversed(lines):
        stripped_line = line.strip()
        if stripped_line:
            if stripped_line.startswith(ACTION_MARKER):
                action_line = stripped_line
            break # Found the last non-empty line, stop searching

    if not action_line:
        print(f"No '{ACTION_MARKER}' line found at the end.")
        return None, None

    action_content_raw = action_line[len(ACTION_MARKER):].strip()

    # Handle potential command first, as it has '='
    if action_content_raw.startswith("cmd="):
        command_string = action_content_raw[len("cmd="):].strip()
        return ("cmd", command_string) if command_string else (None, None)

    # For simple keywords, strip common trailing non-alphanumeric chars like '*'
    action_keyword_cleaned = action_content_raw.rstrip('*,.;:!~ ') # Added ~
    if action_keyword_cleaned != action_content_raw:
        print(f"Cleaned action keyword: {repr(action_keyword_cleaned)}")

    KNOWN_SIMPLE_ACTIONS = [
        "apply_yaml", "login", "logout", "submit"
    ]

    if action_keyword_cleaned in KNOWN_SIMPLE_ACTIONS:
         print(f"Match found! Returning: ({action_keyword_cleaned}, None)")
         return action_keyword_cleaned, None
    else:
        print(f"Warning: Could not parse OSF Action content: '{action_content_raw}' (Cleaned: '{action_keyword_cleaned}')")
        return None, None

def remove_osf_action_line(text):
    """Removes the ~OSF_ACTION: line if it's the last line."""
    lines = text.strip().split('\n')
    if lines and lines[-1].strip().startswith(ACTION_MARKER):
        return "\n".join(lines[:-1]).strip()
    return text.strip()

def get_llm_instance(provider, model_name):
    """Initializes and returns the Langchain LLM instance."""
    print(f"Attempting to initialize LLM: Provider='{provider}', Model='{model_name}'")
    try:
        if provider == 'openai':
            # Use default API key from environment
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif provider == 'ollama':
            print(f"Initializing ChatOllama with base_url='{OLLAMA_BASE_URL}'") # Add debug
            return ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                timout=300
            )
        elif provider == 'ibm_granite':
             # Placeholder - requires langchain-ibm and credentials
             print("WARNING: IBM Granite provider selected but not fully implemented.")
             # Example initialization (replace with actual class and params)
             # if not IBM_API_KEY: raise ValueError("IBM_API_KEY not configured")
             # return ChatWatsonx(
             #     model_id=model_name,
             #     credentials={"apikey": IBM_API_KEY},
             #     project_id=os.environ.get("IBM_PROJECT_ID"), # Often needed
             #     url=IBM_ENDPOINT
             # )
             raise NotImplementedError("IBM Granite provider not implemented yet.")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    except ImportError as e:
         print(f"ERROR: Missing dependency for provider '{provider}'. {e}")
         raise ValueError(f"Dependencies missing for provider '{provider}'. Please install.") from e
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM for Provider='{provider}', Model='{model_name}'. Error: {e}")
        # Optionally raise the error or return None / fallback
        raise ValueError(f"Failed to initialize LLM {provider}:{model_name}") from e


# --- Routes ---

@app.route('/')
def api_root():
    return jsonify({"message": "OpenShift Forward - AI Companion"}), 200

@app.route(f'{API_PREFIX}/clear_session', methods=['GET', 'POST'])
def clear_user_session():
    """Clears the entire session for the current user."""
    session.clear()
    return redirect(url_for('api_root'))
    #return jsonify({"success": True, "message": "Session cleared."})


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


        # --- Process Reply ---
        action_type, action_data = parse_osf_action(bot_reply_full)
        user_visible_text_raw = remove_osf_action_line(bot_reply_full)

        # --- Initialize final_reply_for_chat with the base text ---
        final_reply_for_chat = user_visible_text_raw # Default to full text (minus action)
        yaml_for_panel = None
        command_action = None

        if action_type == "apply_yaml":
            command_action = "apply"

        # --- Check for extracted YAML ---
        extracted_yaml_content = extract_yaml(user_visible_text_raw)
        if extracted_yaml_content:
            yaml_for_panel = extracted_yaml_content
            user_is_logged_in = session.get('cluster_logged_in', False)

            # Try to remove the full YAML block (including ```) from the raw text
            # Use a regex similar to extract_yaml but capture the whole block
            yaml_block_pattern = r"```(yaml|yml)?\s*([\s\S]*?)\s*```"
            match = re.search(yaml_block_pattern, user_visible_text_raw, re.MULTILINE)
            base_chat_message = "Okay, I've placed the generated YAML in the panel." # Default when YAML found

            if match:
                full_yaml_block = match.group(0)
                text_without_yaml = user_visible_text_raw.replace(full_yaml_block, '', 1).strip()
                if text_without_yaml:
                    # Use remaining text + standard note if text exists besides YAML
                    base_chat_message = text_without_yaml + "\n\n(YAML placed in the panel.)"

            # Re-assign final_reply_for_chat only if YAML was found
            final_reply_for_chat = base_chat_message # Start with base message

            # Conditionally add submit/login instructions
            if command_action == 'apply':
                if user_is_logged_in:
                    final_reply_for_chat += f" Review it and use 'Submit to Cluster' ({command_action})."
                else: # Not logged in
                    final_reply_for_chat += " Please log in to a cluster if you want to apply this YAML."

        # Add the assistant's textual reply (without YAML block) to persistent history
        session['conversation'].append({"role": "assistant", "content": final_reply_for_chat})

        # --- Store Action Details ---
        session['last_osf_action_type'] = action_type
        session['last_osf_action_data'] = action_data
        session.modified = True

        # --- Prepare JSON Response ---
        response_payload = {
            "reply": final_reply_for_chat, # Use the final determined value
            "yaml": yaml_for_panel,
            "osf_action": {
                "type": action_type, # e.g., "oc_apply", "cmd", "login", None
                "data": action_data # e.g., None, "oc get pods", None
            },
            "command_action": command_action
        }

        print(f"--- Backend /chat Response ---")
        print(f"Reply Text: {final_reply_for_chat[:100]}...")
        print(f"Extracted YAML: {'Present' if yaml_for_panel else 'None'}")
        print(f"OSF Action: Type='{action_type}', Data='{action_data}'")

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during chat processing: {e}")
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

        print(f"--- OpenShift Login Attempt (via Container) ---")
        print(f"URL: {url}, Username: {username}")

        # Command: Use environment variable $LOGIN_PASSWORD
        login_and_token_cmd_str = (
            f"oc login -u {shlex.quote(username)} -p \"$LOGIN_PASSWORD\" "
            f"--insecure-skip-tls-verify=true {shlex.quote(url)} && "
            f"oc whoami --show-token"
        )

        # Pass password via environment variable to the container
        # WARNING: Review security implications.
        environment_vars = {"LOGIN_PASSWORD": password}

        login_token_success, result_output = run_in_container(
            docker_client=docker_client,
            command_input=login_and_token_cmd_str,
            command_is_string=True, # This tells run_in_container it's a string
            session_auth=None,
            environment_vars=environment_vars,
            timeout=45
        )
        # --- End Container Execution ---

        # --- VALIDATE the result AND EXTRACT TOKEN ---
        # 'result_output' here contains the combined logs from run_in_container
        if login_token_success: # Check if container exited with 0
            raw_output = result_output
            extracted_token = None # Variable to hold the actual token string

            # Define success/error markers
            login_succeeded_msg = "Login successful."
            # Consider potential variations or absence of this message
            has_success_msg = login_succeeded_msg in raw_output
            has_errors = "error:" in raw_output.lower() or "fail" in raw_output.lower() or "Usage:" in raw_output.lower()

            # Try to extract the token regardless of success message, but prioritize if success msg exists
            lines = raw_output.strip().splitlines()
            for line in reversed(lines):
                stripped_line = line.strip()
                if stripped_line.startswith("sha256~"):
                     extracted_token = stripped_line
                     break # Found potential token

            # Determine final success based on presence of token and absence of errors
            if extracted_token and not has_errors:
                print(f"Token extracted successfully: {extracted_token}")
                success = True
                # --- Prepare session data using EXTRACTED token ---
                try:
                    # ... (URL parsing logic remains the same) ...
                     parsed_url = urlparse(url)
                     scheme = parsed_url.scheme if parsed_url.scheme else "https"
                     netloc = parsed_url.netloc; path = parsed_url.path
                     if not netloc:
                        netloc = path; path = ""
                     port_num = parsed_url.port; port_str = f":{port_num}" if port_num else ""
                     hostname = parsed_url.hostname if parsed_url.hostname else netloc.split(':')[0]
                     processed_url = urlunparse((scheme, hostname + port_str, path, '', '', ''))
                     display_port_str = port_str if port_str else ":6443"
                     display_info = f"{hostname}{display_port_str}"

                except Exception as e:
                    # ... (fallback for URL parsing error) ...
                    print(f"Warning: Could not parse provided URL '{url}': {e}. Using raw URL.")
                    processed_url = url; display_info = url

                login_details_for_session = {
                    'cluster_type': 'openshift',
                    'oc_token': extracted_token, # <<< USE THE EXTRACTED TOKEN HERE
                    'oc_server': processed_url,
                    'oc_skip_tls': True, # Because we used the flag
                    'cluster_display': display_info
                }
                output = f"Successfully logged in via container and obtained token for server {display_info}."
                # --- End session data preparation ---

            else:
                # Scenario: Exit code 0, but no token found or errors present
                print(f"Login command exited 0, but token extraction failed or errors found in output: {raw_output[:250]}...")
                success = False
                error_detail = raw_output if raw_output else "No output received."
                if len(error_detail) > 200:
                    error_detail = error_detail[:200] + "..."
                # Distinguish error message slightly
                if not extracted_token:
                    output = f"OpenShift login failed: Could not extract token from server response. Detail: {error_detail}"
                else: # has_errors must be true
                    output = f"OpenShift login failed: Errors detected in server response. Detail: {error_detail}"

        # --- Handle container non-zero exit code ---
        else: # login_token_success is False
            output = f"OpenShift login/token fetch command failed in container. Output: {result_output}"
            success = False # Ensure success is false

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
        session.pop('last_osf_action_data', None)
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
        session.pop('last_osf_action_data', None)
        session.modified = True
        print(f"Login failed. Session cleared.")
        return jsonify({"success": False, "error": output})


@app.route(f'{API_PREFIX}/logout', methods=['POST'])
def logout():
    session.pop('cluster_logged_in', None)
    session.pop('cluster_type', None)
    session.pop('cluster_info', None)
    session.pop('last_command_tool', None)
    session.pop('last_command_action', None)
    session.modified = True
    return jsonify({"success": True})


@app.route(f'{API_PREFIX}/submit', methods=['POST'])
def submit_yaml():
    if not session.get('cluster_logged_in'):
        return jsonify({"success": False, "error": "Not logged into a cluster."}), 403

    data = request.json
    yaml_content = data.get('yaml')

    if not yaml_content:
        return jsonify({"success": False, "error": "YAML content is empty."}), 400

    final_action_verb = "apply" # Default to apply
    try:
        # ... (YAML parsing logic using yaml.safe_load_all remains the same) ...
        yaml_docs = list(yaml.safe_load_all(yaml_content))
        if not yaml_docs:
             return jsonify({"success": False, "error": "YAML content is empty or invalid."}), 400

        first_doc = yaml_docs[0]
        if isinstance(first_doc, dict):
             kind = first_doc.get('kind')
             CREATE_ONLY_KINDS = ["Namespace", "Project", "PersistentVolume"]
             if kind in CREATE_ONLY_KINDS:
                final_action_verb = "create"

    except yaml.YAMLError as e:
        # ... (YAML error handling remains the same) ...
        return jsonify({"success": False, "error": f"Invalid YAML format: {e}"}), 400
    except Exception as e:
        # ... (Other error handling remains the same) ...
        return jsonify({"success": False, "error": f"Error processing YAML: {e}"}), 500

    # --- Prepare Command and Auth Info ---
    cluster_type = session.get('cluster_type')
    tool = None
    command = []
    auth_info_for_container = {'cluster_type': cluster_type} # Base auth info

    if cluster_type == 'openshift':
        tool = 'oc'
        # Get OpenShift details from session
        auth_info_for_container['oc_token'] = session.get('oc_token')
        auth_info_for_container['oc_server'] = session.get('oc_server')
        auth_info_for_container['oc_skip_tls'] = session.get('oc_skip_tls', False)
        print('oc_token:', auth_info_for_container['oc_token'])
        print('oc_server:', auth_info_for_container['oc_server'])
        print('oc_skip_tls:', auth_info_for_container['oc_skip_tls'])
        if not auth_info_for_container['oc_token'] or not auth_info_for_container['oc_server']:
            return jsonify({"success": False, "error": "OpenShift login details missing from session."}), 401
        command = [tool, final_action_verb, "-f", "-"] # Command to run inside container
    elif cluster_type == 'kubernetes':
        tool = 'kubectl'
        # Get Kubernetes context from session
        auth_info_for_container['cluster_context'] = session.get('cluster_context')
        if not auth_info_for_container['cluster_context']:
             return jsonify({"success": False, "error": "Kubernetes context missing from session."}), 401
        # Add context flag to the command itself (container uses mounted config)
        # run_in_container's logic now handles creating the temp config
        command = [tool, final_action_verb, "-f", "-"] # Command to run inside container
        # No need to add --context here, as the mounted config inside container will use it
    else:
         return jsonify({"success": False, "error": f"Unknown cluster type in session: {cluster_type}"}), 500

    # --- Execute in Container ---
    print(f"Attempting to run '{tool} {final_action_verb}' in container for {cluster_type}...")
    # Pass the command, the YAML data, and the specific auth details
    success, output = run_in_container(docker_client, command, yaml_content, auth_info_for_container)

    # --- Clear Session State & Return Result ---
    # Only clear action type/data, not login info
    session.pop('last_osf_action_type', None)
    session.pop('last_osf_action_data', None)
    session.modified = True

    if success:
        print(f"{tool} {final_action_verb} successful (via container).")
        return jsonify({"success": True, "output": output, "tool_used": tool})
    else:
        print(f"{tool} {final_action_verb} failed (via container): {output}")
        return jsonify({"success": False, "error": output, "tool_used": tool})


# --- Main Execution ---
if __name__ == '__main__':
    if not openai_client:
        print("WARNING: OpenAI client failed to initialize. LLM features will not work.")
    if enable_rag and not vector_store:
        print("WARNING: RAG vector store failed to initialize. Chatbot will not use retrieved context.")
    # Production mode only if FLASK_DEBUG=0
    enable_debug = False if os.environ.get('FLASK_DEBUG', '') == '0' else True
    app.run(debug=enable_debug, host='0.0.0.0', port=5001)
