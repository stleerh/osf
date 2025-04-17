import atexit
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import yaml
from urllib.parse import urlparse, urlunparse

import docker
from docker.errors import APIError, NotFound, ImageNotFound

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS # Import CORS
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# --- RAG Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# --- End RAG Imports ---

# --- Local Imports ---
from audio import speech_to_text
from rag_setup import load_or_build_vector_store
from prompt import SYSTEM_PROMPT, RAG_TASK_SYSTEM_INSTRUCTION
# --- End Local Imports ---


load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'mysecret')
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://127.0.0.1:5173"]) # Adjust port if needed

#llm_model = 'gpt-3.5-turbo'
llm_model = 'gpt-4.1-mini'
#llm_model = 'o3-mini'
temperature = 0
enable_rag = True

ACTION_MARKER = "~OSF_ACTION:" # Define the marker
CLI_DOCKER_IMAGE = 'oc-image:latest'
TEMP_KUBECONFIG_DIR = "/tmp/temp_kube_configs"


# --- Initialize OpenAI Client and LLM ---
try:
    openai_client = OpenAI()
    if llm_model != 'o3-mini':
        print(f"Initializing ChatOpenAI with model='{llm_model}', temperature={temperature}")
        llm = ChatOpenAI(model=llm_model, temperature=temperature)
    else:
        print(f"Initializing ChatOpenAI with model='{llm_model}' (temperature omitted)")
        llm = ChatOpenAI(model=llm_model)
except Exception as e:
    print(f"Error initializing OpenAI client or Langchain Chat Model: {e}")
    openai_client = None
    llm = None

 # --- RAG Setup (Conditional) ---
vector_store = None
retriever = None
rag_chain = None

if enable_rag:
    print("Initializing RAG...")
    vector_store = load_or_build_vector_store()
    if vector_store and llm:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 docs
        print("RAG retriever initialized.")

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_TASK_SYSTEM_INSTRUCTION),
            ("human", "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:")
        ])

        # Chain to combine documents into the prompt context
        question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
        # Chain that retrieves documents, then passes them to the question_answer_chain
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Langchain RAG chain created.")

    else:
        print("WARNING: RAG vector store failed to initialize. Falling back to non-RAG mode.")
        retriever = None
        rag_chain = None
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
        # Exit if the required image is missing - fundamental requirement
        print(f"CRITICAL ERROR: Docker image '{CLI_DOCKER_IMAGE}' not found.")
        print("Please build or pull the image. Exiting.")
        sys.exit(1) # Exit the application
    except APIError as e:
        print(f"CRITICAL ERROR: Docker API error during image check: {e}. Is Docker running?")
        sys.exit(1) # Exit the application
except Exception as e:
    # Exit if Docker client cannot be initialized at all
    print(f"CRITICAL ERROR: Failed to initialize Docker client: {e}")
    print("Ensure Docker is installed, running, and permissions are correct. Exiting.")
    sys.exit(1) # Exit the application


# --- Cleanup Temporary Kubeconfigs ---
def cleanup_temp_configs():
     if os.path.exists(TEMP_KUBECONFIG_DIR):
          print(f"Cleaning up temporary kubeconfig directory: {TEMP_KUBECONFIG_DIR}")
          try:
              shutil.rmtree(TEMP_KUBECONFIG_DIR)
          except OSError as e:
              print(f"Error removing temp directory {TEMP_KUBECONFIG_DIR}: {e}")

# Run cleanup when the application exits
atexit.register(cleanup_temp_configs)


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
    lines = text.strip().split('\n')
    for line in reversed(lines):
        stripped_line = line.strip()
        if stripped_line:
            if stripped_line.startswith(ACTION_MARKER):
                action_line = stripped_line
            break # Found the last non-empty line, stop searching

    if not action_line:
        print("No '*OSF_ACTION:' line found at the end.")
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


def run_in_container(command_input, input_data=None, session_auth=None, timeout=60, environment_vars=None, command_is_string=False):
    """
    Runs a command inside a Docker container using user-specific auth from session
    OR provided environment variables (e.g., for login). Handles input data.
    Assumes container ENTRYPOINT is ["/bin/bash", "-c"].

    Args:
        command_input (list or str): Command to execute. List for actions like apply/create,
                                     string for pre-constructed commands like OC login.
        input_data (str, optional): Data to be piped as input (e.g., YAML). Mounts as file.
        session_auth (dict, optional): User's auth info from Flask session.
        timeout (int, optional): Timeout in seconds for container execution. Defaults to 60.
        environment_vars (dict, optional): Environment variables to set inside the container.
                                           Used primarily for OpenShift password login.
        command_is_string (bool, optional): Deprecated/Ignored. Logic now relies on type of command_input.

    Returns:
        tuple: (success: bool, output: str)
               Output contains combined stdout/stderr on success,
               or prioritized stderr/error message on failure.
    """
    # --- Input Validation ---
    if not command_input:
         return False, "No command provided to run_in_container."
    if not isinstance(command_input, (list, str)):
         return False, "command_input must be a list or a string."
    if environment_vars and not isinstance(environment_vars, dict):
         return False, "Invalid environment_vars provided (must be dict)."
    if session_auth and not isinstance(session_auth, dict):
         return False, "Invalid session_auth provided (must be dict)."

    # Check authentication source if session_auth is required by the flow
    # Note: /login call provides environment_vars but no session_auth
    # Note: /submit call provides session_auth but no environment_vars

    # --- Variable Initialization ---
    container_name = f"osf-cli-{uuid.uuid4().hex[:12]}"
    volumes = {}
    environment = environment_vars if environment_vars is not None else {}
    container_kubeconfig_path = "/root/.kube/config" # Standard path in container
    temp_kubeconfig_filename = None # Track kubeconfig file for deletion
    temp_input_filename = None      # Track input file for deletion
    final_command_string = None     # The single string command for bash -c
    current_command_list = None     # Holds the command if input was a list

    try:
        # --- Determine Initial Command Structure ---
        # Define current_command_list or final_command_string based on input type
        if isinstance(command_input, list):
            current_command_list = list(command_input) # Work with a copy
            # final_command_string remains None for now, will be built later
        elif isinstance(command_input, str):
             # Input is a string (e.g., from /login), use it directly
             final_command_string = command_input
             # current_command_list remains None
        # No else needed due to initial validation

        # --- Prepare Input Data File Mount (if input_data provided) ---
        container_input_path = None
        if input_data:
             temp_input_filename = os.path.join(TEMP_KUBECONFIG_DIR, f"input_{uuid.uuid4().hex}.yaml")
             try:
                 with open(temp_input_filename, 'w') as f: f.write(input_data)
             except Exception as e:
                 return False, f"Failed to write temporary input file: {e}"

             container_input_path = "/tmp/input.yaml" # Standard path inside container
             volumes[temp_input_filename] = {'bind': container_input_path, 'mode': 'ro'}

             # Modify the command LIST or STRING to use the file path
             if current_command_list: # Modify the list if input was a list
                 try:
                     f_index = -1
                     for i, arg in enumerate(current_command_list):
                          if (arg == "-f" or arg == "--filename") and i + 1 < len(current_command_list) and current_command_list[i+1] == '-':
                               f_index = i
                               break
                     if f_index != -1:
                         current_command_list[f_index + 1] = container_input_path # Modify the list
                         print(f"Modified command list to use input file: {current_command_list}")
                     else:
                         print(f"Warning: Command list {current_command_list} does not use '-f -', cannot auto-inject input file path.")
                 except Exception as e:
                      return False, f"Failed to modify command list for input file: {e}"

             elif final_command_string: # Modify the string if input was a string
                 if " -f - " in final_command_string:
                      # Use shlex.quote for the path within the string replacement
                      final_command_string = final_command_string.replace(" -f - ", f" -f {shlex.quote(container_input_path)} ", 1)
                      print(f"Modified command string to use mounted input file.")
                 else:
                      print(f"Warning: Could not find ' -f - ' in command string for input data: {final_command_string}")


        # --- Configure Auth based on session_auth (if provided - typically for actions like /submit) ---
        if session_auth:
            cluster_type = session_auth.get('cluster_type')
            if cluster_type == 'kubernetes':
                # Handle Kubernetes: Mount temporary kubeconfig
                context = session_auth.get('cluster_context')
                if not context:
                    return False, "Kubernetes context missing from session auth."
                try:
                    temp_kubeconfig_filename = os.path.join(TEMP_KUBECONFIG_DIR, f"kubeconfig_{uuid.uuid4().hex}")
                    # --- TODO: Implement actual logic to build temp Kubeconfig ---
                    # This needs to load the host config, find the specified context,
                    # extract cluster+user details, and write a minimal config file.
                    # Placeholder content:
                    temp_kubeconfig_dict = {'current-context': context, 'contexts': [], 'clusters': [], 'users': []}
                    print("WARNING: Using placeholder Kubeconfig content. Implement real extraction.")
                    # --- End TODO ---
                    with open(temp_kubeconfig_filename, 'w') as f: yaml.dump(temp_kubeconfig_dict, f)
                    volumes[temp_kubeconfig_filename] = {'bind': container_kubeconfig_path, 'mode': 'ro'}
                    print(f"Prepared temporary kubeconfig for context '{context}' at {temp_kubeconfig_filename}")
                    # The command list (current_command_list) itself doesn't need modification here
                except Exception as e:
                    print(f"Error preparing temporary kubeconfig: {e}")
                    traceback.print_exc() # Print stack trace for debugging
                    return False, f"Error preparing kubeconfig: {e}"

            elif session_auth['cluster_type'] == 'openshift':
                # Handle OpenShift Actions: Inject auth flags into the command list
                if not current_command_list:
                     # This path requires the command to have been passed as a list (e.g., from /submit)
                     return False, "Internal error: Expected command list for OpenShift action, but received string or invalid input."

                token = session_auth.get('oc_token')
                server = session_auth.get('oc_server')
                skip_tls = session_auth.get('oc_skip_tls', False)
                if not token or not server:
                    return False, "OpenShift token/server missing from session for action."

                action_cmd_list = list(current_command_list) # Work with a copy

                # Prepare flags to insert (insert after 'oc' at index 1)
                auth_flags_to_insert = []
                auth_flags_to_insert.append(f"--server={server}")
                auth_flags_to_insert.append(f"--token={token}")
                if skip_tls:
                    auth_flags_to_insert.append("--insecure-skip-tls-verify=true")

                # Insert flags after 'oc'
                if len(action_cmd_list) > 0 and action_cmd_list[0] == 'oc':
                    action_cmd_list[1:1] = auth_flags_to_insert # Insert flags at index 1
                    # Construct the final command string ONLY from this modified action list
                    final_command_string = " ".join(shlex.quote(arg) for arg in action_cmd_list)
                else:
                    return False, f"Invalid OpenShift command list format (expected 'oc' first): {action_cmd_list}"


        # --- Final Command String Construction (if not already set) ---
        # This covers cases like K8s commands, or commands without session_auth
        if final_command_string is None:
            if current_command_list:
                # Join the list (potentially modified by input file path) into the string
                final_command_string = " ".join(shlex.quote(arg) for arg in current_command_list)
            else:
                # Should be unreachable if initial validation and logic are correct
                return False, "Internal error: final command string could not be determined."

        # --- Final Check: Ensure final_command_string is set ---
        if not final_command_string:
             return False, "Internal error: final command string is empty or not set before execution."

        # --- Run the Container ---
        executable_type = "bash" if isinstance(command_input, str) else (current_command_list[0] if current_command_list else "unknown")
        print(f"run_in_container executing: Type='{executable_type}'") # Log type being run
        print(f"Executing final command string via bash -c: '{final_command_string}'") # Log the exact string
        if volumes:
            print(f"Volumes mapped: {list(volumes.keys())}")
        if environment:
            print(f"Environment Variables set: {list(environment.keys())}")

        start_time = time.time()
        container = None
        combined_logs = ""
        stdout_log = ""
        stderr_log = ""
        exit_code = -1

        try:
            # Run container and wait for completion
            # Pass the final_command_string INSIDE A SINGLE-ELEMENT LIST
            # to work correctly with ENTRYPOINT ["/bin/bash", "-c"]
            container = docker_client.containers.run(
                image=CLI_DOCKER_IMAGE,
                command=[final_command_string], # *** WRAP FINAL STRING IN LIST ***
                volumes=volumes,
                environment=environment,
                remove=False,      # Keep container after exit to retrieve logs
                detach=True,       # Start detached
                name=container_name,
                stdout=True,
                stderr=True
            )

            # Wait for container completion FIRST
            result = container.wait(timeout=timeout)
            exit_code = result.get('StatusCode', -1)

            # Get COMBINED logs first as the primary source
            try:
                combined_logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore').strip()
            except Exception as log_err:
                print(f"Error getting combined logs: {log_err}")
                combined_logs = f"Error retrieving logs: {log_err}" # Store error as output

            # Attempt to get separate streams as fallback/debug info
            try:
                stdout_log = container.logs(stdout=True, stderr=False).decode('utf-8', errors='ignore').strip()
            except Exception: pass # Ignore error, combined_logs is primary

            try:
                stderr_log = container.logs(stdout=False, stderr=True).decode('utf-8', errors='ignore').strip()
            except Exception: pass # Ignore error, combined_logs is primary

            # Determine the output to return based on exit code
            output_for_processing = combined_logs if exit_code == 0 else (stderr_log or combined_logs) # Prioritize stderr on failure

        except docker.errors.ContainerError as ce:
             print(f"Docker ContainerError: ExitCode={ce.exit_status}, Cmd={ce.command}, Image={ce.image}, Err={ce.stderr}")
             exit_code = ce.exit_status
             stderr_log = ce.stderr.decode('utf-8', errors='ignore').strip() if ce.stderr else str(ce)
             # Try to get combined logs even on ContainerError
             try:
                 if container:
                     combined_logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore').strip()
                     print(f"DEBUG Combined Logs on ContainerError:\n--- START LOGS ---\n{combined_logs}\n--- END LOGS ---")
                     output_for_processing = stderr_log or combined_logs # Prioritize stderr from exception
                 else:
                      output_for_processing = stderr_log or f"ContainerError: {ce}"
             except Exception:
                  output_for_processing = stderr_log or f"ContainerError: {ce}" # Fallback

        finally:
             # Ensure container is removed
             if container:
                 try:
                     container.remove(force=True)
                     # print(f"Container '{container_name}' removed.") # Optional success log
                 except NotFound:
                     pass # Container already gone or failed to start
                 except APIError as e_rem:
                     print(f"Error removing container '{container_name}': {e_rem}")

        elapsed_time = time.time() - start_time
        print(f"Container '{container_name}' finished. Exit code: {exit_code}. Time: {elapsed_time:.2f}s")
        if stdout_log and exit_code != 0:
            print(f"Container Stdout (on error): {stdout_log[:200]}{'...' if len(stdout_log)>200 else ''}")
        if stderr_log:
            print(f"Container Stderr: {stderr_log[:200]}{'...' if len(stderr_log)>200 else ''}")

        # --- Process Results ---
        if exit_code == 0:
            # Return the combined logs for successful execution
            return True, output_for_processing
        else:
            # Command failed, return the prioritized error output
            error_message = output_for_processing or f"Container exited with code {exit_code} but no output captured."

            # Special handling for "AlreadyExists" on 'create' commands
            # Check the *original* command_input type and content
            original_verb = None
            original_executable = "unknown"
            if isinstance(command_input, list) and len(command_input) > 0:
                 original_executable = command_input[0]
                 if len(command_input) > 1:
                      original_verb = command_input[1]
            # Cannot reliably determine verb from original string input easily

            if original_executable in ['oc', 'kubectl'] and original_verb == 'create' and "AlreadyExists" in error_message:
                 print(f"Info: '{original_executable} create' reported 'AlreadyExists'. Treating as success.")
                 # Return success=True, but with a specific message, not the raw logs
                 return True, f"{original_executable.capitalize()} resource already exists (ignored)."
            else:
                # Return failure with the captured error message
                return False, error_message.strip()

    # --- Exception Handling for Docker/Setup issues ---
    except docker.errors.ImageNotFound as e:
         print(f"CRITICAL Docker ImageNotFound error: {e}")
         return False, f"Required Docker image '{CLI_DOCKER_IMAGE}' not found: {e}"
    except docker.errors.APIError as e:
        print(f"CRITICAL Docker API Error: {e}")
        if "permission denied" in str(e).lower():
             return False, f"Docker API permission error: Ensure the user running the backend has permissions for the Docker socket. Details: {e}"
        return False, f"Docker API error: {e}"
    except Exception as e:
        print(f"Unexpected Error in run_in_container: {e}")
        traceback.print_exc() # Log full stack trace for unexpected errors
        return False, f"Unexpected error during container execution: {e}"
    finally:
        # --- Final Cleanup of Temporary Files ---
        if temp_kubeconfig_filename and os.path.exists(temp_kubeconfig_filename):
             try:
                 os.remove(temp_kubeconfig_filename)
                 # print(f"Deleted temp kubeconfig: {temp_kubeconfig_filename}")
             except OSError as e_del: print(f"Error deleting temp kubeconfig {temp_kubeconfig_filename}: {e_del}")
        if temp_input_filename and os.path.exists(temp_input_filename):
             try:
                 os.remove(temp_input_filename)
                 # print(f"Deleted temp input file: {temp_input_filename}")
             except OSError as e_del: print(f"Error deleting temp input file {temp_input_filename}: {e_del}")


# --- Routes ---

@app.route('/')
def api_root():
    return jsonify({"message": "OpenShift Forward - AI Companion"}), 200

@app.route('/clear_session', methods=['GET', 'POST'])
def clear_user_session():
    """Clears the entire session for the current user."""
    session.clear()
    return redirect(url_for('api_root'))
    #return jsonify({"success": True, "message": "Session cleared."})


# Check initial login status based on session
@app.route('/check_login', methods=['GET'])
def check_login():
    if session.get('cluster_logged_in'):
        return jsonify({
            "isLoggedIn": True,
            "clusterType": session.get('cluster_type'),
            "clusterInfo": session.get('cluster_info')
        })
    else:
        return jsonify({"isLoggedIn": False})

@app.route('/chat', methods=['POST'])
def chat():
    if not openai_client or not llm: # Check both clients
         return jsonify({"error": "OpenAI client or LLM not initialized."}), 500

    data = request.json
    user_prompt = data.get('prompt')
    current_yaml_from_user = data.get('current_yaml')

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

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
        if enable_rag and rag_chain:
            print(f"Invoking RAG chain with model {llm_model}...")
            response = rag_chain.invoke({"input": combined_input})
            bot_reply_full = response.get("answer", "Sorry, I couldn't generate an answer using the available documents.").strip()
            print(f"RAG chain response received.")

        # --- Fallback to Non-RAG (if RAG failed or isn't setup) ---
        else:
            print(f"Using direct LLM call with model {llm_model}...")
            # Construct non_rag_messages including SYSTEM_PROMPT and combined_input potentially split
            # This part needs careful construction if combined_input is long
            non_rag_messages = session['conversation'] # includes system + user text prompt
            # Append the current YAML as a separate user message *for this call only*
            if current_yaml_from_user and current_yaml_from_user.strip():
                non_rag_messages.append({"role": "user", "content": f"[User is currently viewing/editing this YAML]:\n```yaml\n{current_yaml_from_user}\n```"})

            # Ensure system prompt is present
            if len(non_rag_messages) > 0 and non_rag_messages[0]["role"] != "system":
                non_rag_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            common_params = {
                "model": llm_model,
                "messages": non_rag_messages
            }
            print("non:", non_rag_messages)
            if llm_model != 'o3-mini':
                common_params["temperature"] = temperature

            response = openai_client.chat.completions.create(**common_params)
            bot_reply_full = response.choices[0].message.content.strip()
            print(f"Direct LLM response received.")
        # --- End RAG / LLM Call ---
        #print(f"Response:\n{bot_reply_full}") # debugging


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
        command_action = "apply" if action_type in ["oc_apply", "kubectl_apply"] else None
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
        print(f"Derived command_action (for submit): '{command_action}'")

        return jsonify(response_payload)

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": f"LLM or RAG processing error: {e}"}), 500


@app.route('/transcribe', methods=['POST'])
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


@app.route('/login', methods=['POST'])
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


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('cluster_logged_in', None)
    session.pop('cluster_type', None)
    session.pop('cluster_info', None)
    session.pop('last_command_tool', None)
    session.pop('last_command_action', None)
    session.modified = True
    return jsonify({"success": True})


@app.route('/submit', methods=['POST'])
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
    success, output = run_in_container(command, input_data=yaml_content, session_auth=auth_info_for_container)

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
