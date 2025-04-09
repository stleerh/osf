import os
import subprocess
import re
import sys
import io
from urllib.parse import urlparse, urlunparse

from flask import Flask, render_template, request, jsonify, session
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

try:
    openai_client = OpenAI()
    llm = ChatOpenAI(model="o3-mini")
except Exception as e:
    print(f"Error initializing OpenAI client or Langchain Chat Model: {e}")
    openai_client = None
    llm = None


# --- RAG Setup ---
print("Initializing RAG...")
vector_store = load_or_build_vector_store()
if vector_store:
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
# --- End RAG Setup ---


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
    Parses the *OSF_ACTION: line from the end of the text.
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
            if stripped_line.startswith("*OSF_ACTION:"):
                action_line = stripped_line
            break # Found the last non-empty line, stop searching

    if not action_line:
        return None, None

    action_content_raw = action_line[len("*OSF_ACTION:"):].strip()

    # Handle potential command first, as it has '='
    if action_content_raw.startswith("cmd="):
        command_string = action_content_raw[len("cmd="):].strip()
        if command_string:
            return "cmd", command_string
        else:
            print("Warning: Found 'cmd=' action with empty command string.")
            return None, None

    # For simple keywords, strip common trailing non-alphanumeric chars like '*'
    # You could use regex for more complex cleaning if needed:
    # action_keyword = re.sub(r'[^\w_-]+$', '', action_content_raw)
    # Simpler approach: Check known keywords and allow trailing junk
    action_keyword_cleaned = action_content_raw.rstrip('*,.;:! ') # Remove common trailing chars

    KNOWN_SIMPLE_ACTIONS = [
        "oc_apply", "oc_create", "kubectl_apply",
        "kubectl_create", "submit", "login", "logout"
    ]

    if action_keyword_cleaned in KNOWN_SIMPLE_ACTIONS:
         print(f"Cleaned action keyword '{action_content_raw}' -> '{action_keyword_cleaned}'")
         return action_keyword_cleaned, None

    # If it wasn't cmd= or one of the cleaned known actions, parsing failed
    print(f"Warning: Found *OSF_ACTION: but couldn't parse content: '{action_content_raw}' (Cleaned: '{action_keyword_cleaned}')")
    return None, None # Unknown action

def remove_osf_action_line(text):
    """Removes the *OSF_ACTION: line if it's the last line."""
    lines = text.strip().split('\n')
    if lines and lines[-1].strip().startswith("*OSF_ACTION:"):
        return "\n".join(lines[:-1]).strip()
    return text.strip() # Return original text if no action line found at the end


def run_subprocess(command_args, input_data=None, timeout=60):
    """Runs a command using subprocess, returning success (bool) and output (str)."""
    executable = command_args[0] # e.g., 'oc' or 'kubectl'
    try:
        process = subprocess.run(
            command_args,
            input=input_data,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        if process.returncode == 0:
            return True, process.stdout.strip()
        else:
            error_output = process.stderr.strip() if process.stderr.strip() else process.stdout.strip()
            return False, error_output or f"Command '{executable}' failed with exit code {process.returncode}"
    except FileNotFoundError:
        return False, f"Error: '{executable}' command not found. Make sure it's installed and in your PATH."
    except subprocess.TimeoutExpired:
        return False, f"Error: Command '{executable}' timed out after {timeout} seconds."
    except Exception as e:
        return False, f"Error running command '{executable}': {e}"


# --- Routes ---

@app.route('/')
def api_root():
    return jsonify({"message": "OpenShift Forward - AI Companion"}), 200

@app.route('/clear_session', methods=['GET', 'POST'])
def clear_user_session():
    """Clears the entire session for the current user."""
    session.clear()
    return jsonify({"success": True, "message": "Session cleared."})


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
    session['conversation'].append({"role": "user", "content": user_prompt})
    max_history = 10 # Define max user/assistant pairs
    if len(session['conversation']) > max_history * 2 + 1:
         # Keep system prompt, remove oldest user/assistant pair(s) to meet limit
         # Calculate how many items to keep from the end (most recent pairs)
         items_to_keep = max_history * 2
         session['conversation'] = [session['conversation'][0]] + session['conversation'][-items_to_keep:]
         print(f"History truncated (after user msg). Length: {len(session['conversation'])}")
         session.modified = True
    # --- End User History Management ---

    bot_reply_full = "Sorry, something went wrong."

    combined_input = user_prompt # Start with user text
    if current_yaml_from_user and current_yaml_from_user.strip():
        combined_input += f"\n\n[User's Current YAML Editor Content]:\n```yaml\n{current_yaml_from_user}\n```"

    try:
        # --- RAG / LLM Call (as before, using combined_input) ---
        if rag_chain:
            print(f"Invoking RAG chain for prompt: {combined_input}")
            response = rag_chain.invoke({"input": combined_input})
            bot_reply_full = response.get("answer", "Sorry, I couldn't generate an answer using the available documents.").strip()
            print(f"RAG chain response received.")
        # --- Fallback to Non-RAG (if RAG failed or isn't setup) ---
        else:
            print(f"RAG not available. Using direct LLM call for prompt: {user_prompt}")
            # Construct non_rag_messages including SYSTEM_PROMPT and combined_input potentially split
            # This part needs careful construction if combined_input is long
            non_rag_messages = list(session['conversation']) # includes system + user text prompt
            if current_yaml_from_user and current_yaml_from_user.strip():
                non_rag_messages.append({"role": "user", "content": f"[User is currently viewing/editing this YAML]:\n```yaml\n{current_yaml_from_user}\n```"})
            if len(non_rag_messages) > 0 and non_rag_messages[0]["role"] != "system":
                non_rag_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            response = openai_client.chat.completions.create(
                model="o3-mini", messages=non_rag_messages
            )
            bot_reply_full = response.choices[0].message.content.strip()
        # --- End RAG / LLM Call ---
        print(f"Direct LLM response received.\n{bot_reply_full}")


        # --- Process Reply ---
        # 1. Parse the action *before* adding to history or modifying text
        action_type, action_data = parse_osf_action(bot_reply_full)

        # 2. Get text without action line
        user_visible_text_raw = remove_osf_action_line(bot_reply_full)

        # 3. Extract YAML content (doesn't include backticks)
        extracted_yaml_content = extract_yaml(user_visible_text_raw)

        # 4. Determine final reply for chat panel and YAML for panel
        final_reply_for_chat = user_visible_text_raw # Default to full text (minus action)
        yaml_for_panel = None

        if extracted_yaml_content:
            yaml_for_panel = extracted_yaml_content
            # Try to remove the full YAML block (including ```) from the raw text
            # Use a regex similar to extract_yaml but capture the whole block
            yaml_block_pattern = r"```(yaml|yml)?\s*([\s\S]*?)\s*```"
            match = re.search(yaml_block_pattern, user_visible_text_raw, re.MULTILINE)

            if match:
                # If we found the block, remove it and keep surrounding text
                full_yaml_block = match.group(0)
                # Replace only the first occurrence to avoid issues if multiple blocks exist
                text_without_yaml = user_visible_text_raw.replace(full_yaml_block, '', 1).strip()
                # Use remaining text if not empty, otherwise use canned message
                if text_without_yaml:
                     final_reply_for_chat = text_without_yaml
                     # Optionally add a note that YAML was moved
                     final_reply_for_chat += "\n\n(YAML placed in the panel.)"
                else:
                    # If removing YAML leaves nothing, use a canned message
                    final_reply_for_chat = "Okay, I've placed the generated YAML in the panel."
                    # Add command info if applicable (re-fetch derived command_action)
                    _derived_cmd_action = None
                    if action_type in ["oc_apply", "kubectl_apply"]: _derived_cmd_action = "apply"
                    elif action_type in ["oc_create", "kubectl_create"]: _derived_cmd_action = "create"
                    if _derived_cmd_action:
                         final_reply_for_chat += f" Review it and use 'Submit to Cluster' ({_derived_cmd_action})."
            else:
                 # Fallback: If regex didn't find the ``` block but extract_yaml did (e.g., raw YAML),
                 # it's harder to cleanly remove. Use the canned message.
                 final_reply_for_chat = "Okay, I've placed the generated YAML in the panel."
                 _derived_cmd_action = None
                 if action_type in ["oc_apply", "kubectl_apply"]: _derived_cmd_action = "apply"
                 elif action_type in ["oc_create", "kubectl_create"]: _derived_cmd_action = "create"
                 if _derived_cmd_action:
                     final_reply_for_chat += f" Review it and use 'Submit to Cluster' ({_derived_cmd_action})."

        # 5. Add the final chat reply (without YAML) to conversation history
        session['conversation'].append({"role": "assistant", "content": final_reply_for_chat})
        # Apply history limit *again* after adding assistant message
        # Same logic as after adding user message
        if len(session['conversation']) > max_history * 2 + 1:
             items_to_keep = max_history * 2
             session['conversation'] = [session['conversation'][0]] + session['conversation'][-items_to_keep:]
             print(f"History truncated (after assistant msg). Length: {len(session['conversation'])}")
             session.modified = True

        # 6. Store action details in session
        session['last_osf_action_type'] = action_type
        session['last_osf_action_data'] = action_data

        # For compatibility with submit logic, map yaml actions to command_action
        command_action = None
        if action_type in ["oc_apply", "kubectl_apply"]:
            command_action = "apply"
        elif action_type in ["oc_create", "kubectl_create"]:
            command_action = "create"
        # Store this derived action for the submit button logic
        session['last_command_action'] = command_action # Used by /submit endpoint check
        session.modified = True


        # --- Prepare JSON Response ---
        response_payload = {
            "reply": final_reply_for_chat, # Text without the action line
            "yaml": yaml_for_panel,        # YAML extracted from the text reply
            "osf_action": {                # New structured action info
                "type": action_type,       # e.g., "oc_apply", "cmd", "login", None
                "data": action_data        # e.g., None, "oc get pods", None
            }
            ,"command_action": command_action
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
    cluster_type = data.get('cluster_type') # 'openshift' or 'kubernetes'

    if cluster_type == 'openshift':
        url = data.get('url')
        username = data.get('username')
        password = data.get('password') # Handle password securely

        if not all([url, username, password]):
            return jsonify({"success": False, "error": "Missing OpenShift URL, username, or password"}), 400

        # Add default port 6443 if not specified and scheme https if missing
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme if parsed_url.scheme else "https"
        netloc = parsed_url.netloc or parsed_url.path # Handle case where scheme is missing
        port = parsed_url.port
        if not port:
             port = 6443
             # Reconstruct netloc with port
             hostname = parsed_url.hostname if parsed_url.hostname else netloc
             netloc = f"{hostname}:{port}"

        # Ensure path is empty for server URL
        full_url_obj = parsed_url._replace(scheme=scheme, netloc=netloc, path='', params='', query='', fragment='')
        full_url = urlunparse(full_url_obj)
        display_info = netloc # Display hostname:port

        login_command = [
            "oc", "login",
            f"-u={username}",
            f"-p={password}",
            full_url,
            "--insecure-skip-tls-verify=true" # Keep for dev, remove/review for prod
        ]

        print(f"Attempting OpenShift login (password omitted): oc login -u={username} -p=*** {full_url} --insecure-skip-tls-verify=true")
        success, output = run_subprocess(login_command)

    elif cluster_type == 'kubernetes':
        context = data.get('context')
        if not context:
            return jsonify({"success": False, "error": "Missing Kubernetes context name"}), 400

        # 1. Try to switch context
        switch_command = ["kubectl", "config", "use-context", context]
        print(f"Attempting Kubernetes context switch: {' '.join(switch_command)}")
        switch_success, switch_output = run_subprocess(switch_command)

        if not switch_success:
            print(f"Context switch failed: {switch_output}")
            # Distinguish between context not found and other errors
            if "no context exists" in switch_output.lower():
                 error_msg = f"Context '{context}' not found in kubeconfig."
            else:
                 error_msg = f"Failed to switch context: {switch_output}"
            return jsonify({"success": False, "error": error_msg})

        # 2. Verify connection with the new context
        verify_command = ["kubectl", "cluster-info", "--context", context]
         # Alternative: kubectl get ns --context <context> -o name --request-timeout=5s
        print(f"Verifying Kubernetes connection: {' '.join(verify_command)}")
        success, output = run_subprocess(verify_command, timeout=15) # Shorter timeout for verification

        if not success:
             # Attempt to provide a more specific error if possible
             if "Unable to connect" in output or "connection refused" in output:
                 error_msg = f"Switched to context '{context}' but failed to connect to the cluster: {output}"
             else:
                error_msg = f"Switched to context '{context}' but verification failed: {output}"
             # Potentially revert context switch here if desired, though maybe not necessary
             # run_subprocess(["kubectl", "config", "use-context", original_context]) # Needs original_context saved
             return jsonify({"success": False, "error": error_msg})

        display_info = context # Display the context name

    else:
        return jsonify({"success": False, "error": "Invalid cluster type specified"}), 400

    # Common success path
    if success:
        print(f"{cluster_type.capitalize()} login/context switch successful.")
        session['cluster_logged_in'] = True
        session['cluster_type'] = cluster_type # 'openshift' or 'kubernetes'
        session['cluster_info'] = display_info # URL host:port or context name
        # Clear previous command state on new login
        session.pop('last_osf_action_type', None) # Clear command state on new login
        session.pop('last_osf_action_data', None)
        session.pop('last_command_action', None)
        session.modified = True
        return jsonify({
            "success": True,
            "cluster_display": display_info, # For frontend clusterInfo state
            "cluster_type": cluster_type
        })
    else:
        print(f"{cluster_type.capitalize()} login/verification failed: {output}")
        session.pop('cluster_logged_in', None)
        session.pop('cluster_type', None)
        session.pop('cluster_info', None)
        session.modified = True
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
    # The 'command_action' sent from frontend ('apply'/'create') still determines user intent for THIS submission
    requested_action_intent = data.get('command_action')

    if not yaml_content:
        return jsonify({"success": False, "error": "YAML content is empty."}), 400
    if requested_action_intent not in ['apply', 'create']:
         return jsonify({"success": False, "error": "Invalid command action specified."}), 400

    # Retrieve the last action suggested by the LLM from the session
    last_osf_action_type = session.get('last_osf_action_type')

    last_suggested_action_verb = None
    # Derive the action verb ('apply' or 'create') from the specific type
    if last_osf_action_type in ["oc_apply", "kubectl_apply"]:
        last_suggested_action_verb = "apply"
    elif last_osf_action_type in ["oc_create", "kubectl_create"]:
        last_suggested_action_verb = "create"

    # --- Security/Consistency Check ---
    # Check if the user's requested intent (apply/create) matches the verb derived from the LLM's last specific suggestion.
    if requested_action_intent != last_suggested_action_verb:
        # Define the error message *inside* the block where the mismatch is confirmed
        error_msg = f"Action mismatch. User intent is '{requested_action_intent}', but last suggested action was '{last_osf_action_type}' (implying '{last_suggested_action_verb}'). Ask the assistant again."
        if not last_osf_action_type:
             error_msg = f"Action mismatch. User intent is '{requested_action_intent}', but no action was suggested by the assistant recently."

        print(f"Submit Action Mismatch: Requested='{requested_action_intent}', Session Last OSF Action='{last_osf_action_type}' (Verb='{last_suggested_action_verb}')")
        return jsonify({"success": False, "error": error_msg}), 400
    # --- End Check ---


    cluster_type = session.get('cluster_type')
    cluster_info = session.get('cluster_info') # Context name for K8s, display URL for OS

    if not cluster_type:
         return jsonify({"success": False, "error": "Cluster type not found in session. Please log in again."}), 500

    tool = None
    command = []

    # Determine tool based on the *specific* OSF action type stored in session
    if last_osf_action_type == "oc_apply" or last_osf_action_type == "oc_create":
        tool = 'oc'
        if cluster_type != 'openshift': return jsonify({"success": False, "error": f"Action '{last_osf_action_type}' requires OpenShift login, but currently logged into {cluster_type}."}), 400
        command = [tool, requested_action_intent, "-f", "-"] # Use user's intent here
    elif last_osf_action_type == "kubectl_apply" or last_osf_action_type == "kubectl_create":
        tool = 'kubectl'
        if cluster_type != 'kubernetes': return jsonify({"success": False, "error": f"Action '{last_osf_action_type}' requires Kubernetes login, but currently logged into {cluster_type}."}), 400
        if not cluster_info: return jsonify({"success": False, "error": "Kubernetes context not found in session."}), 500
        command = [tool, requested_action_intent, "--context", cluster_info, "-f", "-"] # Use user's intent
    elif last_osf_action_type == "submit":
        # "submit" action type - use default tool based on current login
        if cluster_type == 'openshift':
             tool = 'oc'
             command = [tool, requested_action_intent, "-f", "-"]
        elif cluster_type == 'kubernetes':
             tool = 'kubectl'
             if not cluster_info: return jsonify({"success": False, "error": "Kubernetes context not found for 'submit' action."}), 500
             command = [tool, requested_action_intent, "--context", cluster_info, "-f", "-"]
        else:
             return jsonify({"success": False, "error": "Cannot perform 'submit' action without known cluster type."}), 400
    else:
        # Should not happen if consistency check passed, but handle anyway
        return jsonify({"success": False, "error": f"Cannot submit YAML with last action type: '{last_osf_action_type}'"}), 400


    print(f"Running command: {' '.join(command)}")
    success, output = run_subprocess(command, input_data=yaml_content)
    if success:
        print(f"{tool} {requested_action_intent} successful.")
        return jsonify({"success": True, "output": output, "tool_used": tool})
    else:
        print(f"{tool} {requested_action_intent} failed: {output}")
        return jsonify({"success": False, "error": output, "tool_used": tool})


# --- Main Execution ---
if __name__ == '__main__':
    if not openai_client:
        print("WARNING: OpenAI client failed to initialize. LLM features will not work.")
    if not vector_store:
        print("WARNING: RAG vector store failed to initialize. Chatbot will not use retrieved context.")
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
