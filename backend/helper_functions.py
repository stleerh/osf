# Helper functions

# --- Standard Library Imports ---
import atexit
import os
import shlex # Needed for shlex.quote
import shutil
import time
import traceback # For logging detailed errors
import uuid

# --- Third-Party Imports ---
import docker # The core Docker SDK
from docker.errors import APIError, NotFound, ImageNotFound, ContainerError # Specific Docker exceptions
import yaml # Needed for dumping kubeconfig YAML

# --- Kubernetes Client Imports
from kubernetes import config
from kubernetes import client
from kubernetes.config.config_exception import ConfigException
from kubernetes.client.exceptions import ApiException

# --- Global Constants ---
CLI_DOCKER_IMAGE = 'oc-image:latest' # Define the required Docker image
TEMP_KUBECONFIG_DIR = '/tmp/temp_kube_configs'


def run_in_container(docker_client, command_input, input_data=None, session_auth=None, timeout=60, environment_vars=None, command_is_string=False):
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

