# Helper functions

# --- Standard Library Imports ---
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
OSF_KUBECONFIG_DIR = '/tmp/osf_kube'
OSF_EXEC_DIR = "/tmp/osf_exec"


def run_in_container(docker_client, command_input, session_auth=None, timeout=60, environment_vars=None, temp_dir_mapping=None):
    """
    Runs a command inside a Docker container using user-specific auth from session
    OR provided environment variables (e.g., for login). Handles input data.
    Assumes container ENTRYPOINT is ["/bin/bash", "-c"].

    Args:
        command_input (str): The command string to execute via /bin/bash -c.
        session_auth (dict, optional): User's auth info from Flask session.
        timeout (int, optional): Timeout in seconds for container execution. Defaults to 60.
        environment_vars (dict, optional): Environment variables to set inside the container.
                                           Used primarily for OpenShift password login.
        temp_dir_mapping (dict, optional): Mapping of host paths to container paths for volume mounts (e.g., {'/host/path/temp_run_id': '/app_temp'}).

    Returns:
        tuple: (success: bool, output: str)
               Output contains combined stdout/stderr on success,
               or prioritized stderr/error message on failure.
    """
    # --- Input Validation ---
    if not command_input:
        return False, "No command provided to run_in_container."
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
    exit_code = -1 # Initialize exit_code

    try:
        # --- Add Temporary Directory Volume Mount if mapping is provided ---
        if temp_dir_mapping:
            for host_path, container_path in temp_dir_mapping.items():
                if os.path.exists(host_path):
                    # Use read-only unless script needs to write back (unlikely for now)
                    volumes[host_path] = {'bind': container_path, 'mode': 'ro'}
                    print(f"DEBUG: Added temp execution volume mount: {host_path} -> {container_path}")
                else:
                    print(f"WARNING: Host path for temp execution directory mount does not exist: {host_path}")


        # --- Configure Auth based on session_auth (if provided - typically for actions like /submit) ---
        if session_auth:
            cluster_type = session_auth.get('cluster_type')
            if cluster_type == 'kubernetes':
                # Handle Kubernetes: Mount temporary kubeconfig
                context = session_auth.get('cluster_context')
                if not context:
                    return False, "Kubernetes context missing from session auth."
                try:
                    # --- Generate Temporary Kubeconfig ---
                    # This logic remains important for kubectl
                    print(f"Preparing temporary kubeconfig for context '{context}'...")
                    temp_kubeconfig_filename = os.path.join(OSF_KUBECONFIG_DIR, f"kubeconfig_{uuid.uuid4().hex}")
                    os.makedirs(OSF_KUBECONFIG_DIR, exist_ok=True) # Ensure base dir exists

                    # Load host config to extract context details
                    contexts, active_context = config.list_kube_config_contexts()
                    target_context = None
                    for c in contexts:
                        if c['name'] == context:
                            target_context = c
                            break
                    if not target_context:
                        raise ConfigException(f"Context '{context}' not found in host kubeconfig.")

                    # Find cluster and user details for the target context
                    config_dict = config.load_kube_config_to_dict() # Load the full config
                    target_cluster = None
                    target_user = None

                    for cluster in config_dict.get('clusters', []):
                        if cluster['name'] == target_context['context']['cluster']:
                            target_cluster = cluster
                            break
                    for user in config_dict.get('users', []):
                        if user['name'] == target_context['context']['user']:
                            target_user = user
                            break

                    if not target_cluster or not target_user:
                        raise ConfigException(f"Cluster or user details not found for context '{context}'.")

                    # Build minimal kubeconfig dict
                    temp_kubeconfig_dict = {
                         'apiVersion': 'v1',
                         'kind': 'Config',
                         'current-context': context,
                         'contexts': [{'name': context, 'context': target_context['context']}],
                         'clusters': [target_cluster],
                         'users': [target_user]
                    }
                    # --- End Generate Kubeconfig ---

                    with open(temp_kubeconfig_filename, 'w') as f:
                        yaml.dump(temp_kubeconfig_dict, f)
                    volumes[temp_kubeconfig_filename] = {'bind': container_kubeconfig_path, 'mode': 'ro'}
                    print(f"Prepared temporary kubeconfig for context '{context}' at {temp_kubeconfig_filename}")

                    # Set KUBECONFIG env var inside container
                    environment['KUBECONFIG'] = container_kubeconfig_path

                except (ConfigException, ApiException, Exception) as e:
                    print(f"ERROR preparing temporary kubeconfig for context '{context}': {e}")
                    traceback.print_exc()
                    if temp_kubeconfig_filename and os.path.exists(temp_kubeconfig_filename):
                        try: os.remove(temp_kubeconfig_filename);
                        except: pass
                    return False, f"Error preparing kubeconfig for context '{context}': {e}"

        # --- Run the Container ---
        if volumes:
            # Log final volumes, excluding sensitive kubeconfig path potentially
            logged_volumes = {k: v for k, v in volumes.items() if not k.endswith(temp_kubeconfig_filename or 'nevermatch')}
            print(f"Volumes mapped: {logged_volumes}")
        if environment:
            # Log environment variables, excluding sensitive password
            logged_environment = {k: (v if k != 'LOGIN_PASSWORD' else '***') for k, v in environment.items()}
            print(f"Environment Variables set: {logged_environment}")

        start_time = time.time()
        container = None
        combined_logs = ""
        stdout_log = ""
        stderr_log = ""

        try:
            # Command execution logic remains the same
            container = docker_client.containers.run(
                image=CLI_DOCKER_IMAGE,
                command=[command_input], # Wrap in list for bash -c entrypoint
                volumes=volumes,
                environment=environment,
                remove=False,      # Keep container after exit to retrieve logs
                detach=True,       # Start detached
                name=container_name,
                network_mode='host', # support crc+ssh
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
                    combined_logs = docker_client.logs(container.id, stdout=True, stderr=True).decode('utf-8', errors='ignore').strip()
                    print(f"DEBUG Combined Logs on ContainerError:\n--- START LOGS ---\n{combined_logs}\n--- END LOGS ---")
                    # If stderr from exception is available, use that primarily, otherwise combined
                    output_for_processing = stderr_log or combined_logs
                else:
                    # If container object wasn't even created, use stderr from exception or generic message
                    output_for_processing = stderr_log or f"ContainerError: {ce}"
            except Exception:
                # Fallback if logs can't be retrieved
                output_for_processing = stderr_log or f"ContainerError: {ce}"

        except Exception as e:
            print(f"Unexpected Error during docker.run or wait: {e}")
            traceback.print_exc() # Log full stack trace
            # exit_code might be uninitialized or -1
            output_for_processing = f"Unexpected error during container startup or execution: {e}"
            exit_code = -1 # Ensure exit_code reflects a general error

        finally:
            # Ensure container is removed
            if container:
                try:
                    # Get the container by ID to ensure remove is called on a valid object
                    docker_client.containers.get(container.id).remove(force=True)
                    # print(f"Container '{container_name}' removed.")
                except NotFound:
                    pass # Container already gone or failed to start/be found
                except APIError as e_rem:
                    print(f"Error removing container '{container_name}': {e_rem}")

        elapsed_time = time.time() - start_time
        print(f"Container '{container_name}' finished. Exit code: {exit_code}. Time: {elapsed_time:.2f}s")

        # --- Process Results ---
        if exit_code == 0:
            # Return the combined logs for successful execution
            return True, output_for_processing
        else:
            # Command failed, return the prioritized error output
            error_message = output_for_processing or f"Container exited with code {exit_code} but no output captured."
            return False, error_message.strip()

    # --- Outer Exceptions (Docker setup/connection issues, not container errors) ---
    except docker.errors.ImageNotFound as e:
        print(f"CRITICAL Docker ImageNotFound error: {e}")
        return False, f"Required Docker image '{CLI_DOCKER_IMAGE}' not found: {e}"
    except docker.errors.APIError as e:
        print(f"CRITICAL Docker API Error: {e}")
        if "permission denied" in str(e).lower():
            return False, f"Docker API permission error: Ensure the user running the backend has permissions for the Docker socket. Details: {e}"
        return False, f"Docker API error: {e}"
    except Exception as e:
        print(f"Unexpected Error in run_in_container during initial setup/connection: {e}")
        traceback.print_exc()
        return False, f"Unexpected error during container execution setup: {e}"

    finally: # Add a finally block here to ensure temp kubeconfig is cleaned up
        if temp_kubeconfig_filename and os.path.exists(temp_kubeconfig_filename):
            try:
                os.remove(temp_kubeconfig_filename)
            except OSError as e:
                print(f"Warning: Failed to remove temporary kubeconfig {temp_kubeconfig_filename}: {e}")


# For temporary kubeconfig files from K8s login/run_in_container
def cleanup_kube_configs():
    if os.path.exists(OSF_KUBECONFIG_DIR):
        print(f"Cleaning up temporary kubeconfig directory: {OSF_KUBECONFIG_DIR}")
        try:
            shutil.rmtree(OSF_KUBECONFIG_DIR, ignore_errors=True) # Use ignore_errors for robustness
        except OSError as e:
            print(f"Error removing temp directory {OSF_KUBECONFIG_DIR}: {e}")

# For temporary script/yaml files created by write_temp_files
# This cleans the base directory and all subdirs
def cleanup_exec_dir():
    if os.path.exists(OSF_EXEC_DIR):
        print(f"Cleaning up execution temporary directory: {OSF_EXEC_DIR}")
        try:
            # Use ignore_errors=True for robustness
            shutil.rmtree(OSF_EXEC_DIR, ignore_errors=True)
        except OSError as e:
            print(f"Error removing execution temp directory {OSF_EXEC_DIR}: {e}")
