import React, { useState, useEffect, useRef } from 'react';

function LoginDialog({ isOpen, onClose, onLoginSubmit }) {
    const [clusterType, setClusterType] = useState('openshift');
    const [url, setUrl] = useState('');
    const [username, setUsername] = useState('kubeadmin');
    const [password, setPassword] = useState('');
    const [context, setContext] = useState('');
    const [error, setError] = useState(''); // Local error state for form validation
    const [backendError, setBackendError] = useState(''); // Error from backend attempt
    const [isSubmitting, setIsSubmitting] = useState(false);
    const dialogRef = useRef(null);

    // Reset form when dialog opens/closes
    useEffect(() => {
        if (isOpen) {
            // Reset fields when opening
            setUrl('');
            setUsername('kubeadmin');
            setPassword('');
            setContext('');
            setError('');
            setBackendError(''); // Clear backend error too
            setIsSubmitting(false);
        }
    }, [isOpen]);

    // Clear errors and other type's fields when switching type
    useEffect(() => {
        setError('');
        setBackendError('');
        if (clusterType === 'openshift') {
            setContext('');
        } else {
            setUrl('');
            setUsername('kubeadmin');
            setPassword('');
        }
    }, [clusterType]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(''); // Clear local validation errors
        setBackendError(''); // Clear previous backend errors
        setIsSubmitting(true);

        let loginData = { cluster_type: clusterType };

        // Basic frontend validation
        if (clusterType === 'openshift') {
            if (!url || !username || !password) {
                setError("Please fill in all OpenShift fields.");
                setIsSubmitting(false);
                return;
            }
            loginData = { ...loginData, url, username, password };
        } else { // kubernetes
            if (!context) {
                setError("Please enter the Kubernetes context name.");
                setIsSubmitting(false);
                return;
            }
            loginData = { ...loginData, context };
        }

        // Call the login function passed from App.jsx
        const result = await onLoginSubmit(loginData);

        setIsSubmitting(false);

        if (!result.success) {
            // Display the error message from the backend
            setBackendError(result.error || "An unknown login error occurred.");
        } else {
            onClose();
        }
    };

    // Control dialog visibility using the ref
    useEffect(() => {
        if (dialogRef.current) {
            if (isOpen && !dialogRef.current.open) {
                dialogRef.current.showModal();
            } else if (!isOpen && dialogRef.current.open) {
                dialogRef.current.close();
            }
        }
    }, [isOpen]);

    // Handle manual close (like pressing Esc)
    const handleDialogClose = (e) => {
        // Check if the dialog is closing naturally (e.g., Esc key)
        // and call the onClose prop passed from the parent
        if (onClose && dialogRef.current && !dialogRef.current.open) {
            // This might fire too early or late depending on browser; onClose in App is safer
            // onClose();
        }
    };

    const handleCancelClick = () => {
        if (onClose) onClose();
    }


    return (
        // Render the dialog only when isOpen is true to simplify state management inside
        // Or always render and use CSS to hide? Using conditional rendering for simplicity.
        isOpen ? (
            <dialog ref={dialogRef} onClose={handleDialogClose} id="login-dialog">
                <h2>Cluster Login</h2>
                <form id="login-form" onSubmit={handleSubmit}>
                    {/* Cluster Type Radios */}
                    <div className="form-group">
                        <label>Cluster Type:</label>
                        <input
                            type="radio" id="type-openshift" name="cluster_type" value="openshift"
                            checked={clusterType === 'openshift'}
                            onChange={(e) => setClusterType(e.target.value)}
                            disabled={isSubmitting}
                        /> <label htmlFor="type-openshift">OpenShift</label>
                        <input
                            type="radio" id="type-kubernetes" name="cluster_type" value="kubernetes"
                            checked={clusterType === 'kubernetes'}
                            onChange={(e) => setClusterType(e.target.value)}
                            disabled={isSubmitting}
                        /> <label htmlFor="type-kubernetes">Kubernetes</label>
                    </div>

                    {/* OpenShift Fields */}
                    <div id="openshift-fields" style={{ display: clusterType === 'openshift' ? 'block' : 'none' }}>
                        <label htmlFor="cluster-url">Cluster URL:</label>
                        <input type="text" id="cluster-url" name="cluster-url" placeholder="api.your-cluster.com[:port]"
                                     value={url} onChange={(e) => setUrl(e.target.value)} disabled={isSubmitting} />
                        <label htmlFor="username">Username:</label>
                        <input type="text" id="username" name="username" value={username}
                                     onChange={(e) => setUsername(e.target.value)} disabled={isSubmitting} />
                        <label htmlFor="password">Password:</label>
                        <input type="password" id="password" name="password" value={password}
                                     onChange={(e) => setPassword(e.target.value)} disabled={isSubmitting} />
                    </div>

                    {/* Kubernetes Fields */}
                    <div id="kubernetes-fields" style={{ display: clusterType === 'kubernetes' ? 'block' : 'none' }}>
                        <label htmlFor="k8s-context">Kubeconfig Context:</label>
                        <input type="text" id="k8s-context" name="k8s-context" placeholder="e.g., kind-mycluster, minikube"
                                     value={context} onChange={(e) => setContext(e.target.value)} disabled={isSubmitting} />
                        <small>Assumes default kubeconfig location (~/.kube/config). Ensure the context is configured correctly.</small>
                    </div>

                    {/* Error Messages */}
                    {error && <p id="login-error" className="error-message">{error}</p>}
                    {backendError && <p id="backend-login-error" className="error-message">{backendError}</p>}


                    {/* Buttons */}
                    <div className="dialog-buttons">
                        <button type="submit" id="login-submit" disabled={isSubmitting}>
                            {isSubmitting ? 'Logging in...' : 'Login'}
                        </button>
                        <button type="button" id="login-cancel" onClick={handleCancelClick} disabled={isSubmitting}>
                            Cancel
                        </button>
                    </div>
                </form>
            </dialog>
        ) : null // Don't render the dialog if not open
    );
}

export default LoginDialog;
