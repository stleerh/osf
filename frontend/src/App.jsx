import React, { useState, useEffect, useRef } from 'react';
import LoginIndicator from './components/LoginIndicator';
import ChatPanel from './components/ChatPanel';
import YamlPanel from './components/YamlPanel';
import LoginDialog from './components/LoginDialog';
// Global styles are in index.css and imported in main.jsx.

function App() {
    const [messages, setMessages] = useState([]);
    const [yamlContent, setYamlContent] = useState('');
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [clusterInfo, setClusterInfo] = useState('');
    const [clusterType, setClusterType] = useState(null);
    const [lastCommandAction, setLastCommandAction] = useState(null);
    const [isLoginDialogOpen, setIsLoginDialogOpen] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false); // General lock for API calls (chat, transcribe)
    const [isSubmittingYaml, setIsSubmittingYaml] = useState(false); // Specific lock for YAML submit

    // --- Refs ---
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const streamRef = useRef(null);
    const isCancelledRef = useRef(false);

    const initialMessageAdded = useRef(false); // Use a ref to track

    // --- Handlers ---
    const addMessage = (sender, message, type = '') => {
        // Ensure message is a string, provide fallback for safety
        const messageText = typeof message === 'string' ? message : JSON.stringify(message) || '';
        setMessages(prev => [...prev, { sender, message: messageText, type }]);
    };

    const handleYamlChange = (newYaml) => {
        setYamlContent(newYaml);
    };


    // --- Microphone Handlers ---
    const stopMediaTracks = () => {
        if (streamRef.current) {
            console.log("App.jsx: Stopping media tracks.");
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
    };

    const handleStartRecording = async () => {
        // Reset cancellation flag at the start
        isCancelledRef.current = false;
        // Clear previous stream ref just in case
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (isRecording || isProcessing || isSubmittingYaml) return;
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addMessage('bot', 'Audio recording is not supported by your browser.', 'error');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Use a common mimeType if available, otherwise let the browser decide
            const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? { mimeType: 'audio/webm;codecs=opus' }
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? { mimeType: 'audio/webm' }
                    : {}; // Let browser choose default if webm isn't supported
            mediaRecorderRef.current = new MediaRecorder(stream, options);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = async () => {
                // --- Check Cancellation Flag ---
                if (isCancelledRef.current) {
                    console.log("App.jsx: Recording was cancelled, skipping audio processing.");
                    isCancelledRef.current = false; // Reset flag
                    // Ensure processing lock is released if it was somehow set
                    if (isProcessing) setIsProcessing(false);
                    return; // Exit early
                }
                // --- End Check ---

                const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm'; // Get actual mimeType used
                const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
                console.log(`App.jsx: Created audio blob, size: ${audioBlob.size}, type: ${mimeType}`);

                // Stop stream tracks *after* blob is created (or if cancelled)
                // streamRef.current?.getTracks().forEach(track => track.stop()); // Moved stopping tracks to stop/cancel handlers

                if (audioBlob.size > 0) {
                    await sendAudioToServer(audioBlob);
                } else {
                    console.warn("App.jsx: No audio data recorded in onstop.");
                    if (isProcessing) {
                        setIsProcessing(false); // Ensure lock is released if no data
                    }
                }
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
            //setIsProcessing(true); // Lock input while recording

        } catch (err) {
            console.error('Error accessing microphone:', err);
            let errorMsg = `Error accessing microphone: ${err.message}.`;
            if (err.name === 'NotAllowedError') {
                errorMsg += ' Please ensure microphone permission is granted.';
            } else if (err.name === 'NotFoundError') {
                errorMsg = 'No microphone found. Please ensure one is connected and enabled.';
            }
            addMessage('bot', errorMsg, 'error');
        }
    };

    const handleStopRecording = () => { // This is for NORMAL stop (process audio)
        console.log("App.jsx: handleStopRecording entered (normal stop).");
        isCancelledRef.current = false; // Ensure flag is false for normal stop

        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log("App.jsx: Calling mediaRecorderRef.current.stop() for normal stop.");
            mediaRecorderRef.current.stop(); // Triggers 'onstop' which checks flag
        } else {
            console.warn("App.jsx: Normal stop condition not met.");
            stopMediaTracks(); // Still try to cleanup tracks if recorder state is weird
        }
        setIsRecording(false); // Update UI immediately
        // Don't clear chunks here, onstop needs them
        // stopMediaTracks(); // Moved track stopping inside onstop/cancel handlers or here? Let's try stopping here too for robustness.
        stopMediaTracks(); // Stop tracks immediately on user action
    };

    // --- Cancel Recording ---
    const handleCancelRecording = () => {
        console.log("App.jsx: handleCancelRecording entered.");
        isCancelledRef.current = true; // Set flag to prevent processing in onstop

        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log("App.jsx: Calling mediaRecorderRef.current.stop() due to cancel.");
            mediaRecorderRef.current.stop(); // Triggers 'onstop', which will now exit early
        } else {
            console.warn("App.jsx: Cancel condition not met (recorder not recording?).");
        }

        // Clean up immediately
        audioChunksRef.current = []; // Clear any captured chunks
        stopMediaTracks(); // Stop the tracks
        setIsRecording(false); // Update UI state
         // Ensure processing lock is off if it was somehow engaged
         if (isProcessing) {
             console.log("App.jsx: Resetting isProcessing on cancel.");
             setIsProcessing(false);
         }
    };

    const sendAudioToServer = async (audioBlob) => {
        if (isProcessing || isSubmittingYaml) return; // Check locks again

        setIsProcessing(true); // Lock during transcription/sending
        addMessage('bot', 'Transcribing audio...', 'info');
        const formData = new FormData();
        // Provide a filename hint for the backend
        const filename = `recording.${audioBlob.type.split('/')[1]?.split(';')[0] || 'webm'}`;
        formData.append('audio', audioBlob, filename);

        try {
            const response = await fetch('/transcribe', {
                method: 'POST',
                credentials: 'include', // Include cookies if needed by backend (though less likely for transcribe)
                body: formData
                // Note: Don't set Content-Type header manually for FormData, browser does it
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP error! Status: ${response.status}` }));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            addMessage('bot', 'Transcription complete.', 'info');

            // Send the transcribed text as a new chat message
            if (data.text) {
                // Release the lock *before* calling handleSendMessage,
                // as it sets its own lock.
                setIsProcessing(false);
                await handleSendMessage(data.text);
            } else {
                 addMessage('bot', 'Received empty transcription.', 'info');
                 setIsProcessing(false); // Release lock if no text
            }

        } catch (error) {
            console.error('Error transcribing audio:', error);
            addMessage('bot', `Error transcribing audio: ${error.message}`, 'error');
            setIsProcessing(false); // Release lock on error
        }
    }


    const handleLoginSubmit = async (loginData) => {
        // This function is called by LoginDialog onSubmit
        // It returns { success: boolean, error?: string }
        setIsProcessing(true); // Use general processing lock during login

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                credentials: 'include', // Send cookies
                body: JSON.stringify(loginData)
            });

            const data = await response.json();

            if (data.success) {
                setIsLoggedIn(true);
                setClusterInfo(data.cluster_display);
                setClusterType(data.cluster_type);
                addMessage('bot', `Successfully logged into ${data.cluster_type} cluster: ${data.cluster_display}`, 'info');
                setYamlContent(''); // Clear state on successful login
                setLastCommandAction(null);
                setIsProcessing(false);
                return { success: true }; // Return success status
            } else {
                addMessage('bot', `Login failed: ${data.error}`, 'error'); // Show error in chat
                setIsLoggedIn(false); // Ensure logged out state
                setClusterInfo('');
                setClusterType(null);
                setIsProcessing(false);
                return { success: false, error: data.error || "Login failed." }; // Return error message
            }
        } catch (error) {
            console.error('Login request failed:', error);
            const errorMsg = `Login request network error: ${error.message}`;
            addMessage('bot', errorMsg, 'error');
            setIsLoggedIn(false);
            setClusterInfo('');
            setClusterType(null);
            setIsProcessing(false);
            return { success: false, error: errorMsg }; // Return network error
        }
    };

    const handleLogout = async () => {
        if (isProcessing || isSubmittingYaml) return;
        setIsProcessing(true);
        try {
            const response = await fetch('/logout', {
                method: 'POST',
                credentials: 'include' // Send cookies
            });
            const data = await response.json(); // Assume logout always returns JSON
            if (data.success) {
                setIsLoggedIn(false);
                setClusterInfo('');
                setClusterType(null);
                setYamlContent('');
                setLastCommandAction(null);
                addMessage('bot', 'Logged out successfully.', 'info');
            } else {
                addMessage('bot', `Logout failed: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Logout request failed:', error);
            addMessage('bot', `Logout request network error: ${error.message}`, 'error');
        } finally {
            setIsProcessing(false);
        }
    };


    // Submit handler (called by button OR by 'submit' action)
    const handleSubmitYaml = async (yaml, actionIntent) => {
        // actionIntent is 'apply' or 'create' based on lastCommandAction state
        if (!isLoggedIn || !yaml || !actionIntent || isProcessing || isSubmittingYaml) {
            console.warn("Submit YAML conditions not met inside handler.", {isLoggedIn, yaml: !!yaml, actionIntent, isProcessing, isSubmittingYaml});
            return;
        }

        addMessage('bot', `Submitting YAML via '${actionIntent}' intent...`, 'info');
        setIsSubmittingYaml(true);

        try {
            // Send the user's intent ('apply' or 'create') along with the YAML
            // The backend will verify this against the *specific* last suggested OSF action
            const response = await fetch('/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ yaml: yaml, command_action: actionIntent }) // Pass user intent
            });

            const data = await response.json();
            // Backend now returns tool_used based on its logic
            const toolUsed = data.tool_used || 'cluster tool';

            if (data.success) {
                addMessage('bot', `YAML submitted successfully via '${toolUsed} ${actionIntent}'.\nOutput:\n<pre>${data.output || '(No output)'}</pre>`, 'info');
            } else {
                const errorMessage = `Error submitting YAML via '${toolUsed} ${actionIntent}':\n<pre>${data.error || 'Unknown error'}</pre>`;
                addMessage('bot', errorMessage, 'error');
            }

        } catch (error) {
            console.error('Error submitting YAML:', error);
            addMessage('bot', `Submit YAML network error: ${error.message}`, 'error');
        } finally {
            setIsSubmittingYaml(false);
        }
    };


    // --- Chat Message Handler (Processing OSF Action) ---
    const handleSendMessage = async (prompt) => {
        if (!prompt || isProcessing || isSubmittingYaml) return; // Check all locks

        addMessage('user', prompt);
        setIsProcessing(true);
        setLastCommandAction(null); // Reset action/YAML on new user message

        // --- Prepare payload with current YAML ---
        const payload = {
            prompt: prompt,
            // Include the current state of the YAML panel in the request.
            // The backend can use this to inform the LLM's next response.
            current_yaml: yamlContent || null
        };

        try {
            // Use Vite proxy
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                // Send credentials (cookies) for session handling
                credentials: 'include',
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `HTTP error! Status: ${response.status}` }));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            addMessage('bot', data.reply);
            // Update YAML panel ONLY if the backend provides new YAML.
            // Do NOT overwrite user edits otherwise.
            if (data.yaml !== undefined && data.yaml !== null) {
                 setYamlContent(data.yaml); // Update YAML panel if provided
            }

            // Update state needed for submit button enable/disable
            setLastCommandAction(data.command_action || null);

            // --- Process the OSF Action ---
            const action = data.osf_action;
            if (action && action.type) {
                console.log(`Processing OSF Action: ${action.type}`, action.data || '');
                switch (action.type) {
                    case 'oc_apply':
                    case 'oc_create':
                    case 'kubectl_apply':
                    case 'kubectl_create':
                        // Action handled by enabling the submit button via lastCommandAction state.
                        // User clicks the button.
                        addMessage('bot', `Action suggested: ${action.type}. Use the 'Submit to Cluster' button after reviewing the YAML.`, 'info');
                        break;

                    case 'submit':
                        // Simulate clicking the submit button
                        addMessage('bot', 'Action requested: Submitting YAML to cluster...', 'info');
                        // Call the submit handler directly.
                        // It uses the *current* yamlContent state and the *derived* 'apply'/'create'
                        // state (lastCommandAction) which should have been set by the backend
                        // if the submit action was related to a previous apply/create suggestion.
                        // We need to ensure lastCommandAction has the correct 'apply'/'create' value
                        // *before* calling this. The backend logic now sets this correctly.
                        if (lastCommandAction && yamlContent) { // Check if there's an action and yaml
                            handleSubmitYaml(yamlContent, lastCommandAction);
                        } else {
                            addMessage('bot', 'Cannot auto-submit: No YAML content or associated action (apply/create) found.', 'error');
                        }
                        break; // <<<--- CHANGE HERE

                    case 'cmd':
                        // Display the command(s) to the user. Maybe add a "copy" button later?
                        // Add ; as newline for better readability if multiple commands
                        const commandsToShow = (action.data || '').replace(/;/g, ';\n');
                        addMessage('bot', `Suggested Command(s):\n<pre>${commandsToShow}</pre>`, 'info');
                        // Note: We are NOT executing these commands automatically for security.
                        break;
                    case 'login':
                        // Prompt user to log in, maybe open the dialog?
                        addMessage('bot', 'Action required: Please log in to proceed.', 'info');
                        // Optionally open the login dialog automatically:
                        if (!isLoggedIn) {
                             setIsLoginDialogOpen(true);
                        }
                        break;
                    case 'logout':
                        // Simulate clicking the logout button
                        addMessage('bot', 'Action requested: Logging out...', 'info');
                        handleLogout(); // Call the existing logout handler
                        break;
                    default:
                        console.warn(`Received unknown OSF Action type: ${action.type}`);
                }
            }
            // --- End Process OSF Action ---

        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('bot', `Error communicating with assistant: ${error.message}`, 'error');
            setLastCommandAction(null);
        } finally {
            setIsProcessing(false);
        }
    };


    // --- Effects ---
    // Check initial login status
    useEffect(() => {
        const checkInitialLogin = async () => {
            setIsProcessing(true); // Indicate loading
            try {
                // Use Vite proxy - path is relative to frontend host
                const response = await fetch('/check_login');
                if (!response.ok) {
                     // Try to get error message from backend if possible
                     let errorMsg = `HTTP error! Status: ${response.status}`;
                     try {
                         const errData = await response.json();
                         errorMsg = errData.error || errorMsg;
                     } catch (e) { /* ignore json parsing error */ }
                     throw new Error(errorMsg);
                }
                const data = await response.json();
                if (data.isLoggedIn) {
                    setIsLoggedIn(true);
                    setClusterInfo(data.clusterInfo);
                    setClusterType(data.clusterType);
                    if (!initialMessageAdded.current) {
                        addMessage('bot', `Resumed session. Logged into ${data.clusterType?.toUpperCase()} (${data.clusterInfo})`, 'info');
                        initialMessageAdded.current = true;
                    }
                } else {
                    if (!initialMessageAdded.current) {
                        addMessage('bot', 'Welcome! Enter a prompt or use the microphone. Login to OpenShift or Kubernetes to apply configurations.');
                        initialMessageAdded.current = true;
                    }
                }
            } catch (error) {
                console.error("Error checking login status:", error);
                if (!initialMessageAdded.current) {
                    addMessage('bot', `Could not check login status: ${error.message}. Please ensure the backend is running.`, 'error');
                    addMessage('bot', 'Welcome! Please enter a prompt or use the microphone.');
                    initialMessageAdded.current = true;
                }
            } finally {
                setIsProcessing(false);
            }
        };


        if (!initialMessageAdded.current) {
            checkInitialLogin();
        }
    }, []); // Keep empty dependency array


    // -- Render ---
    return (
        <div className="app-container">
            {/* Use header tag for semantics, style with CSS */}
            <header className="app-header">
                <h1>OpenShift Forward - AI Companion</h1>
                <LoginIndicator
                    isLoggedIn={isLoggedIn}
                    clusterInfo={clusterInfo}
                    clusterType={clusterType}
                    onLoginClick={() => setIsLoginDialogOpen(true)}
                    onLogoutClick={handleLogout}
                />
            </header>

            <main className="main-content">
                <ChatPanel
                    // Pass all needed props, including handlers and states
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    onStartRecording={handleStartRecording}
                    onStopRecording={handleStopRecording}
                    onCancelRecording={handleCancelRecording}
                    isRecording={isRecording}
                    isProcessing={isProcessing}
                    isSubmittingYaml={isSubmittingYaml}
                />
                <YamlPanel
                    yamlContent={yamlContent}
                    onYamlChange={handleYamlChange}
                    isLoggedIn={isLoggedIn}
                    // Pass lastCommandAction to enable/disable submit button
                    lastCommandAction={lastCommandAction}
                    onSubmitYaml={handleSubmitYaml}
                    isSubmittingYaml={isSubmittingYaml}
                />
            </main>

            <LoginDialog
                isOpen={isLoginDialogOpen}
                onClose={() => setIsLoginDialogOpen(false)}
                onLoginSubmit={handleLoginSubmit}
            />
        </div>
    );
}

export default App;
