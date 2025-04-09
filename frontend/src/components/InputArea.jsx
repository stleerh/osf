import React, { useState } from 'react';

function InputArea({ onSendMessage, onStartRecording, onStopRecording, onCancelRecording, isRecording, isProcessing, isSubmittingYaml }) {
    const [inputText, setInputText] = useState('');

    const handleSend = () => {
        if (inputText.trim()) {
            onSendMessage(inputText.trim());
            setInputText(''); // Clear input after sending
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleMicClick = () => {
        console.log("clcking");
        if (isRecording) {
            console.log("stop recording");
            onStopRecording();
        } else {
            console.log("start recording");
            onStartRecording();
        }
    }

    // Calculate if ANY blocking action is happening (for textarea/send button)
    const isAnyInputBlocked = isRecording || isProcessing || isSubmittingYaml;
    const canUseMic = !(isProcessing || isSubmittingYaml); // Mic enabled unless processing/submitting
    const canUseSendOrCancel = !(isProcessing || isSubmittingYaml); // Send/Cancel also blocked by processing/submitting

    return (
        <div className="input-area">
            <textarea
                id="user-input"
                placeholder="Enter prompt (Shift+Enter for new line)..."
                rows="3"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isAnyInputBlocked}
            />
            <div className="button-stack">
                <button
                    id="mic-button"
                    title={isRecording ? "Stop recording" : "Record prompt"}
                    onClick={handleMicClick}
                    className={isRecording ? 'recording' : ''}
                    // Mic button ONLY disabled if processing OR submitting
                    disabled={!canUseMic}
                >
                    <i className={`fas ${isRecording ? 'fa-stop' : 'fa-microphone'}`}></i>
                </button>

                {/* --- Conditional Send / Cancel Button --- */}
                {isRecording ? (
                    // Show Cancel button when recording
                    <button
                        id="cancel-button"
                        title="Cancel recording"
                        onClick={onCancelRecording} // <-- Use cancel handler
                        // Disabled if processing/submitting (though unlikely while recording)
                        disabled={!canUseSendOrCancel}
                    >
                        <i className="fas fa-window-close"></i> {/* Cancel Icon */}
                    </button>
                ) : (
                    // Show Send button when not recording
                    <button
                        id="send-button"
                        title="Send prompt (Enter)"
                        onClick={handleSend}
                        // Send button disabled if recording(should be false here)/processing/submitting OR no text
                        disabled={!canUseSendOrCancel || !inputText.trim()}
                    >
                        <i className="fas fa-paper-plane"></i>
                    </button>
                )}
                {/* --- End Conditional Button --- */}
            </div>
        </div>
    );
}

export default InputArea;
