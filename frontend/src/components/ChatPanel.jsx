import React, { useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import InputArea from './InputArea'; // Import the new InputArea

function ChatPanel({ messages, onSendMessage, onStartRecording, onStopRecording, onCancelRecording, isRecording, isProcessing, isSubmittingYaml }) {
    const chatPanelRef = useRef(null);

    // Auto-scroll chat panel
    useEffect(() => {
        if (chatPanelRef.current) {
            chatPanelRef.current.scrollTop = chatPanelRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <div className="panel chat-panel-container">
            <h2>Chat</h2>
            <div id="chat-panel" className="scrollable" ref={chatPanelRef}>
                {messages.map((msg, index) => (
                    // Use a more stable key if messages can be inserted/deleted, index is ok for append-only
                    <ChatMessage key={index} sender={msg.sender} message={msg.message} type={msg.type} />
                ))}
            </div>
            {/* Use the InputArea component */}
            <InputArea
                 onSendMessage={onSendMessage}
                 onStartRecording={onStartRecording}
                 onStopRecording={onStopRecording}
                 onCancelRecording={onCancelRecording}
                 isRecording={isRecording}
                 isProcessing={isProcessing}
                 isSubmittingYaml={isSubmittingYaml}
            />
        </div>
    );
}

export default ChatPanel;
