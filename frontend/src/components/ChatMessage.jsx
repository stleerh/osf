import React from 'react';

// Same as previous version
function ChatMessage({ sender, message, type }) {
    const createMarkup = (htmlString) => {
        // WARNING: Only use if you trust the source or sanitize it first.
        // Used here for <pre> tags in info/error messages from the backend.
        return { __html: htmlString };
    };

    let displayName = 'Unknown';
    if (sender === 'user') {
        displayName = 'You';
    } else if (sender === 'bot') {
        displayName = 'OpenShift Bot';
    } else {
         // Capitalize other potential senders if needed
         displayName = sender.charAt(0).toUpperCase() + sender.slice(1);
    }

    const senderClass = sender === 'user' ? 'user' : 'bot'; // used for styling
    const typeClass = type || ''; // error, info

    return (
        <div className={`message ${senderClass} ${typeClass}`}>
            <strong>{displayName}:</strong>
            {/* Render HTML content safely if needed, otherwise render as text */}
            {(type === 'info' || type === 'error') && (message?.includes('<pre>') || message?.includes('</pre>')) ? (
                 <div dangerouslySetInnerHTML={createMarkup(message)} />
            ) : (
                <p>{message}</p>
            )}
        </div>
    );
}

export default ChatMessage;
