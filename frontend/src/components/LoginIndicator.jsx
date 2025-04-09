import React from 'react';

function LoginIndicator({ isLoggedIn, clusterInfo, clusterType, onLoginClick, onLogoutClick }) {
    const statusText = isLoggedIn
        ? `${clusterType?.toUpperCase()} (${clusterInfo})`
        : 'Status: Not Logged In';

    return (
        <div id="login-section">
            <span id="cluster-status" className={isLoggedIn ? 'logged-in' : ''}>
                    {statusText}
            </span>
            {isLoggedIn ? (
                <button id="logout-button" onClick={onLogoutClick} style={{ display: 'block' }}>Log Out</button>
            ) : (
                <button id="login-button" onClick={onLoginClick} style={{ display: 'block' }}>Log In</button>
            )}
        </div>
    );
}

export default LoginIndicator;
