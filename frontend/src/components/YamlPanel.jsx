import React from 'react';

function YamlPanel({ yamlContent, onYamlChange, isLoggedIn, onSubmitYaml, isSubmittingYaml }) {
    const hasYaml = yamlContent && yamlContent.trim().length > 0;
    const canSubmit = isLoggedIn && hasYaml && !isSubmittingYaml;

    let submitTitle = "Submit to Cluster";
    if (isSubmittingYaml) {
        submitTitle = "Submitting...";
    } else if (!isLoggedIn) {
        submitTitle = "Log in to a cluster first";
    } else if (!hasYaml) {
        submitTitle = "YAML panel is empty";
    }

    const handleChange = (event) => {
        // Call the handler passed from App.jsx to update the state
        onYamlChange(event.target.value);
    }

    return (
        <div className="panel yaml-panel-container">
            <h2>YAML</h2>
            <textarea
                id="yaml-panel"
                className="scrollable"
                spellCheck="false"
                value={yamlContent || ''}
                onChange={handleChange}
            />
            <button
                id="submit-button"
                disabled={!canSubmit}
                title={submitTitle}
                onClick={() => onSubmitYaml(yamlContent)}
            >
                {isSubmittingYaml ? 'Submitting...' : 'Submit to Cluster'}
            </button>
        </div>
    );
}

export default YamlPanel;
