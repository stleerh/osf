// frontend/src/components/SettingsDialog.jsx
import React, { useState, useEffect } from 'react';
import './SettingsDialog.css'; // Create this CSS file for styling

const SettingsDialog = ({
    isOpen,
    onClose,
    currentProvider,
    currentModel,
    onSave,
    availableModels // { openai: [...], ollama: { models: [...], error: null|string }, ibm_granite: [...] }
}) => {
    const [selectedProvider, setSelectedProvider] = useState(currentProvider);
    const [selectedModel, setSelectedModel] = useState(currentModel);

    // Update local state if props change (e.g., after saving)
    useEffect(() => {
        setSelectedProvider(currentProvider);
        setSelectedModel(currentModel);
    }, [currentProvider, currentModel]);

    const handleSave = () => {
        onSave(selectedProvider, selectedModel);
        onClose();
    };

    const renderModelSelector = () => {
        switch (selectedProvider) {
            case 'openai':
                return (
                    <select
                        id="openai-model-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={!availableModels?.openai?.length}
                    >
                        {availableModels?.openai?.map(model => (
                            <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                );
            case 'ollama': // Ollama is optional
                const ollamaData = availableModels?.ollama;
                return (
                     <>
                        {ollamaData?.error && <p className="error-text">Error fetching Ollama models: {ollamaData.error}</p>}
                         <select
                            id="ollama-model-select"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            disabled={!ollamaData?.models?.length || !!ollamaData?.error}
                        >
                            {!ollamaData?.models?.length && !ollamaData?.error && <option>Loading...</option>}
                            {ollamaData?.models?.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                     </>
                );
             case 'ibm_granite':
                 // Add inputs for API Key/Endpoint if needed, but prefer backend config
                 return (
                     <select
                        id="ibm-model-select"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        disabled={!availableModels?.ibm_granite?.length}
                     >
                         {availableModels?.ibm_granite?.map(model => (
                             <option key={model} value={model}>{model}</option>
                         ))}
                     </select>
                 );
            default:
                return <p>Select a provider.</p>;
        }
    };

    if (!isOpen) return null;

    return (
        <div className="settings-dialog-overlay">
            <div className="settings-dialog">
                <h2>LLM Settings</h2>
                <div className="settings-section">
                    <label>Provider:</label>
                    <div className="provider-options">
                        <label>
                            <input
                                type="radio"
                                name="provider"
                                value="openai"
                                checked={selectedProvider === 'openai'}
                                onChange={() => {
                                    setSelectedProvider('openai');
                                    // Set default model for the provider if available
                                    setSelectedModel(availableModels?.openai?.[0] || '');
                                }}
                            /> OpenAI
                        </label>
                        {availableModels?.ollama && !availableModels.ollama.error && (
                            <label>
                                <input
                                    type="radio"
                                    name="provider"
                                    value="ollama"
                                    checked={selectedProvider === 'ollama'}
                                    onChange={() => {
                                        setSelectedProvider('ollama');
                                        setSelectedModel(availableModels?.ollama?.models?.[0] || '');
                                    }}
                                /> Ollama
                            </label>
                        )}
                        <label style={{ display: 'none' }}> // Hide until implemented
                            <input
                                type="radio"
                                name="provider"
                                value="ibm_granite"
                                checked={selectedProvider === 'ibm_granite'}
                                onChange={() => {
                                    setSelectedProvider('ibm_granite');
                                    setSelectedModel(availableModels?.ibm_granite?.[0] || '');
                                }}
                            /> IBM Granite
                        </label>
                    </div>
                </div>

                <div className="settings-section">
                    <label htmlFor={`${selectedProvider}-model-select`}>Model:</label>
                    {renderModelSelector()}
                </div>

                <div className="settings-actions">
                    <button onClick={onClose} className="button-secondary">Cancel</button>
                    <button onClick={handleSave} className="button-primary" disabled={!selectedModel}>Save</button>
                </div>
            </div>
        </div>
    );
};

export default SettingsDialog;
