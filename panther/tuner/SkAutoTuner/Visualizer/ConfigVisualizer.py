# filepath: d:\Gaser\Univeristy\GP\myPanther\panther\panther\utils\SkAutoTuner\ConfigVisualizer.py
import json
import logging
import webbrowser
from typing import Any, Dict, List, Optional

import torch.nn as nn

from ..layer_type_mapping import LAYER_TYPE_MAPPING
from .ModelVisualizer import ModelVisualizer

# Setup a logger for this module
logger = logging.getLogger(__name__)


class ConfigVisualizer(ModelVisualizer):
    """
    An extension of ModelVisualizer that allows users to generate
    SkAutoTuner configurations visual interface.
    """

    @staticmethod
    def create_config_visualization(
        model: nn.Module, output_path: Optional[str] = None, open_browser: bool = False
    ) -> str:
        """
        Creates an interactive visualization that allows users to select layers and
        generate SkAutoTuner configurations.

        Args:
            model: The neural network model to visualize and configure
            output_path: Path to save the generated HTML file (if None, a temporary file is created)
            open_browser: Whether to automatically open the visualization in a browser

        Returns:
            The path to the generated HTML file
        """
        # Generate basic model visualization HTML
        visualization_path = ModelVisualizer.create_interactive_visualization(
            model=model, output_path=output_path, open_browser=False
        )

        # Identify layers that support sketching
        layer_types = ConfigVisualizer._get_mappable_layer_types(model)

        # Available parameter options per layer type
        layer_config_options = ConfigVisualizer._get_layer_config_options()

        # Inject configuration UI into the HTML
        ConfigVisualizer._enhance_visualization_with_config_ui(
            visualization_path, layer_types, layer_config_options
        )

        # Optionally open the result in a browser
        if open_browser:
            webbrowser.open(f"file://{visualization_path}")
        return visualization_path

    @staticmethod
    def _get_mappable_layer_types(model: nn.Module) -> Dict[str, List[str]]:
        """
        Gets a dictionary mapping layer types found in the model to their sketchable parameters.

        Args:
            model: The neural network model

        Returns:
            Dictionary mapping layer types to their sketchable parameters
        """
        # Collect layers whose types are in our mapping
        layer_types: Dict[str, List[str]] = {}
        for name, module in model.named_modules():
            layer_type = type(module)
            if layer_type in LAYER_TYPE_MAPPING:
                type_name = layer_type.__name__
                layer_types.setdefault(type_name, []).append(name)
        return layer_types

    @staticmethod
    def _get_layer_config_options() -> Dict[str, Dict[str, List[Any]]]:
        """
        Gets the configuration options for each layer type.

        Returns:
            Dictionary mapping layer types to their configuration options
        """
        # This is a mapping from layer type to parameter options
        # In a real implementation, this would be more complete and based on the capabilities
        # of the SkAutoTuner's supported sketching parameters
        config_options = {
            "Linear": {
                "sketch": ["qr", "svd", "srd"],
                "sketch_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "Conv2d": {
                "sketch": ["qr", "svd", "srd"],
                "sketch_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "kernel_type": ["full", "sr"],
            },
            "MultiheadAttention": {
                "sketch": ["qr", "svd", "srd"],
                "sketch_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "head_ratio": [0.25, 0.5, 0.75],
            },
            # Add more layer types and their options as needed
        }

        return config_options

    @staticmethod
    def _enhance_visualization_with_config_ui(
        visualization_path: str,
        layer_types: Dict[str, List[str]],
        config_options: Dict[str, Dict[str, List[Any]]],
    ) -> None:
        """
        Enhances the HTML visualization with a configuration UI.

        Args:
            visualization_path: Path to the visualization HTML file
            layer_types: Dictionary mapping layer types to layer names
            config_options: Dictionary mapping layer types to configuration options
        """
        # Load existing HTML
        with open(visualization_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Prepare UI fragments
        config_ui_js = ConfigVisualizer._generate_config_ui_js(
            layer_types, config_options
        )
        config_ui_css = ConfigVisualizer._generate_config_ui_css()
        config_ui_html = ConfigVisualizer._generate_config_ui_html()

        # Inject CSS into existing <style> block
        html_content = html_content.replace("</style>", config_ui_css + "</style>")
        # Insert panel HTML and JS before closing body
        insertion = config_ui_html + "<script>" + config_ui_js + "</script>"
        html_content = html_content.replace("</body>", insertion + "</body>")

        # Save modifications
        with open(visualization_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    @staticmethod
    def _generate_config_ui_html() -> str:
        """
        Generates the HTML for the configuration UI.

        Returns:
            HTML string for the configuration UI
        """
        return """
        <div id="configPanel" class="config-panel">
            <div class="config-header">
                <h2>Layer Configuration Generator</h2>
                <button id="closeConfig" title="Close configuration panel">X</button>
            </div>
            <div class="config-body">        <div class="config-section">
                    <h3>Selected Layers <span class="selection-help">(Hold Ctrl/Cmd and click to select multiple layers)</span></h3>
                    <div class="selection-tip">
                        <div class="tip-icon">!</div>
                        <div class="tip-text">Select multiple layers by holding <kbd>Ctrl</kbd> (or <kbd>Cmd</kbd> on Mac) while clicking on layers in the visualization. A blue border will appear when multi-selection mode is active.</div>
                    </div>
                    <div id="selectedLayers" class="selected-layers">
                        <p class="no-selection">No layers selected. Click on layers in the visualization to select them.</p>
                    </div>
                </div>
                <div class="config-section">
                    <h3>Layer Parameters</h3>
                    <div id="paramOptions" class="param-options">
                        <p class="no-selection">Select layer(s) to configure parameters.</p>
                    </div>
                </div>
                <div class="config-section">
                    <h3>Configuration Options</h3>
                    <div class="config-options">
                        <label>
                            <input type="checkbox" id="separateConfig" checked>
                            Configure layers separately
                        </label>
                        <label>
                            <input type="checkbox" id="copyWeights" checked>
                            Copy weights when replacing layers
                        </label>
                    </div>
                </div>
                <div class="config-section">
                    <h3>Generated Configuration</h3>
                    <div class="code-container">
                        <pre id="generatedCode" class="generated-code">
# No configuration generated yet.
# Select layers and parameters to generate code.
</pre>
                        <button id="copyCode" class="code-button" title="Copy code to clipboard">Copy Code</button>
                    </div>
                </div>
                <div class="config-actions">
                    <button id="generateConfig" class="action-button">Generate Configuration</button>
                    <button id="clearSelection" class="action-button secondary">Clear Selection</button>
                </div>
            </div>
        </div>        <button id="showConfig" class="show-config-button">Generate Layer Config</button>
        <div id="multiSelectTooltip" class="multi-select-tooltip">Hold Ctrl/Cmd for multi-select</div>
        """

    @staticmethod
    def _generate_config_ui_css() -> str:
        """
        Generates the CSS for the configuration UI.

        Returns:
            CSS string for the configuration UI
        """
        # Basic styles for the configuration panel
        return """
        .config-panel {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .config-panel .config-body {
            background: white;
            padding: 20px;
            max-width: 700px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            overflow-y: auto;
            max-height: 90vh;
        }
        .config-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .config-header h2 {
            margin: 0;
            color: #333;
        }
        #closeConfig {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #777;
        }
        .config-section {
            margin-bottom: 20px;
        }
        .config-section h3 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #444;
            font-size: 16px;
        }
        .selected-layers {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            max-height: 150px;
            overflow-y: auto;
        }
        .layer-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            margin-bottom: 5px;
            background: #f5f5f5;
            border-radius: 3px;
        }
        .layer-item:last-child {
            margin-bottom: 0;
        }
        .layer-name {
            font-weight: bold;
            margin-right: 10px;
        }
        .layer-type {
            color: #777;
            font-size: 0.9em;
        }
        .remove-layer {
            background: none;
            border: none;
            color: #d33;
            cursor: pointer;
            font-size: 16px;
        }
        .param-group {
            margin-bottom: 15px;
        }
        .param-name {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .param-values {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .param-checkbox {
            display: flex;
            align-items: center;
            background: #f5f5f5;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .param-checkbox input {
            margin-right: 5px;
        }
        .config-options {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        .code-container {
            position: relative;
        }
        .generated-code {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: monospace;
            margin: 0;
        }
        .code-button {
            position: absolute;
            top: 5px;
            right: 5px;
            background: #eee;
            border: none;
            border-radius: 3px;
            padding: 3px 8px;
            cursor: pointer;
        }
        .code-button:hover {
            background: #ddd;
        }
        .config-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 20px;
        }
        .action-button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        .action-button:not(.secondary) {
            background: #4a86e8;
            color: white;
        }
        .action-button.secondary {
            background: #f5f5f5;
            color: #444;
        }        .no-selection {
            color: #777;
            font-style: italic;
        }
        .show-config-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #4a86e8;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .multi-select-tooltip {
            display: none;
            position: fixed;
            bottom: 60px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            pointer-events: none;
            z-index: 1001;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .node-selected {
            stroke: #4a86e8 !important;
            stroke-width: 3px !important;
        }
        .layer-type-group {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #fafafa;
        }
        .layer-type-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }
        .layer-type-items {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .type-section {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 15px;
            background-color: #fafafa;
        }
        .type-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }
        .type-warning {
            background-color: #fff8e6;
            border-left: 4px solid #ffc107;
            padding: 8px 12px;
            margin-bottom: 15px;
            color: #856404;
            font-size: 0.9em;
        }
        .param-options {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }
        .selection-help {
            font-size: 0.8em;
            color: #666;
            font-weight: normal;
            font-style: italic;
        }        .selection-tip {
            margin-bottom: 10px;
            padding: 10px;
            background-color: #f0f7ff;
            border-left: 4px solid #4a86e8;
            font-size: 0.9em;
            color: #333;
            display: flex;
            align-items: flex-start;
        }
        .tip-icon {
            font-size: 1.2em;
            margin-right: 10px;
        }
        .tip-text {
            flex: 1;
        }
        kbd {
            background-color: #f1f1f1;
            border: 1px solid #ccc;
            border-radius: 3px;
            box-shadow: 0 1px 0 rgba(0,0,0,0.2);
            color: #333;
            display: inline-block;
            font-family: monospace;
            font-size: 0.9em;
            line-height: 1;
            padding: 2px 5px;
            margin: 0 2px;
        }        .multi-selection-active {
            background-color: #e6f7ff !important;
            border: 2px dashed #4a86e8 !important;
            position: relative;
            transition: all 0.3s ease;
        }
        .multi-selection-active::before {
            content: "Multi-select mode";
            position: absolute;
            top: -18px;
            right: 10px;
            font-size: 12px;
            background: #4a86e8;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .layers-included {
            font-size: 0.85em;
            color: #555;
            margin-bottom: 10px;
            padding: 5px 0;
            border-bottom: 1px dotted #ddd;
        }
        .multi-select-tooltip {
            display: none;
            position: fixed;
            bottom: 60px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            pointer-events: none;
            z-index: 1001;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        """

    @staticmethod
    def _generate_config_ui_js(
        layer_types: Dict[str, List[str]],
        config_options: Dict[str, Dict[str, List[Any]]],
    ) -> str:
        """
        Generates the JavaScript for the configuration UI.

        Args:
            layer_types: Dictionary mapping layer types to layer names
            config_options: Dictionary mapping layer types to configuration options

        Returns:
            JavaScript string for the configuration UI
        """
        # Convert data to JSON strings for JavaScript
        layer_types_json = json.dumps(layer_types)
        config_options_json = json.dumps(config_options)

        return f"""
        // Layer types data from Python
        const layerTypes = {layer_types_json};
        // Configuration options from Python
        const configOptions = {config_options_json};
        // Store selected layers as array of objects: {{name, type}}
        let selectedLayers = [];
        // Map layer name to type for quick lookup
        let layerNameToType = {{}};

        // Initialize layer name to type mapping
        for (const [type, names] of Object.entries(layerTypes)) {{
            for (const name of names) {{
                layerNameToType[name] = type;
            }}
        }}

        // Helper: update selected layers UI
        function updateSelectedLayersUI() {{
            const container = document.getElementById('selectedLayers');

            if (selectedLayers.length === 0) {{
                container.innerHTML = '<p class="no-selection">No layers selected. Click on layers in the visualization to select them.</p>';
                return;
            }}

            // Group layers by type for better organization
            const layersByType = {{}};
            selectedLayers.forEach(layer => {{
                if (!layersByType[layer.type]) {{
                    layersByType[layer.type] = [];
                }}
                layersByType[layer.type].push(layer);
            }});

            let html = '';

            // Create a section for each type
            for (const [type, layers] of Object.entries(layersByType)) {{
                html += `<div class="layer-type-group">
                    <div class="layer-type-header">${{type}} (${{layers.length}})</div>
                    <div class="layer-type-items">`;

                for (const layer of layers) {{
                    html += `<div class="layer-item">
                        <span class="layer-name">${{layer.name}}</span>
                        <button class="remove-layer" title="Remove layer" data-layer="${{layer.name}}">&times;</button>
                    </div>`;
                }}

                html += `</div></div>`;
            }}

            container.innerHTML = html;
        }}

        // Helper: update parameter options UI for multiple types
        function updateParamOptionsUI() {{
            const container = document.getElementById('paramOptions');

            if (selectedLayers.length === 0) {{
                container.innerHTML = '<p class="no-selection">Select layer(s) to configure parameters.</p>';
                return;
            }}

            // Get all unique layer types from selected layers
            const uniqueTypes = [...new Set(selectedLayers.map(l => l.type))];

            let html = '';

            // Create UI for each type's parameters
            uniqueTypes.forEach(type => {{
                const layersOfType = selectedLayers.filter(l => l.type === type);
                const typeOptions = configOptions[type];

                if (!typeOptions || Object.keys(typeOptions).length === 0) {{
                    return;
                }}

                html += `<div class="type-section">
                    <div class="type-header">${{type}} (${{layersOfType.length}} layer${{layersOfType.length > 1 ? 's' : ''}})</div>`;

                // Show which layers are included when multiple are selected
                if (layersOfType.length > 1) {{
                    html += `<div class="layers-included">
                        Layers included: ${{layersOfType.map(l => l.name).join(', ')}}
                    </div>`;
                }}

                // Add parameter groups for this type
                for (const [paramName, paramValues] of Object.entries(typeOptions)) {{
                    html += `<div class="param-group" data-layer-type="${{type}}">
                        <span class="param-name">${{paramName}}</span>
                        <div class="param-values">`;

                    // Create checkbox for each value
                    for (const value of paramValues) {{
                        html += `<label class="param-checkbox">
                            <input type="checkbox" name="param-${{type}}-${{paramName}}" value="${{value}}" data-type="${{type}}" data-param="${{paramName}}">
                            ${{value}}
                        </label>`;
                    }}

                    html += `</div></div>`;
                }}

                html += `</div>`;
            }});

            container.innerHTML = html || '<p class="no-selection">No configurable parameters for selected layer types.</p>';
        }}

        // Helper: get selected parameter values grouped by layer type
        function getSelectedParamsByType() {{
            const paramsByType = {{}};

            // Get all parameter groups
            document.querySelectorAll('.param-group').forEach(group => {{
                const layerType = group.getAttribute('data-layer-type');
                if (!layerType) return;

                const paramName = group.querySelector('.param-name').textContent;
                const checkedInputs = group.querySelectorAll('input:checked');

                if (checkedInputs.length > 0) {{
                    if (!paramsByType[layerType]) {{
                        paramsByType[layerType] = {{}};
                    }}

                    paramsByType[layerType][paramName] = Array.from(checkedInputs).map(input => input.value);
                }}
            }});

            return paramsByType;
        }}

        // Helper: update generated code UI
        function updateGeneratedCodeUI() {{
            const codeBox = document.getElementById('generatedCode');

            if (selectedLayers.length === 0) {{
                codeBox.textContent = '# No configuration generated yet.\\n# Select layers and parameters to generate code.';
                return;
            }}

            const paramsByType = getSelectedParamsByType();

            if (Object.keys(paramsByType).length === 0) {{
                codeBox.textContent = '# Please select parameters for at least one layer type.';
                return;
            }}

            const separate = document.getElementById('separateConfig').checked;
            const copyWeights = document.getElementById('copyWeights').checked;

            // Group layers by type
            const layersByType = {{}};
            selectedLayers.forEach(layer => {{
                if (!layersByType[layer.type]) {{
                    layersByType[layer.type] = [];
                }}
                layersByType[layer.type].push(layer.name);
            }});

            // Generate configurations for each type
            const configs = [];

            for (const [type, layerNames] of Object.entries(layersByType)) {{
                // Only generate config if parameters are selected for this type
                if (paramsByType[type] && Object.keys(paramsByType[type]).length > 0) {{
                    configs.push(generateLayerConfigCode(layerNames, paramsByType[type], separate, copyWeights));
                }}
            }}

            // If no valid configs, show message
            if (configs.length === 0) {{
                codeBox.textContent = '# Please select parameters for at least one layer type.';
                return;
            }}

            // Generate full config code
            let fullCode = '';

            if (configs.length === 1) {{
                // If only one config, no need for TuningConfigs wrapper
                fullCode = configs[0];
            }} else {{
                // Wrap multiple configs in TuningConfigs
                fullCode = 'TuningConfigs(\\n    configs=[\\n' + configs.join(',\\n') + '\\n    ]\\n)';
            }}

            codeBox.textContent = fullCode;
        }}

        // Generate LayerConfig code
        function generateLayerConfigCode(layerNames, params, separate, copyWeights) {{
            let layerNamesStr;

            if (layerNames.length === 1) {{
                layerNamesStr = `\\"${{layerNames[0]}}\\"`;
            }} else {{
                layerNamesStr = `[${{layerNames.map(name => `\\"${{name}}\\"`).join(', ')}}]`;
            }}

            const paramsItems = [];

            for (const [paramName, paramValues] of Object.entries(params)) {{
                if (paramValues.length > 0) {{
                    const valuesStr = paramValues.map(value => {{
                        return isNaN(parseFloat(value)) ? `\\"${{value}}\\"` : value;
                    }}).join(', ');

                    paramsItems.push(`\\"${{paramName}}\\": [${{valuesStr}}]`);
                }}
            }}

            const paramsStr = `{{${{paramsItems.join(', ')}}}}`;

            return `    LayerConfig(\\n        layer_names=${{layerNamesStr}},\\n        params=${{paramsStr}},\\n        separate=${{separate}},\\n        copy_weights=${{copyWeights}}\\n    )`;
        }}

        // Show tooltip for multiselect
        function showMultiSelectTooltip() {{
            const tooltip = document.getElementById('multiSelectTooltip');
            tooltip.style.display = 'block';
            setTimeout(() => {{
                tooltip.style.opacity = '1';
            }}, 10);

            // Hide after 3 seconds
            setTimeout(() => {{
                tooltip.style.opacity = '0';
                setTimeout(() => {{
                    tooltip.style.display = 'none';
                }}, 300);
            }}, 3000);
        }}

        // Attach event listeners after DOM loaded
        document.addEventListener('DOMContentLoaded', function() {{
            // Show multiselect tooltip once when page loads
            setTimeout(showMultiSelectTooltip, 2000);

            // Show/hide config panel
            document.getElementById('showConfig').onclick = function() {{
                document.getElementById('configPanel').style.display = 'flex';
            }};

            document.getElementById('closeConfig').onclick = function() {{
                document.getElementById('configPanel').style.display = 'none';
            }};

            // Remove layer from selection
            document.getElementById('selectedLayers').onclick = function(e) {{
                if (e.target.classList.contains('remove-layer')) {{
                    const name = e.target.getAttribute('data-layer');
                    selectedLayers = selectedLayers.filter(l => l.name !== name);

                    // Unhighlight in visualization
                    unhighlightLayerNode(name);

                    // Update UI
                    updateSelectedLayersUI();
                    updateParamOptionsUI();
                    updateGeneratedCodeUI();
                }}
            }};

            // Clear selection
            document.getElementById('clearSelection').onclick = function() {{
                // Unhighlight all selected layers
                for (const layer of selectedLayers) {{
                    unhighlightLayerNode(layer.name);
                }}

                selectedLayers = [];

                // Update UI
                updateSelectedLayersUI();
                updateParamOptionsUI();
                updateGeneratedCodeUI();
            }};

            // Generate config
            document.getElementById('generateConfig').onclick = function() {{
                updateGeneratedCodeUI();
            }};

            // Copy code
            document.getElementById('copyCode').onclick = function() {{
                const code = document.getElementById('generatedCode').textContent;
                navigator.clipboard.writeText(code);

                // Show feedback
                const button = document.getElementById('copyCode');
                const originalText = button.textContent;
                button.textContent = 'Copied!';

                setTimeout(() => {{
                    button.textContent = originalText;
                }}, 2000);
            }};

            // Update config when options change
            document.getElementById('separateConfig').onchange = updateGeneratedCodeUI;
            document.getElementById('copyWeights').onchange = updateGeneratedCodeUI;

            // Update params when changed
            document.getElementById('paramOptions').addEventListener('change', function(e) {{
                if (e.target.tagName === 'INPUT' && e.target.type === 'checkbox') {{
                    updateGeneratedCodeUI();
                }}
            }});

            // Layer selection in visualization
            const svg = document.querySelector('svg');

            // Handle keydown/keyup for Ctrl/Cmd key to show multiselection is available
            document.addEventListener('keydown', function(e) {{
                if (e.ctrlKey || e.metaKey) {{
                    document.getElementById('selectedLayers').classList.add('multi-selection-active');
                    showMultiSelectTooltip();
                }}
            }});

            document.addEventListener('keyup', function(e) {{
                if (!e.ctrlKey && !e.metaKey) {{
                    document.getElementById('selectedLayers').classList.remove('multi-selection-active');
                }}
            }});

            if (svg) {{
                svg.addEventListener('click', function(e) {{
                    let node = e.target;

                    // Traverse up to find node with data-layer-name
                    while (node && !node.getAttribute('data-name') && node !== svg) {{
                        node = node.parentNode;
                    }}

                    if (node && node.getAttribute && node.getAttribute('data-name')) {{
                        const name = node.getAttribute('data-name');
                        // Root element is not a layer
                        if (name === 'root') return;
                        const type = layerNameToType[name];

                        if (!type) return; // Skip if not a recognized layer type

                        const idx = selectedLayers.findIndex(l => l.name === name);

                        // If not holding Ctrl key, clear other selections unless clicking on already selected layer
                        if (!e.ctrlKey && !e.metaKey && idx === -1) {{
                            // Clear previous selections
                            for (const layer of selectedLayers) {{
                                unhighlightLayerNode(layer.name);
                            }}
                            selectedLayers = [];
                        }}

                        // Toggle selection of clicked layer
                        if (idx === -1) {{
                            // Add to selection
                            selectedLayers.push({{name, type}});
                            highlightLayerNode(name);
                        }} else {{
                            // Remove from selection
                            selectedLayers.splice(idx, 1);
                            unhighlightLayerNode(name);
                        }}

                        // Update UI
                        updateSelectedLayersUI();
                        updateParamOptionsUI();
                        updateGeneratedCodeUI();
                    }}
                }});
            }}

            // Highlight/unhighlight helpers
            window.highlightLayerNode = function(name) {{
                // Find all nodes with data-name=name
                const nodes = document.querySelectorAll(`[data-name='${{name}}']`);
                nodes.forEach(n => n.classList.add('node-selected'));
            }};

            window.unhighlightLayerNode = function(name) {{
                const nodes = document.querySelectorAll(`[data-name='${{name}}']`);
                nodes.forEach(n => n.classList.remove('node-selected'));
            }};

            // Initial UI
            updateSelectedLayersUI();
            updateParamOptionsUI();
            updateGeneratedCodeUI();
        }});
        """


# Register this module to make it available via import
if __name__ == "__main__":
    print(
        "ConfigVisualizer module loaded. Use ConfigVisualizer.create_config_visualization() to create an interactive configuration tool."
    )
