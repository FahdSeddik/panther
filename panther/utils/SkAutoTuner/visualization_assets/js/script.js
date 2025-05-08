// Setup interactive behavior
document.addEventListener('DOMContentLoaded', function() {
  const svgElement = document.querySelector('#svgPanZoomContainer svg'); // Target the SVG within the container
  const infoPanel = document.getElementById('infoPanel');
  const copyNotification = document.getElementById('copyNotification');
  const searchInput = document.getElementById('searchInput');
  const clearSearch = document.getElementById('clearSearch');
  const nodeMenu = document.getElementById('nodeMenu');
  const exportJSONBtn = document.getElementById('exportJSON');
  const expandAllBtn = document.getElementById('expandAll');
  const collapseAllBtn = document.getElementById('collapseAll');
  const themeToggleBtn = document.getElementById('themeToggle');
  const helpButton = document.getElementById('helpButton');
  const helpTooltip = document.getElementById('helpTooltip');
  const closeHelpTooltip = document.getElementById('closeHelpTooltip');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const visualizationContainer = document.getElementById('visualizationContainer');
  const resizeHandle = document.getElementById('resizeHandle');

  // Track collapsed state of nodes by their data-name attribute
  const collapsedNodes = new Set();
  let svgPanZoomInstance; // To store the svg-pan-zoom instance

  // Function to show notification messages
  function showNotification(message, type = 'info') {
    if (!copyNotification) return;
    
    // Clear any existing timeout
    if (window.notificationTimeout) {
      clearTimeout(window.notificationTimeout);
    }
    
    // Set message and show
    copyNotification.textContent = message;
    copyNotification.className = 'copy-notification'; // Reset classes
    copyNotification.classList.add(type); // Add type class (success, error, info)
    copyNotification.style.opacity = '1';
    
    // Hide after delay
    window.notificationTimeout = setTimeout(() => {
      copyNotification.style.opacity = '0';
    }, 2000);
  }

  // Show loading indicator
  function showLoading(show = true) {
    if (loadingIndicator) {
      loadingIndicator.style.display = show ? 'block' : 'none';
    }
  }

  // Initialize SVG Pan Zoom (if the library is available)
  function initSvgPanZoom() {
    if (window.svgPanZoom && svgElement) {
      svgPanZoomInstance = svgPanZoom(svgElement, {
        zoomEnabled: true,
        controlIconsEnabled: false, // We use custom controls
        panEnabled: true,
        minZoom: 0.1,
        maxZoom: 10,
        zoomScaleSensitivity: 0.2,
        fit: true,
        center: true,
        onZoom: function() {
            // Future use: if needed to update something on zoom
        },
        onPan: function() {
            // Future use: if needed to update something on pan
        }
      });
      // Custom zoom buttons
      document.getElementById('zoomIn').addEventListener('click', () => svgPanZoomInstance.zoomIn());
      document.getElementById('zoomOut').addEventListener('click', () => svgPanZoomInstance.zoomOut());
      document.getElementById('resetZoom').addEventListener('click', () => svgPanZoomInstance.resetZoom());
    } else {
      console.warn('svg-pan-zoom library not found or SVG element missing. Pan and zoom will be basic.');
      // Fallback basic zoom (if svg-pan-zoom is not used)
      let zoomLevel = 1;
      const zoomInBasic = document.getElementById('zoomIn');
      const zoomOutBasic = document.getElementById('zoomOut');
      const resetZoomBasic = document.getElementById('resetZoom');

      function updateBasicZoom() {
          if (svgElement) {
            svgElement.style.transform = `scale(${zoomLevel})`;
            svgElement.style.transformOrigin = 'center center'; // Or 'top left'
          }
      }
      if (zoomInBasic) zoomInBasic.addEventListener('click', () => { zoomLevel = Math.min(10, zoomLevel + 0.1); updateBasicZoom(); });
      if (zoomOutBasic) zoomOutBasic.addEventListener('click', () => { zoomLevel = Math.max(0.1, zoomLevel - 0.1); updateBasicZoom(); });
      if (resetZoomBasic) resetZoomBasic.addEventListener('click', () => { zoomLevel = 1; updateBasicZoom(); });
    }
  }
  
  // Call init after SVG content is loaded (or if it's inline, DOMContentLoaded is enough)
  // If SVG is loaded dynamically, this might need to be called after SVG is injected.
  if (svgElement) {
    initSvgPanZoom();
  } else {
    console.error("SVG element for pan/zoom not found.");
  }


  // Theme Toggling
  function applyTheme(theme) {
    document.body.classList.toggle('dark-theme', theme === 'dark');
    localStorage.setItem('visualizationTheme', theme);
  }

  themeToggleBtn.addEventListener('click', () => {
    const currentTheme = document.body.classList.contains('dark-theme') ? 'dark' : 'light';
    applyTheme(currentTheme === 'dark' ? 'light' : 'dark');
  });

  // Load saved theme
  const savedTheme = localStorage.getItem('visualizationTheme');
  if (savedTheme) {
    applyTheme(savedTheme);
  } else {
    // Default to light or respect OS preference if desired
    applyTheme('light'); 
  }

  // Help Tooltip
  function toggleHelp(show) {
    if (helpTooltip) {
        helpTooltip.style.display = show ? 'block' : 'none';
        helpTooltip.setAttribute('aria-hidden', !show);
        if (show) helpTooltip.focus(); // For accessibility
    }
  }
  if (helpButton) helpButton.addEventListener('click', (e) => { e.stopPropagation(); toggleHelp(helpTooltip.style.display !== 'block'); });
  if (closeHelpTooltip) closeHelpTooltip.addEventListener('click', () => toggleHelp(false));


  // Global click listener to hide menus/tooltips
  document.addEventListener('click', function(e) {
    if (nodeMenu) nodeMenu.style.display = 'none';
    // If help is open and click is outside, close it
    if (helpTooltip && helpTooltip.style.display === 'block' && !helpTooltip.contains(e.target) && e.target !== helpButton) {
        // toggleHelp(false); // This line was causing issues with the help button itself.
    }
  });
  
  // Export JSON
  if (exportJSONBtn) exportJSONBtn.addEventListener('click', function() {
      showLoading(true);
      try {
        const json = JSON.stringify(window.moduleInfo || {}, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model_info.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url); // Clean up
        showNotification('Model data exported as JSON.', 'success');
      } catch (err) {
        console.error('Failed to export JSON:', err);
        showNotification('Error exporting JSON.', 'error');
      } finally {
        showLoading(false);
      }
  });
  
  // Function to get all descendant nodes (recursively)
  function getAllDescendantNodes(parentNodePath) {
      if (!svgElement || !parentNodePath) return [];
      return Array.from(svgElement.querySelectorAll('[data-name]')).filter(n => {
          const path = n.getAttribute('data-name');
          return path && path !== parentNodePath && path.startsWith(parentNodePath + '.');
      });
  }

  // Function to get direct child nodes
  function getDirectChildNodes(parentNodePath) {
      if (!svgElement || !parentNodePath) return [];
      return Array.from(svgElement.querySelectorAll('[data-name]')).filter(n => {
          const path = n.getAttribute('data-name');
          if (!path || path === parentNodePath || !path.startsWith(parentNodePath + '.')) return false;
          // A direct child's path should be parentNodePath.childName (no further dots)
          const remainder = path.substring(parentNodePath.length + 1);
          return remainder.indexOf('.') === -1;
      });
  }
  
  // Function to toggle node collapse state
  function toggleNodeCollapse(nodeToToggle, collapse, forceRecursive = false) {
      const modulePath = nodeToToggle.getAttribute('data-name');
      if (!modulePath) return;

      const directChildren = getDirectChildNodes(modulePath);

      if (collapse) {
          nodeToToggle.classList.add('collapsed-node');
          collapsedNodes.add(modulePath);
          directChildren.forEach(child => {
              child.classList.add('hidden-node');
              // If recursively collapsing, also hide children of children
              if (forceRecursive || collapsedNodes.has(child.getAttribute('data-name'))) {
                 getAllDescendantNodes(child.getAttribute('data-name')).forEach(n => n.classList.add('hidden-node'));
              }
          });
      } else {
          nodeToToggle.classList.remove('collapsed-node');
          collapsedNodes.delete(modulePath);
          directChildren.forEach(child => {
              child.classList.remove('hidden-node');
              // If this child itself was collapsed, don't expand its children unless explicitly told to
              if (!collapsedNodes.has(child.getAttribute('data-name'))) {
                  // This logic ensures that expanding a parent doesn't auto-expand a child that was individually collapsed.
                  // To show children of this child, they must not be hidden due to their own parent being collapsed.
                  const grandChildren = getDirectChildNodes(child.getAttribute('data-name'));
                  grandChildren.forEach(gc => {
                      if (!collapsedNodes.has(child.getAttribute('data-name'))) { // only show if immediate parent is now expanded
                           gc.classList.remove('hidden-node');
                      }
                  });
              }
          });
      }
      updateNodeExpansionIndicator(nodeToToggle);
  }

  function updateNodeExpansionIndicator(node) {
    const indicator = node.querySelector('.expansion-indicator');
    if (indicator) {
        indicator.textContent = collapsedNodes.has(node.getAttribute('data-name')) ? '+' : '-';
    }
  }
  
  // Add handlers to all potential nodes
  function setupNodeInteractions() {
    if (!svgElement) return;
    showLoading(true); // Show loading before processing nodes

    const allGraphNodes = svgElement.querySelectorAll('g[id^="node_"]');
    allGraphNodes.forEach(node => {
        const modulePath = node.getAttribute('data-name');
        if (!modulePath) {
            // console.warn('Node found without data-name:', node.id);
            return; // Skip nodes without a data-name, they are not part of the module tree
        }

        // Add expansion indicator if it has children
        const directChildren = getDirectChildNodes(modulePath);
        if (directChildren.length > 0) {
            let indicator = node.querySelector('.expansion-indicator');
            if (!indicator) {
                const textElement = node.querySelector('text'); // Assuming text exists for label
                if (textElement) {
                    indicator = document.createElementNS("http://www.w3.org/2000/svg", "text");
                    indicator.setAttribute('class', 'expansion-indicator');
                    // Position relative to the existing text or a corner of the node
                    // This might need adjustment based on your node's SVG structure
                    const bbox = textElement.getBBox ? textElement.getBBox() : { x: 0, y: 0, width: 0, height: 0 };
                    indicator.setAttribute('x', bbox.x - 10); // Example positioning
                    indicator.setAttribute('y', bbox.y + bbox.height / 2 + 5); // Adjust as needed
                    indicator.style.cursor = 'pointer';
                    indicator.style.fontSize = '16px'; // Make it more visible
                    indicator.style.fontWeight = 'bold';
                    node.appendChild(indicator);
                }
            }
            if (indicator) {
                 indicator.textContent = collapsedNodes.has(modulePath) ? '+' : '-';
            }
        }


        node.style.cursor = 'pointer';
        
        // Double-click to collapse/expand
        node.addEventListener('dblclick', function(e) {
            e.stopPropagation();
            const currentModulePath = this.getAttribute('data-name');
            if (!currentModulePath) return;
            const isCollapsed = collapsedNodes.has(currentModulePath);
            toggleNodeCollapse(this, !isCollapsed);
        });
        
        // Single-click to show info and highlight
        node.addEventListener('click', function(e) {
            e.stopPropagation();
            const currentModulePath = this.getAttribute('data-name');
            if (!currentModulePath) return;
            
            displayModuleInfo(currentModulePath);
            
            allGraphNodes.forEach(n => n.classList.remove('highlight-node'));
            this.classList.add('highlight-node');
        });
        
        // Context menu
        node.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const currentModulePath = this.getAttribute('data-name');
            if (!currentModulePath) return;

            nodeMenu.style.display = 'block';
            nodeMenu.style.left = `${e.pageX}px`;
            nodeMenu.style.top = `${e.pageY}px`;
            nodeMenu.setAttribute('aria-hidden', 'false');
            
            // Update context menu items based on the node
            document.getElementById('copyName').onclick = (ev) => { ev.stopPropagation(); copyModuleName(currentModulePath); nodeMenu.style.display = 'none'; };
            document.getElementById('showDetails').onclick = (ev) => { ev.stopPropagation(); displayModuleInfo(currentModulePath); nodeMenu.style.display = 'none'; };
            document.getElementById('highlightPath').onclick = (ev) => { ev.stopPropagation(); highlightModulePath(currentModulePath); nodeMenu.style.display = 'none'; };
            
            const toggleCollapseItem = document.getElementById('toggleCollapse');
            const isNodeCollapsed = collapsedNodes.has(currentModulePath);
            toggleCollapseItem.textContent = isNodeCollapsed ? 'Expand Node' : 'Collapse Node';
            toggleCollapseItem.onclick = (ev) => {
                ev.stopPropagation();
                toggleNodeCollapse(this, !isNodeCollapsed);
                nodeMenu.style.display = 'none';
            };
        });

        // Make nodes draggable (Alt + Drag or Middle Mouse Drag)
        makeNodeDraggable(node);

        // Apply initially collapsed state if needed (e.g. after search or full collapse)
        if (collapsedNodes.has(modulePath) && !node.classList.contains('collapsed-node')) {
            toggleNodeCollapse(node, true, true); // force recursive hide
        } else if (!collapsedNodes.has(modulePath) && node.classList.contains('collapsed-node')) {
            // This case should ideally not happen if state is consistent
            node.classList.remove('collapsed-node');
        }
        updateNodeExpansionIndicator(node);
    });
    showLoading(false); // Hide loading after processing
  }
  
  // Call setupNodeInteractions once SVG is ready
  if (svgElement) {
      // If SVG is complex, defer to allow rendering first
      setTimeout(setupNodeInteractions, 100);
  }


  function copyModuleName(moduleName) {
    navigator.clipboard.writeText(moduleName).then(() => {
        showNotification(`Module name "${moduleName}" copied!`, 'success');
    }).catch(err => {
        console.error('Failed to copy module name: ', err);
        showNotification('Error copying name.', 'error');
    });
  }

  function highlightModulePath(moduleName) {
    if (!svgElement) return;
    const allGraphNodes = svgElement.querySelectorAll('g[id^="node_"]');
    allGraphNodes.forEach(n => n.classList.add('fade'));
    
    const parts = moduleName.split('.');
    let currentPath = '';
    for (let i = 0; i < parts.length; i++) {
        currentPath = i === 0 ? parts[i] : `${currentPath}.${parts[i]}`;
        const nodeToHighlight = svgElement.querySelector(`g[data-name="${currentPath}"]`);
        if (nodeToHighlight) {
            nodeToHighlight.classList.remove('fade');
            nodeToHighlight.classList.add('highlight-path-segment'); // For distinct path highlight
        }
    }
    
    setTimeout(() => { 
        allGraphNodes.forEach(n => {
            n.classList.remove('fade');
            n.classList.remove('highlight-path-segment');
        });
    }, 3000);
  }

  function makeNodeDraggable(nodeEl) {
      let isDragging = false;
      let dragOffsetX, dragOffsetY;
      let originalTransform = '';

      nodeEl.addEventListener('mousedown', function(e) {
          if (e.altKey || e.button === 1) { // Alt key or Middle mouse button
              e.stopPropagation(); // Prevent pan if svg-pan-zoom is active
              
              isDragging = true;
              nodeEl.classList.add('dragging');
              originalTransform = nodeEl.getAttribute('transform') || '';
              
              // Get initial mouse position relative to SVG viewport
              const CTM = svgElement.getScreenCTM();
              const initialMouseX = (e.clientX - CTM.e) / CTM.a;
              const initialMouseY = (e.clientY - CTM.f) / CTM.d;

              // Get current translation of the node
              let currentTranslateX = 0, currentTranslateY = 0;
              const transformMatch = originalTransform.match(/translate\s*\(([^,\s]+)[,\s]+([^\)]+)\)/);
              if (transformMatch) {
                  currentTranslateX = parseFloat(transformMatch[1]);
                  currentTranslateY = parseFloat(transformMatch[2]);
              }
              
              dragOffsetX = initialMouseX - currentTranslateX;
              dragOffsetY = initialMouseY - currentTranslateY;
              
              // Bring to front
              nodeEl.parentNode.appendChild(nodeEl);
              if(svgPanZoomInstance) svgPanZoomInstance.disablePan(); // Disable pan while dragging node
          }
      });

      document.addEventListener('mousemove', function(e) { // Listen on document for wider drag area
          if (isDragging) {
              e.preventDefault();
              const CTM = svgElement.getScreenCTM();
              const mouseX = (e.clientX - CTM.e) / CTM.a;
              const mouseY = (e.clientY - CTM.f) / CTM.d;
              
              const newTranslateX = mouseX - dragOffsetX;
              const newTranslateY = mouseY - dragOffsetY;
              
              nodeEl.setAttribute('transform', `translate(${newTranslateX}, ${newTranslateY})`);
              updateEdges(nodeEl, newTranslateX, newTranslateY);
          }
      });

      document.addEventListener('mouseup', function() {
          if (isDragging) {
              isDragging = false;
              nodeEl.classList.remove('dragging');
              if(svgPanZoomInstance) svgPanZoomInstance.enablePan();
          }
      });
  }
  
  // Basic edge update (highly dependent on Graphviz output structure)
  // This is a placeholder and likely needs significant refinement based on actual SVG structure of edges.
  function updateEdges(node, newTranslateX, newTranslateY) {
    if (!svgElement) return;
    const nodeId = node.getAttribute('id'); // Assumes node has an ID used by edges
    if (!nodeId) return;

    // This is a very simplified example. Edges in Graphviz are complex.
    // It's often better to re-render with Graphviz if dynamic edge updates are critical and complex.
    // Or, use a JS graph library that handles this.

    // Find edges connected to this node (data-source or data-target attributes were added in Python)
    const edges = svgElement.querySelectorAll(`path[data-source="${nodeId}"], path[data-target="${nodeId}"]`);
    
    edges.forEach(edge => {
        // Graphviz edges are paths. Their 'd' attribute defines the line.
        // Updating these correctly without knowing the exact start/end points relative
        // to the node's new position is very hard.
        // A common approach is to find the control points of the path and adjust them.
        // However, Graphviz's path 'd' attribute can be complex (e.g., splines).

        // For a very simple line, you might parse 'd', find M (moveto) and L (lineto)
        // and adjust. But this won't work for splines (C, S commands).
    });
  }
  
  // Collapse/Expand All
  if (expandAllBtn) expandAllBtn.addEventListener('click', function() {
      showLoading(true);
      if (!svgElement) { showLoading(false); return; }
      const allGraphNodes = svgElement.querySelectorAll('g[id^="node_"]');
      allGraphNodes.forEach(node => {
          const nodePath = node.getAttribute('data-name');
          if (nodePath && collapsedNodes.has(nodePath)) { // Only expand if it was collapsed
            toggleNodeCollapse(node, false);
          }
          node.classList.remove('hidden-node'); // Ensure all are visible
      });
      collapsedNodes.clear(); // All are now expanded
      allGraphNodes.forEach(node => updateNodeExpansionIndicator(node));
      showNotification('All nodes expanded.', 'info');
      showLoading(false);
  });
  
  if (collapseAllBtn) collapseAllBtn.addEventListener('click', function() {
      showLoading(true);
      if (!svgElement) { showLoading(false); return; }
      const allGraphNodes = svgElement.querySelectorAll('g[id^="node_"]');
      // Collapse all nodes that have children, starting from top-level
      const topLevelNodes = Array.from(allGraphNodes).filter(node => {
          const path = node.getAttribute('data-name');
          // A top-level node in the context of the graph is one whose data-name does not contain '.'
          // or it's the special 'root' node.
          return path && (path === 'root' || !path.includes('.'));
      });

      topLevelNodes.forEach(node => {
          const modulePath = node.getAttribute('data-name');
          if (getDirectChildNodes(modulePath).length > 0) { // Only collapse if it has children
            toggleNodeCollapse(node, true, true); // true for recursive collapse
          }
      });
      // Any node that is not a top-level but has children and is not already part of a collapsed parent path
      // should also be marked as collapsed if we want a "full" collapse.
      // For now, collapsing top-level recursively should handle most cases.
      allGraphNodes.forEach(node => updateNodeExpansionIndicator(node));
      showNotification('All expandable nodes collapsed.', 'info');
      showLoading(false);
  });
  
  // Function to display module information
  function displayModuleInfo(moduleName) {
      const info = window.moduleInfo ? window.moduleInfo[moduleName] : null;
      if (!infoPanel) return;

      if (!info) {
          infoPanel.innerHTML = `<h3 id="infoPanelHeader">Module: ${moduleName}</h3><p>No detailed information available.</p>`;
          return;
      }
      let html = `<h3 id="infoPanelHeader">Module: ${moduleName}</h3>`;
      html += `<div class="layer-path" lang="en" dir="ltr">${moduleName}</div>`; // Added lang/dir for accessibility
      html += `<button id="copyNameBtnInfoPanel" class="info-panel-button" style="margin-bottom:8px;">Copy Full Name</button>`;
      html += `<table aria-labelledby="infoPanelHeader">`; // Aria for table context
      html += `<thead><tr><th scope="col">Property</th><th scope="col">Value</th></tr></thead>`;
      html += `<tbody>`;
      html += `<tr><th scope="row">Type</th><td>${info.type || 'N/A'}</td></tr>`;
      html += `<tr><th scope="row">Parameters</th><td>${info.parameters?.toLocaleString() || '0'}</td></tr>`;
      html += `<tr><th scope="row">Trainable</th><td>${info.trainable ? 'Yes' : 'No'}</td></tr>`;
      html += `<tr><th scope="row">Class</th><td><code lang="en">${info.class || 'N/A'}</code></td></tr>`;
      html += `<tr><th scope="row">Docstring</th><td>${info.docstring || 'N/A'}</td></tr>`;
      
      for (const [key, value] of Object.entries(info)) {
          if (!['type', 'parameters', 'trainable', 'class', 'docstring'].includes(key)) {
              html += `<tr><th scope="row">${key.charAt(0).toUpperCase() + key.slice(1)}</th><td>${JSON.stringify(value)}</td></tr>`;
          }
      }
      html += `</tbody></table>`;
      infoPanel.innerHTML = html;
      
      const copyBtnInfo = document.getElementById('copyNameBtnInfoPanel');
      if (copyBtnInfo) {
          copyBtnInfo.onclick = function() {
              copyModuleName(moduleName);
          };
      }
  }
  
  // Enhanced search
  if (searchInput) searchInput.addEventListener('input', function() {
    const searchTerm = this.value.trim().toLowerCase();
    const allGraphNodes = svgElement ? svgElement.querySelectorAll('g[id^="node_"]') : [];

    if (searchTerm === '') {
        allGraphNodes.forEach(node => {
            node.style.opacity = 1;
            node.classList.remove('hidden-by-search');
            // Restore visibility based on collapsed state, not just make all visible
            const modulePath = node.getAttribute('data-name');
            if (modulePath) {
                const isHiddenByCollapse = Array.from(collapsedNodes).some(collapsedParentPath => 
                    modulePath.startsWith(collapsedParentPath + '.') && modulePath !== collapsedParentPath
                );
                if (isHiddenByCollapse) {
                    node.classList.add('hidden-node');
                }
            }
        });
        return;
    }

    // Determine if we're using a special filter syntax (key:value)
    const isSpecialFilter = searchTerm.includes(':');
    let filterKey, filterValue;
    
    if (isSpecialFilter) {
        const parts = searchTerm.split(':');
        filterKey = parts[0].trim();
        filterValue = parts.slice(1).join(':').trim(); // In case value contains colons too
    }

    // Search in moduleInfo data (if available) and highlight matching nodes
    const moduleInfoData = window.moduleInfo || {};
    
    allGraphNodes.forEach(node => {
        const modulePath = node.getAttribute('data-name');
        if (!modulePath) {
            node.classList.add('hidden-by-search');
            return;
        }
        
        // Default to hiding
        node.classList.add('hidden-by-search');
        node.style.opacity = 0.3;
        
        // Special filter handling
        if (isSpecialFilter) {
            const moduleData = moduleInfoData[modulePath];
            if (!moduleData) return; // No data to filter on
            
            // Handle different filter types
            let matchesFilter = false;
            
            switch(filterKey) {
                case 'type':
                    matchesFilter = moduleData.type && moduleData.type.toLowerCase().includes(filterValue);
                    break;
                case 'trainable':
                    matchesFilter = 
                        (filterValue === 'yes' && moduleData.trainable) || 
                        (filterValue === 'no' && !moduleData.trainable);
                    break;
                case 'params':
                case 'parameters':
                    if (filterValue.startsWith('>')) {
                        const threshold = parseInt(filterValue.substring(1).trim());
                        matchesFilter = !isNaN(threshold) && moduleData.parameters > threshold;
                    } else if (filterValue.startsWith('<')) {
                        const threshold = parseInt(filterValue.substring(1).trim());
                        matchesFilter = !isNaN(threshold) && moduleData.parameters < threshold;
                    } else {
                        const exactValue = parseInt(filterValue);
                        matchesFilter = !isNaN(exactValue) && moduleData.parameters === exactValue;
                    }
                    break;
                default:
                    // Try to match against any property
                    for (const [key, value] of Object.entries(moduleData)) {
                        if (
                            (key.toLowerCase() === filterKey && 
                             String(value).toLowerCase().includes(filterValue)) || 
                            (String(value).toLowerCase().includes(searchTerm))
                        ) {
                            matchesFilter = true;
                            break;
                        }
                    }
            }
            
            if (matchesFilter) {
                node.classList.remove('hidden-by-search');
                node.style.opacity = 1;
                // Also show all parent nodes in the path
                const parts = modulePath.split('.');
                let currentPath = '';
                for (let i = 0; i < parts.length; i++) {
                    currentPath = i === 0 ? parts[i] : `${currentPath}.${parts[i]}`;
                    const parentNode = svgElement.querySelector(`g[data-name="${currentPath}"]`);
                    if (parentNode) {
                        parentNode.classList.remove('hidden-by-search');
                        parentNode.style.opacity = 1;
                        // Ensure parent nodes are expanded
                        if (collapsedNodes.has(currentPath)) {
                            toggleNodeCollapse(parentNode, false);
                        }
                    }
                }
            }
            
        } else {
            // Simple name search
            if (modulePath.toLowerCase().includes(searchTerm)) {
                node.classList.remove('hidden-by-search');
                node.style.opacity = 1;
                
                // Make sure parents are visible and expanded
                const parts = modulePath.split('.');
                let currentPath = '';
                for (let i = 0; i < parts.length; i++) {
                    currentPath = i === 0 ? parts[i] : `${currentPath}.${parts[i]}`;
                    const parentNode = svgElement.querySelector(`g[data-name="${currentPath}"]`);
                    if (parentNode && parentNode !== node) {
                        parentNode.classList.remove('hidden-by-search');
                        parentNode.style.opacity = 1;
                        
                        // Expand parent if collapsed
                        if (collapsedNodes.has(currentPath)) {
                            toggleNodeCollapse(parentNode, false);
                        }
                    }
                }
            }
        }
    });
  });

  // Search clearing
  if (clearSearch) clearSearch.addEventListener('click', function() {
    if (searchInput) searchInput.value = '';
    // Trigger the input event to clear search results
    const event = new Event('input', { bubbles: true });
    searchInput.dispatchEvent(event);
    searchInput.focus();
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Focus search with Ctrl+F
    if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault(); // Prevent browser's find
        searchInput.focus();
    }
    
    // Clear search with Escape
    if (e.key === 'Escape') {
        if (searchInput && document.activeElement === searchInput) {
            searchInput.value = '';
            const event = new Event('input', { bubbles: true });
            searchInput.dispatchEvent(event);
        } else if (helpTooltip && helpTooltip.style.display === 'block') {
            toggleHelp(false);
        }
    }
    
    // Show help with ? or Ctrl+H
    if (e.key === '?' || ((e.ctrlKey || e.metaKey) && e.key === 'h')) {
        e.preventDefault();
        toggleHelp(helpTooltip.style.display !== 'block');
    }
  });

  // SVG-pan-zoom keyboard handlers
  document.addEventListener('keydown', function(e) {
      if (!svgPanZoomInstance) return;
      
      // Zoom with + and -
      if (e.key === '+' || e.key === '=') {
          svgPanZoomInstance.zoomIn();
      }
      if (e.key === '-' || e.key === '_') {
          svgPanZoomInstance.zoomOut();
      }
      // Reset with 0
      if (e.key === '0') {
          svgPanZoomInstance.resetZoom();
      }
  });
  
  // Make resize handle functional
  if (resizeHandle && visualizationContainer) {
      let startY, startHeight;
      
      resizeHandle.addEventListener('mousedown', function(e) {
          startY = e.clientY;
          startHeight = parseInt(document.defaultView.getComputedStyle(visualizationContainer).height, 10);
          document.documentElement.addEventListener('mousemove', doDrag, false);
          document.documentElement.addEventListener('mouseup', stopDrag, false);
      });
      
      function doDrag(e) {
          visualizationContainer.style.height = (startHeight + e.clientY - startY) + 'px';
      }
      
      function stopDrag() {
          document.documentElement.removeEventListener('mousemove', doDrag, false);
          document.documentElement.removeEventListener('mouseup', stopDrag, false);
          // After resize, make sure SVG pan-zoom instance is updated
          if (svgPanZoomInstance) {
              svgPanZoomInstance.resize();
              svgPanZoomInstance.fit();
              svgPanZoomInstance.center();
          }
      }
  }
  
  // Load svg-pan-zoom library dynamically if not available
  if (!window.svgPanZoom && svgElement) {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js';
      script.onload = function() {
          console.log('svg-pan-zoom library loaded dynamically');
          initSvgPanZoom();
      };
      document.head.appendChild(script);
  }
});