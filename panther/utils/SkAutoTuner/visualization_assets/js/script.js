// Setup interactive behavior
document.addEventListener('DOMContentLoaded', function() {
  const svg = document.querySelector('svg');
  const infoPanel = document.getElementById('infoPanel');
  const copyNotification = document.getElementById('copyNotification');
  const searchInput = document.getElementById('searchInput');
  const clearSearch = document.getElementById('clearSearch');
  const nodeMenu = document.getElementById('nodeMenu');
  const exportJSONBtn = document.getElementById('exportJSON');
  const expandAllBtn = document.getElementById('expandAll');
  const collapseAllBtn = document.getElementById('collapseAll');
  
  // Track collapsed state of nodes
  const collapsedNodes = new Set();
  
  // Add context menu to SVG
  document.addEventListener('click', function() {
      nodeMenu.style.display = 'none';
  });

  // Zoom functionality
  let zoomLevel = 1;
  const zoomIn = document.getElementById('zoomIn');
  const zoomOut = document.getElementById('zoomOut');
  const resetZoom = document.getElementById('resetZoom');
  
  function updateZoom() {
      svg.style.transform = `scale(${zoomLevel})`;
      svg.style.transformOrigin = 'top left';
  }
  
  zoomIn.addEventListener('click', () => {
      zoomLevel += 0.1;
      updateZoom();
  });
  
  zoomOut.addEventListener('click', () => {
      if (zoomLevel > 0.2) zoomLevel -= 0.1;
      updateZoom();
  });
  
  resetZoom.addEventListener('click', () => {
      zoomLevel = 1;
      updateZoom();
  });
  
  // Export JSON
  exportJSONBtn.addEventListener('click', function() {
      const json = JSON.stringify(moduleInfo, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model_info.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
  });
  
  // Function to get children of a node
  function getChildNodes(node) {
      const modulePath = node.getAttribute('data-name');
      if (!modulePath) return [];
      
      return Array.from(svg.querySelectorAll('[data-name]')).filter(n => {
          const path = n.getAttribute('data-name');
          return path && path !== modulePath && path.startsWith(modulePath + '.');
      });
  }
  
  // Function to toggle node collapse
  function toggleNodeCollapse(node, collapse) {
      const modulePath = node.getAttribute('data-name');
      if (!modulePath) return;
      
      // Get direct children
      const childNodes = getChildNodes(node);
      const directChildren = childNodes.filter(n => {
          const path = n.getAttribute('data-name');
          const pathParts = path.replace(modulePath + '.', '').split('.');
          return pathParts.length === 1;
      });

      // Toggle visibility of direct children
      directChildren.forEach(child => {
          if (collapse) {
              child.classList.add('hidden-node');
              // Add to collapsed set
              collapsedNodes.add(modulePath);
              // Also hide any descendants
              getChildNodes(child).forEach(n => n.classList.add('hidden-node'));
          } else {
              child.classList.remove('hidden-node');
              // Remove from collapsed set
              collapsedNodes.delete(modulePath);
              // Only show direct descendants, don't expand everything
              getChildNodes(child).forEach(n => {
                  const parentPath = n.getAttribute('data-name').split('.');
                  parentPath.pop(); // Remove last part to get parent
                  const immediateParent = parentPath.join('.');
                  // Only show it if its immediate parent is not collapsed
                  if (!collapsedNodes.has(immediateParent)) {
                      n.classList.remove('hidden-node');
                  }
              });
          }
      });
      
      // Update node appearance
      if (collapse) {
          node.classList.add('collapsed-node');
      } else {
          node.classList.remove('collapsed-node');
      }
  }
  
  // Add double-click handlers to nodes for collapsing/expanding
  const nodes = svg.querySelectorAll('g[id^="node_"]');
  nodes.forEach(node => {
      // Make nodes draggable
      makeNodeDraggable(node);
      
      node.style.cursor = 'pointer';
      
      // Add double-click event for collapsing/expanding
      node.addEventListener('dblclick', function(e) {
          e.stopPropagation();
          const isCollapsed = collapsedNodes.has(this.getAttribute('data-name'));
          toggleNodeCollapse(this, !isCollapsed);
      });
      
      // Add click event
      node.addEventListener('click', function(e) {
          e.stopPropagation();
          // Get the module name from this node
          const moduleName = this.getAttribute('data-name');
          
          if (!moduleName) {
              console.error('No data-name attribute found on node:', this);
              return;
          }
          
          // Copy to clipboard and handle callback properly
          navigator.clipboard.writeText(moduleName).then(function() {
              // Show notification
              copyNotification.textContent = `Module name "${moduleName}" copied to clipboard!`;
              copyNotification.style.opacity = 1;
              setTimeout(function() {
                  copyNotification.style.opacity = 0;
              }, 2000);
          }).catch(function(err) {
              console.error('Failed to copy module name: ', err);
          });
          
          // Display module information
          displayModuleInfo(moduleName);
          
          // Highlight the selected node
          nodes.forEach(n => n.classList.remove('highlight-node'));
          this.classList.add('highlight-node');
      });
      
      // Add context menu
      node.addEventListener('contextmenu', function(e) {
          e.preventDefault();
          const moduleName = this.getAttribute('data-name');
          
          if (!moduleName) {
              console.error('No data-name attribute found on node:', this);
              return;
          }
          
          // Show context menu
          nodeMenu.style.display = 'block';
          nodeMenu.style.left = (e.pageX) + 'px';
          nodeMenu.style.top = (e.pageY) + 'px';
          
          // Set up menu items
          document.getElementById('copyName').onclick = function(e) {
              e.stopPropagation();
              navigator.clipboard.writeText(moduleName);
              nodeMenu.style.display = 'none';
              copyNotification.textContent = `Module name "${moduleName}" copied to clipboard!`;
              copyNotification.style.opacity = 1;
              setTimeout(function() { 
                  copyNotification.style.opacity = 0; 
              }, 2000);
          };
          
          document.getElementById('showDetails').onclick = function(e) {
              e.stopPropagation();
              displayModuleInfo(moduleName);
              nodeMenu.style.display = 'none';
          };
          
          document.getElementById('highlightPath').onclick = function(e) {
              e.stopPropagation();
              // Highlight module and its parents
              const parts = moduleName.split('.');
              let path = '';
              nodes.forEach(n => n.classList.add('fade'));
              
              // Highlight each part of the path
              for(let i = 0; i < parts.length; i++) {
                  path = path ? path + '.' + parts[i] : parts[i];
                  const node = svg.querySelector(`[data-name="${path}"]`);
                  if(node) node.classList.remove('fade');
              }
              
              setTimeout(function() { 
                  nodes.forEach(n => n.classList.remove('fade')); 
              }, 3000);
              
              nodeMenu.style.display = 'none';
          };
      });
  });
  
  // Make nodes draggable
  function makeNodeDraggable(node) {
      let isDragging = false;
      let offsetX, offsetY;
      
      node.addEventListener('mousedown', function(e) {
          // Only enable dragging with Alt key pressed or middle mouse button
          if (e.altKey || e.button === 1) {
              e.stopPropagation();
              e.preventDefault();
              
              isDragging = true;
              
              // Get current transform or create a new one
              const transform = node.getAttribute('transform') || '';
              let translateX = 0, translateY = 0;
              
              // Extract existing translate values if they exist
              const match = transform.match(/translate\(([^,]+),([^)]+)\)/);
              if (match) {
                  translateX = parseFloat(match[1]);
                  translateY = parseFloat(match[2]);
              }
              
              // Calculate offset
              const svgRect = svg.getBoundingClientRect();
              const nodeRect = node.getBoundingClientRect();
              offsetX = (e.clientX - svgRect.left) - (nodeRect.left - svgRect.left + translateX);
              offsetY = (e.clientY - svgRect.top) - (nodeRect.top - svgRect.top + translateY);
              
              // Set cursor style
              node.style.cursor = 'grabbing';
              
              // Move the node to the front
              const parent = node.parentNode;
              parent.appendChild(node);
          }
      });
      
      svg.addEventListener('mousemove', function(e) {
          if (isDragging) {
              const svgRect = svg.getBoundingClientRect();
              const x = (e.clientX - svgRect.left) - offsetX;
              const y = (e.clientY - svgRect.top) - offsetY;
              
              // Apply the new transform
              node.setAttribute('transform', `translate(${x},${y})`);
              
              // Update edge positions if needed
              updateEdges(node);
          }
      });
      
      svg.addEventListener('mouseup', function() {
          if (isDragging) {
              isDragging = false;
              node.style.cursor = 'pointer';
          }
      });
      
      svg.addEventListener('mouseleave', function() {
          if (isDragging) {
              isDragging = false;
              node.style.cursor = 'pointer';
          }
      });
  }
  
  // Update the edges connected to the moved node
  function updateEdges(node) {
      // Find all edges connected to this node
      // This would depend on how edges are represented in your SVG
      // This is a simplified version that assumes edges are path elements with source/target markers
      const nodeId = node.id;
      const edges = svg.querySelectorAll('path.edge');
      edges.forEach(edge => {
          // Check if this edge is connected to the moved node
          // This would need to be adapted based on your specific SVG structure
          const edgeSource = edge.getAttribute('data-source');
          const edgeTarget = edge.getAttribute('data-target');
          
          if (edgeSource === nodeId || edgeTarget === nodeId) {
              // Update the edge - this would need specific implementation
              // based on how your edges are defined
              // For instance, if using Graphviz's edges, you might need to
              // recompute the path attributes
          }
      });
  }
  
  // Collapse/Expand All buttons
  expandAllBtn.addEventListener('click', function() {
      // Expand all nodes
      collapsedNodes.forEach(nodePath => {
          const node = svg.querySelector(`[data-name="${nodePath}"]`);
          if (node) toggleNodeCollapse(node, false);
      });
      collapsedNodes.clear();
      nodes.forEach(node => node.classList.remove('hidden-node'));
  });
  
  collapseAllBtn.addEventListener('click', function() {
      // Collapse top-level nodes (those with one segment in their path)
      const topLevelNodes = Array.from(nodes).filter(node => {
          const path = node.getAttribute('data-name');
          return path && !path.includes('.');
      });
      
      topLevelNodes.forEach(node => {
          toggleNodeCollapse(node, true);
      });
  });
  
  // Function to display module information
  function displayModuleInfo(moduleName) {
      const info = moduleInfo[moduleName];
      if (!info) {
          infoPanel.innerHTML = `<h3>Module: ${moduleName}</h3><p>No detailed information available</p>`;
          return;
      }
      
      let html = `<h3>Module: ${moduleName}</h3>`;
      html += `<div class="layer-path">${moduleName}</div>`;
      html += `<table>`;
      html += `<tr><th>Property</th><th>Value</th></tr>`;
      
      // Add basic properties
      html += `<tr><td>Type</td><td>${info.type}</td></tr>`;
      html += `<tr><td>Parameters</td><td>${info.parameters.toLocaleString()}</td></tr>`;
      html += `<tr><td>Trainable</td><td>${info.trainable ? 'Yes' : 'No'}</td></tr>`;
      
      // Add specific properties
      for (const [key, value] of Object.entries(info)) {
          if (!['type', 'parameters', 'trainable'].includes(key)) {
              html += `<tr><td>${key}</td><td>${JSON.stringify(value)}</td></tr>`;
          }
      }
      
      html += `</table>`;
      infoPanel.innerHTML = html;
  }
  
  // Search functionality
  searchInput.addEventListener('input', function() {
      const searchTerm = this.value.toLowerCase();
      if (searchTerm === '') {
          nodes.forEach(node => {
              node.style.opacity = 1;
          });
          return;
      }
      
      nodes.forEach(node => {
          const moduleName = node.getAttribute('data-name') || '';
          if (moduleName.toLowerCase().includes(searchTerm)) {
              node.style.opacity = 1;
          } else {
              node.style.opacity = 0.2;
          }
      });
  });
  
  clearSearch.addEventListener('click', function() {
      searchInput.value = '';
      nodes.forEach(node => {
          node.style.opacity = 1;
      });
  });
  
  // Add keyboard shortcuts
  document.addEventListener('keydown', function(e) {
      // Ctrl+F to focus search
      if (e.ctrlKey && e.key === 'f') {
          e.preventDefault();
          searchInput.focus();
      }
      
      // Esc to clear search
      if (e.key === 'Escape') {
          searchInput.value = '';
          nodes.forEach(node => {
              node.style.opacity = 1;
          });
      }
  });
  
  // Make the visualization container resizable
  const container = document.querySelector('.visualization-container');
  let startY, startHeight;
  
  function initResize(e) {
      startY = e.clientY;
      startHeight = parseInt(document.defaultView.getComputedStyle(container).height, 10);
      document.documentElement.addEventListener('mousemove', doResize, false);
      document.documentElement.addEventListener('mouseup', stopResize, false);
  }
  
  function doResize(e) {
      container.style.height = (startHeight + e.clientY - startY) + 'px';
  }
  
  function stopResize() {
      document.documentElement.removeEventListener('mousemove', doResize, false);
      document.documentElement.removeEventListener('mouseup', stopResize, false);
  }
  
  // Add a resize handle
  const resizeHandle = document.createElement('div');
  resizeHandle.style.cursor = 'ns-resize';
  resizeHandle.style.height = '10px';
  resizeHandle.style.backgroundColor = '#f0f0f0';
  resizeHandle.style.borderTop = '1px solid #ccc';
  resizeHandle.style.marginBottom = '10px';
  container.after(resizeHandle);
  
  resizeHandle.addEventListener('mousedown', initResize, false);
  
  // Add help tooltip
  const helpText = document.createElement('div');
  helpText.className = 'help-tooltip';
  helpText.innerHTML = `
    <h3>Keyboard Shortcuts</h3>
    <ul>
      <li><b>Double-click</b>: Collapse/expand node</li>
      <li><b>Alt + Drag</b>: Move node</li>
      <li><b>Ctrl+F</b>: Search</li>
      <li><b>Esc</b>: Clear search</li>
    </ul>
  `;
  document.body.appendChild(helpText);
  
  // Show help on '?' key
  document.addEventListener('keydown', function(e) {
      if (e.key === '?' || (e.ctrlKey && e.key === 'h')) {
          e.preventDefault();
          helpText.style.display = helpText.style.display === 'block' ? 'none' : 'block';
      }
  });
});