// Setup interactive behavior
document.addEventListener('DOMContentLoaded', function() {
  const svg = document.querySelector('svg');
  const infoPanel = document.getElementById('infoPanel');
  const copyNotification = document.getElementById('copyNotification');
  const searchInput = document.getElementById('searchInput');
  const clearSearch = document.getElementById('clearSearch');
  const nodeMenu = document.getElementById('nodeMenu');
  const exportJSONBtn = document.getElementById('exportJSON');
  
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
  
  // Add click handlers to nodes
  const nodes = svg.querySelectorAll('[id^="node_"]');
  nodes.forEach(node => {
      node.style.cursor = 'pointer';
      
      // Add click event
      node.addEventListener('click', function(e) {
          e.stopPropagation();
          // Get the module name from this node
          const moduleName = this.getAttribute('data-name');
          
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
                  const node = document.querySelector(`[data-name="${path}"]`);
                  if(node) node.classList.remove('fade');
              }
              
              setTimeout(function() { 
                  nodes.forEach(n => n.classList.remove('fade')); 
              }, 3000);
              
              nodeMenu.style.display = 'none';
          };
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
          const moduleName = node.getAttribute('data-name').toLowerCase();
          if (moduleName.includes(searchTerm)) {
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
});