// app/static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize processLogger
    processLogger.init();
    
    // Helper function to show loader
    window.showLoader = function(message, initialLog) {
        processLogger.show(message, initialLog);
    };
    
    // Helper function to hide loader
    window.hideLoader = function() {
        processLogger.hide();
    };
    
    // Helper function to add log entry
    window.addLog = function(message) {
        processLogger.addLog(message);
    };
    
    // Form submission handling with loader
    const forms = document.querySelectorAll('form[data-show-loader="true"]');
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const message = this.dataset.loaderMessage || 'Processing...';
            const initialLog = this.dataset.initialLog || 'Starting process...';
            showLoader(message, initialLog);
        });
    });
    
    // Tooltips initialization
    const tooltipTriggers = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggers.length > 0) {
        Array.from(tooltipTriggers).forEach(tooltipTriggerEl => {
            new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Table sorting functionality
    const sortableTables = document.querySelectorAll('.sortable');
    sortableTables.forEach(table => {
        const headers = table.querySelectorAll('th[data-sort]');
        
        headers.forEach(header => {
            header.addEventListener('click', function() {
                const sortKey = this.dataset.sort;
                const sortDirection = this.dataset.sortDirection === 'asc' ? 'desc' : 'asc';
                
                // Update header state
                headers.forEach(h => {
                    h.dataset.sortDirection = '';
                    h.classList.remove('sorting-asc', 'sorting-desc');
                });
                
                this.dataset.sortDirection = sortDirection;
                this.classList.add(sortDirection === 'asc' ? 'sorting-asc' : 'sorting-desc');
                
                // Get table rows
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                // Sort rows
                rows.sort((a, b) => {
                    const aValue = a.querySelector(`td[data-field="${sortKey}"]`).textContent.trim();
                    const bValue = b.querySelector(`td[data-field="${sortKey}"]`).textContent.trim();
                    
                    // Check if values are numbers
                    const aNum = parseFloat(aValue);
                    const bNum = parseFloat(bValue);
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {
                        return sortDirection === 'asc' ? aNum - bNum : bNum - aNum;
                    }
                    
                    // String comparison
                    return sortDirection === 'asc' ? 
                        aValue.localeCompare(bValue) : 
                        bValue.localeCompare(aValue);
                });
                
                // Append sorted rows
                rows.forEach(row => tbody.appendChild(row));
            });
        });
    });
    
    // File upload preview
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileNameElement = document.getElementById(this.dataset.filenameTarget);
            if (!fileNameElement) return;
            
            if (this.files && this.files.length > 0) {
                const file = this.files[0];
                
                // Format file size
                const sizeInKB = file.size / 1024;
                let fileSize;
                
                if (sizeInKB < 1024) {
                    fileSize = sizeInKB.toFixed(2) + ' KB';
                } else {
                    fileSize = (sizeInKB / 1024).toFixed(2) + ' MB';
                }
                
                // Update filename display
                fileNameElement.textContent = file.name + ' (' + fileSize + ')';
                
                // Show the preview element if specified
                const previewElement = document.getElementById(this.dataset.previewTarget);
                if (previewElement) {
                    previewElement.classList.remove('d-none');
                }
            }
        });
    });
    
    // Sidebar scrolling
    const stickyElements = document.querySelectorAll('.sticky-top');
    if (stickyElements.length > 0) {
        const navbar = document.querySelector('.navbar');
        const navbarHeight = navbar ? navbar.offsetHeight : 0;
        
        stickyElements.forEach(element => {
            const currentTop = element.style.top;
            const topValue = currentTop ? parseInt(currentTop) : 0;
            element.style.top = (navbarHeight + topValue) + 'px';
        });
    }
    
    // Auto-disable checkboxes when related select changes
    const relatedSelects = document.querySelectorAll('select[data-related-checkboxes]');
    relatedSelects.forEach(select => {
        select.addEventListener('change', function() {
            const selectedValue = this.value;
            const checkboxSelector = this.dataset.relatedCheckboxes;
            const checkboxes = document.querySelectorAll(checkboxSelector);
            
            checkboxes.forEach(checkbox => {
                if (checkbox.value === selectedValue) {
                    checkbox.checked = false;
                    checkbox.disabled = true;
                } else {
                    checkbox.disabled = false;
                }
            });
        });
        
        // Trigger on load
        if (select.value) {
            const event = new Event('change');
            select.dispatchEvent(event);
        }
    });
    
    // Feature column selection toggle all
    const toggleAllFeatures = document.querySelector('.toggle-all-features');
    if (toggleAllFeatures) {
        toggleAllFeatures.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetSelector = this.dataset.target;
            const checkboxes = document.querySelectorAll(targetSelector);
            const targetState = this.dataset.state === 'select' ? true : false;
            
            checkboxes.forEach(checkbox => {
                if (!checkbox.disabled) {
                    checkbox.checked = targetState;
                }
            });
            
            // Toggle button state
            this.dataset.state = targetState ? 'deselect' : 'select';
            this.textContent = targetState ? 'Deselect All' : 'Select All';
        });
    }
});