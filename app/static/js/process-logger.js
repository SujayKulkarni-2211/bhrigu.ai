// app/static/js/process-logger.js

// Process logger for displaying progress with logs
const processLogger = {
    loaderOverlay: null,
    loaderText: null,
    loaderLogs: null,
    
    init: function() {
        this.loaderOverlay = document.getElementById('loaderOverlay');
        this.loaderText = document.getElementById('loaderText');
        this.loaderLogs = document.getElementById('loaderLogs');
    },
    
    show: function(message, initialLog = '') {
        if (!this.loaderOverlay) this.init();
        
        this.loaderText.textContent = message || 'Processing...';
        if (initialLog) this.addLog(initialLog);
        
        this.loaderOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    },
    
    hide: function() {
        if (!this.loaderOverlay) return;
        
        this.loaderOverlay.style.display = 'none';
        document.body.style.overflow = '';
        this.clearLogs();
    },
    
    addLog: function(message) {
        if (!this.loaderLogs) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
        
        this.loaderLogs.appendChild(logEntry);
        this.loaderLogs.scrollTop = this.loaderLogs.scrollHeight;
    },
    
    updateMessage: function(message) {
        if (!this.loaderText) return;
        this.loaderText.textContent = message;
    },
    
    clearLogs: function() {
        if (!this.loaderLogs) return;
        this.loaderLogs.innerHTML = '';
    }
};