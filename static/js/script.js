// Global variables
let downloadUrl = '';

// Page Navigation Functions
function showHomePage() {
    hideAllPages();
    document.getElementById('home-page').classList.remove('hidden');
}

function showSimplePredictor() {
    hideAllPages();
    document.getElementById('simple-page').classList.remove('hidden');
}

function showAdvancedPredictor() {
    hideAllPages();
    document.getElementById('advanced-page').classList.remove('hidden');
}

function hideAllPages() {
    document.getElementById('home-page').classList.add('hidden');
    document.getElementById('simple-page').classList.add('hidden');
    document.getElementById('advanced-page').classList.add('hidden');
}

// Simple Predictor Functions
function predictSimple(event) {
    event.preventDefault();
    
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const originalText = btnText.innerHTML;
    
    // Show loading state
    btnText.innerHTML = 'Analyzing...';
    btnText.classList.add('loading');
    submitBtn.disabled = true;

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    console.log('🔄 Sending prediction request...', data);

    // Make API call to Flask backend
    fetch('/predict_simple', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log('📊 Prediction response:', result);
        
        if (result.success) {
            const price = Math.round(result.predicted_price);
            document.getElementById('predicted-price').textContent = price.toLocaleString();
            document.getElementById('simple-result').classList.remove('hidden');
            document.getElementById('simple-result').scrollIntoView({ 
                behavior: 'smooth',
                block: 'center' 
            });
            
            // Show success message
            showNotification('✅ Prediction completed successfully!', 'success');
        } else {
            showNotification('❌ Error: ' + result.error, 'error');
            console.error('Prediction error:', result.error);
        }
    })
    .catch(error => {
        console.error('🚨 Network error:', error);
        showNotification('🚨 Network error: Unable to connect to server', 'error');
    })
    .finally(() => {
        // Reset button state
        btnText.innerHTML = originalText;
        btnText.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

function resetSimpleForm() {
    document.getElementById('simple-form').reset();
    document.getElementById('simple-result').classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    hideNotification();
}

// Advanced Predictor Functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const fileName = file.name;
        const fileSize = formatFileSize(file.size);
        
        document.getElementById('file-name').textContent = fileName;
        document.getElementById('file-size').textContent = fileSize;
        document.getElementById('file-info').classList.remove('hidden');
        document.getElementById('upload-btn').disabled = false;
        
        // Validate file type
        if (!fileName.toLowerCase().endsWith('.xlsx') && !fileName.toLowerCase().endsWith('.xls')) {
            showNotification('⚠️ Please select a valid Excel file (.xlsx or .xls)', 'error');
            resetFileSelection();
            return;
        }
        
        // Validate file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            showNotification('⚠️ File size exceeds 16MB limit', 'error');
            resetFileSelection();
            return;
        }
        
        showNotification('✅ File selected successfully', 'success');
    }
}

function resetFileSelection() {
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('upload-btn').disabled = true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function predictAdvanced(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById('upload-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const originalText = btnText.innerHTML;
    
    // Show loading state
    btnText.innerHTML = 'Processing...';
    btnText.classList.add('loading');
    submitBtn.disabled = true;

    const formData = new FormData(event.target);
    
    console.log('📤 Uploading file for batch prediction...');

    fetch('/predict_advanced', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(result => {
        console.log('📊 Batch prediction response:', result);
        
        if (result.success) {
            downloadUrl = result.download_url;
            
            // Update result message with details
            const resultMessage = `Successfully processed ${result.total_properties || 'your'} properties. Average predicted value: $${result.avg_prediction ? Math.round(result.avg_prediction).toLocaleString() : 'N/A'}`;
            document.getElementById('result-message').textContent = resultMessage;
            
            document.getElementById('advanced-result').classList.remove('hidden');
            document.getElementById('advanced-result').scrollIntoView({ 
                behavior: 'smooth',
                block: 'center' 
            });
            
            showNotification('✅ File processed successfully!', 'success');
        } else {
            showNotification('❌ Error: ' + result.error, 'error');
            console.error('Processing error:', result.error);
        }
    })
    .catch(error => {
        console.error('🚨 Upload error:', error);
        showNotification('🚨 Network error: Unable to upload file', 'error');
    })
    .finally(() => {
        // Reset button state
        btnText.innerHTML = originalText;
        btnText.classList.remove('loading');
        submitBtn.disabled = false;
    });
}

function resetAdvancedForm() {
    document.getElementById('advanced-form').reset();
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('upload-btn').disabled = true;
    document.getElementById('advanced-result').classList.add('hidden');
    downloadUrl = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    hideNotification();
}

function downloadResults() {
    if (downloadUrl) {
        console.log('📥 Downloading results...', downloadUrl);
        window.location.href = downloadUrl;
        showNotification('📥 Download started...', 'success');
    } else {
        showNotification('❌ No file available for download', 'error');
    }
}

// Drag and Drop Functionality
function setupDragAndDrop() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (!uploadArea || !fileInput) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);

    // Handle click to open file dialog
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const file = files[0];
            if (file.name.toLowerCase().endsWith('.xlsx') || file.name.toLowerCase().endsWith('.xls')) {
                fileInput.files = files;
                handleFileSelect({ target: { files: files } });
            } else {
                showNotification('⚠️ Please drop a valid Excel file (.xlsx or .xls)', 'error');
            }
        }
    }
}

// Notification System
function showNotification(message, type = 'info') {
    // Remove existing notification
    hideNotification();
    
    const notification = document.createElement('div');
    notification.id = 'notification';
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        font-weight: 500;
        z-index: 1000;
        max-width: 400px;
        word-wrap: break-word;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Apply type-specific styles
    if (type === 'success') {
        notification.style.backgroundColor = 'rgba(16, 185, 129, 0.15)';
        notification.style.border = '1px solid rgba(16, 185, 129, 0.3)';
        notification.style.color = '#6ee7b7';
    } else if (type === 'error') {
        notification.style.backgroundColor = 'rgba(220, 38, 38, 0.15)';
        notification.style.border = '1px solid rgba(220, 38, 38, 0.3)';
        notification.style.color = '#fca5a5';
    } else {
        notification.style.backgroundColor = 'rgba(99, 102, 241, 0.15)';
        notification.style.border = '1px solid rgba(99, 102, 241, 0.3)';
        notification.style.color = '#c7d2fe';
    }
    
    document.body.appendChild(notification);
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        hideNotification();
    }, 5000);
    
    // Allow manual close
    notification.addEventListener('click', hideNotification);
}

function hideNotification() {
    const notification = document.getElementById('notification');
    if (notification) {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }
}

// Ripple Effect for Buttons
function addRippleEffect() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Don't add ripple to disabled buttons
            if (this.disabled) return;
            
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            this.style.position = 'relative';
            this.style.overflow = 'hidden';
            this.appendChild(ripple);
            
            setTimeout(() => {
                if (ripple.parentNode) {
                    ripple.remove();
                }
            }, 600);
        });
    });
}

// Form Validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;
    
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.style.borderColor = '#dc2626';
            isValid = false;
        } else {
            field.style.borderColor = '';
        }
    });
    
    return isValid;
}

// Add CSS animations
function addAnimations() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(2);
                opacity: 0;
            }
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-out;
        }
    `;
    document.head.appendChild(style);
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 Housing Price Predictor initialized');
    
    // Setup all functionality
    setupDragAndDrop();
    addRippleEffect();
    addAnimations();
    
    // Add fade-in animation to page elements
    const pageElements = document.querySelectorAll('.option-card, .form-container');
    pageElements.forEach((element, index) => {
        setTimeout(() => {
            element.classList.add('fade-in');
        }, index * 100);
    });
    
    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add form validation on input
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.hasAttribute('required') && !this.value.trim()) {
                    this.style.borderColor = '#dc2626';
                } else {
                    this.style.borderColor = '';
                }
            });
            
            input.addEventListener('input', function() {
                if (this.style.borderColor === 'rgb(220, 38, 38)') {
                    this.style.borderColor = '';
                }
            });
        });
    });
    
    console.log('✅ All features initialized successfully');
});