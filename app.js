// app.js - Client-side Sentiment Analysis Application

// DOM Elements
const tokenInput = document.getElementById('token-input');
const saveTokenBtn = document.getElementById('save-token');
const analyzeBtn = document.getElementById('analyze-btn');
const reviewDisplay = document.getElementById('review-display');
const resultArea = document.getElementById('result-area');
const statusDiv = document.getElementById('status');

// Application State
let sentimentPipeline = null;
let reviews = [];
let modelReady = false;
let isAnalyzing = false;

// Initialize the application
async function initApp() {
    showStatus('Loading reviews dataset...', 'loading');
    
    // Load saved token if available
    const savedToken = localStorage.getItem('hf_token');
    if (savedToken) {
        tokenInput.value = savedToken;
        showStatus('Using saved token for model download', 'success');
    }
    
    // Load reviews from TSV file
    try {
        await loadReviews();
        showStatus(`Loaded ${reviews.length} reviews successfully`, 'success');
    } catch (error) {
        showStatus(`Error loading reviews: ${error.message}`, 'error');
        return;
    }
    
    // Initialize the sentiment analysis model
    await initModel();
}

// Load reviews from TSV file
async function loadReviews() {
    return new Promise((resolve, reject) => {
        Papa.parse('https://raw.githubusercontent.com/curiousily/Getting-Things-Done-with-Pytorch/main/data/reviews_test.tsv', {
            download: true,
            delimiter: '\t',
            header: true,
            complete: (results) => {
                if (results.data && results.data.length > 0) {
                    // Extract review text from the 'text' column
                    reviews = results.data
                        .map(row => row.text)
                        .filter(text => text && text.trim().length > 0);
                    
                    if (reviews.length > 0) {
                        resolve();
                    } else {
                        reject(new Error('No valid reviews found in the dataset'));
                    }
                } else {
                    reject(new Error('Failed to parse reviews data'));
                }
            },
            error: (error) => {
                reject(new Error(`Failed to load reviews: ${error.message}`));
            }
        });
    });
}

// Initialize the sentiment analysis model
async function initModel() {
    showStatus('Downloading sentiment model... This may take a moment.', 'loading');
    
    try {
        // Dynamically import Transformers.js
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0');
        
        // Configure Transformers.js with token if available
        const token = localStorage.getItem('hf_token');
        const config = token ? { token } : {};
        
        // Create the sentiment analysis pipeline
        sentimentPipeline = await pipeline('text-classification', 
            'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
            config
        );
        
        modelReady = true;
        showStatus('Model ready! You can now analyze reviews.', 'success');
    } catch (error) {
        showStatus(`Error loading model: ${error.message}`, 'error');
        console.error('Model initialization error:', error);
    }
}

// Save token to localStorage
saveTokenBtn.addEventListener('click', () => {
    const token = tokenInput.value.trim();
    
    if (token) {
        localStorage.setItem('hf_token', token);
        showStatus('Token saved successfully', 'success');
        
        // If model isn't ready yet, reinitialize with the new token
        if (!modelReady) {
            initModel();
        }
    } else {
        localStorage.removeItem('hf_token');
        showStatus('Token cleared', 'success');
    }
});

// Analyze a random review
analyzeBtn.addEventListener('click', async () => {
    if (!modelReady) {
        showStatus('Model is still loading. Please wait...', 'error');
        return;
    }
    
    if (isAnalyzing) {
        return;
    }
    
    if (reviews.length === 0) {
        showStatus('No reviews available to analyze', 'error');
        return;
    }
    
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    // Select a random review
    const randomIndex = Math.floor(Math.random() * reviews.length);
    const review = reviews[randomIndex];
    
    // Display the selected review
    reviewDisplay.textContent = review;
    
    // Show analyzing status
    showStatus('Analyzing sentiment...', 'loading');
    
    try {
        // Run sentiment analysis locally in the browser
        const result = await sentimentPipeline(review);
        
        if (result && result.length > 0) {
            const sentiment = result[0];
            displaySentiment(sentiment.label, sentiment.score);
            showStatus('Analysis complete!', 'success');
        } else {
            throw new Error('No sentiment result returned');
        }
    } catch (error) {
        showStatus(`Analysis error: ${error.message}`, 'error');
        displaySentiment('ERROR', 0);
    } finally {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Random Review';
    }
});

// Display sentiment result
function displaySentiment(label, confidence) {
    const icon = resultArea.querySelector('.sentiment-icon i');
    const sentimentText = resultArea.querySelector('.sentiment-text');
    const confidenceText = resultArea.querySelector('.confidence');
    
    // Map labels to our sentiment categories
    let sentiment, displayLabel;
    
    if (label.includes('POSITIVE') || label === 'POSITIVE') {
        sentiment = 'positive';
        displayLabel = 'POSITIVE';
        icon.className = 'fas fa-thumbs-up positive';
    } else if (label.includes('NEGATIVE') || label === 'NEGATIVE') {
        sentiment = 'negative';
        displayLabel = 'NEGATIVE';
        icon.className = 'fas fa-thumbs-down negative';
    } else {
        sentiment = 'neutral';
        displayLabel = 'NEUTRAL';
        icon.className = 'fas fa-question-circle neutral';
    }
    
    // Update UI
    sentimentText.textContent = displayLabel;
    sentimentText.className = `sentiment-text ${sentiment}`;
    
    const confidencePercent = (confidence * 100).toFixed(1);
    confidenceText.textContent = `${confidencePercent}% confidence`;
}

// Show status messages
function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.style.display = 'block';
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            if (statusDiv.textContent === message) {
                statusDiv.style.display = 'none';
            }
        }, 3000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', initApp);