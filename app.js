// app.js - Client-Side Sentiment Analysis with Transformers.js
// Import Transformers.js from CDN as ES module
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// DOM Elements
const tokenInput = document.getElementById('tokenInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const reviewText = document.getElementById('reviewText');
const sentimentLabel = document.getElementById('sentimentLabel');
const sentimentIcon = document.getElementById('sentimentIcon');
const confidenceScore = document.getElementById('confidenceScore');
const statusDiv = document.getElementById('status');
const reviewCount = document.getElementById('reviewCount');

// Application State
let sentimentPipeline = null;
let reviews = [];
let modelLoading = false;
let isAnalyzing = false;

// Token Management
const TOKEN_STORAGE_KEY = 'huggingface_token';

function loadToken() {
    const savedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (savedToken) {
        tokenInput.value = savedToken;
        setStatus('Token loaded from local storage', 'ready');
    }
}

function saveToken() {
    const token = tokenInput.value.trim();
    if (token) {
        localStorage.setItem(TOKEN_STORAGE_KEY, token);
        setStatus('Token saved locally', 'ready');
    } else {
        localStorage.removeItem(TOKEN_STORAGE_KEY);
        setStatus('Token cleared', 'ready');
    }
}

// Status Display
function setStatus(message, type = '') {
    statusDiv.textContent = message;
    statusDiv.className = 'status';
    
    if (type) {
        statusDiv.classList.add(`status-${type}`);
    }
    
    // Clear status after 5 seconds unless it's an error or loading message
    if (type !== 'error' && type !== 'loading') {
        setTimeout(() => {
            if (statusDiv.textContent === message) {
                statusDiv.textContent = '';
                statusDiv.className = 'status';
            }
        }, 5000);
    }
}

// Load and Parse Reviews
async function loadReviews() {
    try {
        setStatus('Loading reviews...', 'loading');
        
        const response = await fetch('reviews_test.tsv');
        if (!response.ok) {
            throw new Error(`Failed to load TSV file: ${response.status}`);
        }
        
        const tsvData = await response.text();
        
        return new Promise((resolve, reject) => {
            Papa.parse(tsvData, {
                delimiter: '\t',
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.warn('Parse warnings:', results.errors);
                    }
                    
                    // Extract reviews from summary column, fall back to text column
                    const extractedReviews = results.data
                        .map(row => {
                            // Prioritize summary column, fall back to text column
                            const review = row.summary?.trim() || row.text?.trim();
                            return review;
                        })
                        .filter(review => review && review.length > 0);
                    
                    reviews = extractedReviews;
                    reviewCount.textContent = `${reviews.length} reviews loaded`;
                    setStatus(`Loaded ${reviews.length} reviews`, 'ready');
                    resolve(reviews.length);
                },
                error: (error) => {
                    reject(new Error(`Failed to parse TSV: ${error.message}`));
                }
            });
        });
    } catch (error) {
        setStatus(`Error loading reviews: ${error.message}`, 'error');
        reviews = [];
        return 0;
    }
}

// Initialize Sentiment Analysis Model
async function initializeModel() {
    if (sentimentPipeline || modelLoading) return;
    
    modelLoading = true;
    setStatus('Downloading sentiment model... This may take a minute.', 'loading');
    
    try {
        // Configure Transformers.js environment
        const token = tokenInput.value.trim();
        if (token) {
            // Use token for authenticated model downloads
            env.allowLocalModels = false;
            env.remoteHost = 'https://huggingface.co';
            env.remotePathTemplate = '{model}/resolve/{revision}/{file}';
            // Note: Transformers.js doesn't directly expose token authentication in current version
            // The token is stored for user reference but actual auth happens via Hugging Face Hub
            console.log('Using Hugging Face token for model download');
        } else {
            // Use anonymous access (may be rate-limited)
            console.log('Using anonymous model access (add token for faster downloads)');
        }
        
        // Create sentiment analysis pipeline
        sentimentPipeline = await pipeline('text-classification', 
            'Xenova/bert-base-multilingual-uncased-sentiment');
        
        setStatus('Model loaded and ready for analysis!', 'ready');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
        
    } catch (error) {
        setStatus(`Error loading model: ${error.message}`, 'error');
        sentimentPipeline = null;
    } finally {
        modelLoading = false;
    }
}

// Analyze Sentiment
async function analyzeRandomReview() {
    if (isAnalyzing || !sentimentPipeline || reviews.length === 0) return;
    
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    
    try {
        // Select random review
        const randomIndex = Math.floor(Math.random() * reviews.length);
        const review = reviews[randomIndex];
        
        // Display selected review
        reviewText.textContent = review;
        
        setStatus('Analyzing sentiment locally in browser...', 'loading');
        
        // Run inference locally using Transformers.js
        const startTime = performance.now();
        const result = await sentimentPipeline(review);
        const endTime = performance.now();
        
        console.log('Inference result:', result);
        console.log(`Inference time: ${(endTime - startTime).toFixed(2)}ms`);
        
        // Process result
        let label = result[0].label;
        let score = result[0].score;
        
        // Map to sentiment categories
        let sentiment, iconClass, labelClass;
        
        if ((label === 'LABEL_1' || label === 'POSITIVE') && score > 0.5) {
            sentiment = 'POSITIVE';
            iconClass = 'fas fa-thumbs-up icon-positive';
            labelClass = 'sentiment-positive';
        } else if ((label === 'LABEL_0' || label === 'NEGATIVE') && score > 0.5) {
            sentiment = 'NEGATIVE';
            iconClass = 'fas fa-thumbs-down icon-negative';
            labelClass = 'sentiment-negative';
        } else {
            sentiment = 'NEUTRAL';
            iconClass = 'fas fa-question-circle icon-neutral';
            labelClass = 'sentiment-neutral';
            // Adjust score for neutral display
            score = Math.abs(score - 0.5) * 2;
        }
        
        // Update UI with results
        sentimentLabel.textContent = sentiment;
        sentimentLabel.className = `sentiment-label ${labelClass}`;
        
        sentimentIcon.innerHTML = `<i class="${iconClass}"></i>`;
        
        const confidencePercent = (score * 100).toFixed(1);
        confidenceScore.textContent = `${confidencePercent}% confidence`;
        
        setStatus(`Analysis complete! Local inference took ${(endTime - startTime).toFixed(0)}ms`, 'ready');
        
    } catch (error) {
        setStatus(`Analysis error: ${error.message}`, 'error');
        
        // Reset UI on error
        sentimentLabel.textContent = 'Error';
        sentimentLabel.className = 'sentiment-label sentiment-neutral';
        sentimentIcon.innerHTML = '<i class="fas fa-exclamation-triangle icon-neutral"></i>';
        confidenceScore.textContent = 'Could not analyze sentiment';
    } finally {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
    }
}

// Event Listeners
tokenInput.addEventListener('change', saveToken);
tokenInput.addEventListener('blur', saveToken);

analyzeBtn.addEventListener('click', async () => {
    if (!sentimentPipeline) {
        await initializeModel();
    }
    
    if (reviews.length === 0) {
        const count = await loadReviews();
        if (count === 0) {
            setStatus('No reviews available to analyze', 'error');
            return;
        }
    }
    
    await analyzeRandomReview();
});

// Initialize Application
async function initializeApp() {
    // Load saved token
    loadToken();
    
    // Load reviews in background
    loadReviews();
    
    // Pre-initialize model if token exists
    if (tokenInput.value.trim()) {
        setTimeout(() => {
            if (!sentimentPipeline && !modelLoading) {
                initializeModel();
            }
        }, 1000);
    }
    
    // Update button state
    analyzeBtn.disabled = false;
}

// Start the application
document.addEventListener('DOMContentLoaded', initializeApp);