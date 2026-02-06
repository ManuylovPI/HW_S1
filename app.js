// app.js - Client-Side Sentiment Analyzer

// Import Transformers.js as ES module
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// DOM Elements
const tokenInput = document.getElementById('tokenInput');
const saveTokenBtn = document.getElementById('saveTokenBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const reviewContent = document.getElementById('reviewContent');
const reviewCount = document.getElementById('reviewCount');
const sentimentDisplay = document.getElementById('sentimentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const sentimentIcon = document.getElementById('sentimentIcon');
const status = document.getElementById('status');

// Application State
let reviews = [];
let classifier = null;
let modelLoaded = false;
let reviewsLoaded = false;

// TSV File URL
const TSV_URL = 'https://raw.githubusercontent.com/dylan-jiang/amazon-reviews-sentiment-analysis/master/reviews_test.tsv';

// Initialize the application
async function init() {
    status.textContent = 'Initializing...';
    
    // Load saved token if exists
    const savedToken = localStorage.getItem('hf_token');
    if (savedToken) {
        tokenInput.value = savedToken;
        env.remoteToken = savedToken;
        status.textContent = 'Using saved token for model download...';
    }
    
    // Set up event listeners
    saveTokenBtn.addEventListener('click', saveToken);
    analyzeBtn.addEventListener('click', analyzeRandomReview);
    
    // Load reviews and model in parallel
    await Promise.all([loadReviews(), loadModel()]);
    
    if (reviewsLoaded && modelLoaded) {
        status.textContent = 'Ready to analyze reviews!';
        status.className = 'status status-ready';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
    }
}

// Save token to localStorage
function saveToken() {
    const token = tokenInput.value.trim();
    if (token) {
        localStorage.setItem('hf_token', token);
        env.remoteToken = token;
        status.textContent = 'Token saved successfully!';
        
        // If model isn't loaded yet, try to reload with token
        if (!modelLoaded) {
            status.textContent = 'Token saved. Reloading model with authentication...';
            loadModel();
        }
    } else {
        localStorage.removeItem('hf_token');
        env.remoteToken = null;
        status.textContent = 'Token cleared. Using anonymous model download.';
    }
}

// Load and parse TSV file
async function loadReviews() {
    try {
        status.textContent = 'Loading reviews dataset...';
        
        const response = await fetch(TSV_URL);
        const tsvData = await response.text();
        
        // Parse TSV with PapaParse
        const results = Papa.parse(tsvData, {
            header: true,
            delimiter: '\t',
            skipEmptyLines: true
        });
        
        // Extract review texts from summary or text columns
        reviews = results.data
            .map(row => {
                // Use summary if it has content, otherwise use text
                const reviewText = (row.summary && row.summary.trim()) || (row.text && row.text.trim());
                return reviewText;
            })
            .filter(text => text && text.trim().length > 0);
        
        reviewsLoaded = true;
        console.log(`Loaded ${reviews.length} reviews`);
        reviewCount.textContent = `${reviews.length} reviews loaded`;
        reviewCount.style.display = 'block';
        
        if (reviews.length === 0) {
            throw new Error('No reviews loaded from dataset');
        }
        
        return true;
    } catch (error) {
        console.error('Error loading reviews:', error);
        status.textContent = `Error loading reviews: ${error.message}`;
        status.className = 'status status-error';
        reviewsLoaded = false;
        return false;
    }
}

// Load sentiment analysis model
async function loadModel() {
    try {
        status.textContent = 'Downloading sentiment model...';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Model...';
        
        // Use a verified working model for Transformers.js
        classifier = await pipeline(
            'text-classification', 
            'Xenova/bert-base-multilingual-uncased-sentiment',
            { progress_callback: onModelProgress }
        );
        
        modelLoaded = true;
        console.log('Model loaded successfully');
        
        if (reviewsLoaded) {
            status.textContent = 'Model ready!';
            status.className = 'status status-ready';
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
        } else {
            status.textContent = 'Model loaded, waiting for reviews...';
        }
        
        return true;
    } catch (error) {
        console.error('Error loading model:', error);
        
        // Try fallback model
        status.textContent = 'Trying fallback model...';
        try {
            classifier = await pipeline(
                'text-classification',
                'Xenova/distilbert-base-uncased-emotion',
                { progress_callback: onModelProgress }
            );
            
            modelLoaded = true;
            console.log('Fallback model loaded successfully');
            status.textContent = 'Fallback model loaded!';
            
            if (reviewsLoaded) {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
            }
            
            return true;
        } catch (fallbackError) {
            console.error('Error loading fallback model:', fallbackError);
            status.textContent = `Failed to load model: ${fallbackError.message}`;
            status.className = 'status status-error';
            modelLoaded = false;
            return false;
        }
    }
}

// Model download progress callback
function onModelProgress(data) {
    if (data.status === 'downloading') {
        const percent = Math.round((data.loaded / data.total) * 100);
        status.textContent = `Downloading model: ${percent}%`;
        status.className = 'status status-loading';
    }
}

// Analyze a random review
async function analyzeRandomReview() {
    if (!modelLoaded || !reviewsLoaded || reviews.length === 0) {
        status.textContent = 'Model or reviews not loaded yet. Please wait.';
        status.className = 'status status-error';
        return;
    }
    
    // Disable button during analysis
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    status.textContent = 'Analyzing sentiment...';
    status.className = 'status status-loading';
    
    // Select random review
    const randomIndex = Math.floor(Math.random() * reviews.length);
    const reviewText = reviews[randomIndex];
    
    // Display the selected review
    reviewContent.textContent = reviewText;
    reviewContent.className = 'review-content';
    
    try {
        // Run sentiment analysis locally in browser
        const result = await classifier(reviewText);
        
        // Process results
        let sentimentLabel = result[0].label;
        let confidence = result[0].score;
        
        // Map model output to simplified sentiment
        let simplifiedSentiment, iconClass, displayClass;
        
        if (sentimentLabel.includes('5 star') || sentimentLabel.includes('5 stars') || 
            sentimentLabel.includes('4 star') || sentimentLabel.includes('4 stars') ||
            sentimentLabel === 'positive' || sentimentLabel === 'joy') {
            simplifiedSentiment = 'POSITIVE';
            iconClass = 'icon-positive';
            displayClass = 'sentiment-positive';
            sentimentIcon.className = 'fas fa-thumbs-up';
        } else if (sentimentLabel.includes('1 star') || sentimentLabel.includes('1 stars') || 
                   sentimentLabel.includes('2 star') || sentimentLabel.includes('2 stars') ||
                   sentimentLabel === 'negative' || sentimentLabel === 'sadness' || 
                   sentimentLabel === 'anger' || sentimentLabel === 'fear') {
            simplifiedSentiment = 'NEGATIVE';
            iconClass = 'icon-negative';
            displayClass = 'sentiment-negative';
            sentimentIcon.className = 'fas fa-thumbs-down';
        } else {
            simplifiedSentiment = 'NEUTRAL';
            iconClass = 'icon-neutral';
            displayClass = 'sentiment-neutral';
            sentimentIcon.className = 'fas fa-meh';
        }
        
        // Update UI with results
        sentimentDisplay.textContent = simplifiedSentiment;
        sentimentDisplay.className = `sentiment-display ${displayClass}`;
        
        confidenceDisplay.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
        sentimentIcon.className = `fas ${sentimentIcon.className.split(' ')[1]} ${iconClass}`;
        
        // Update status
        status.textContent = 'Analysis complete!';
        status.className = 'status status-ready';
        
    } catch (error) {
        console.error('Error during analysis:', error);
        sentimentDisplay.textContent = 'ERROR';
        sentimentDisplay.className = 'sentiment-display sentiment-negative';
        confidenceDisplay.textContent = 'Failed to analyze review';
        sentimentIcon.className = 'fas fa-exclamation-triangle icon-negative';
        
        status.textContent = `Analysis error: ${error.message}`;
        status.className = 'status status-error';
    } finally {
        // Re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Export for potential debugging
window.app = {
    reviews,
    classifier,
    modelLoaded,
    reviewsLoaded,
    analyzeRandomReview,
    loadModel,
    loadReviews
};