// Import Transformers.js as ES module
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// Application state
let reviews = [];
let sentimentPipeline = null;

// DOM elements
const statusMessageEl = document.getElementById('statusMessage');
const errorContainerEl = document.getElementById('errorContainer');
const errorMessageEl = document.getElementById('errorMessage');
const analyzeBtn = document.getElementById('analyzeBtn');
const reviewPlaceholderEl = document.getElementById('reviewPlaceholder');
const reviewTextEl = document.getElementById('reviewText');
const resultContainerEl = document.getElementById('resultContainer');
const sentimentIconEl = document.getElementById('sentimentIcon');
const sentimentLabelEl = document.getElementById('sentimentLabel');
const confidenceTextEl = document.getElementById('confidenceText');
const progressFillEl = document.getElementById('progressFill');

// Initialize application on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadReviews();
        await initializeModel();
    } catch (error) {
        showError(`Failed to initialize application: ${error.message}`);
    }
});

// Load and parse TSV file
async function loadReviews() {
    updateStatus('Loading reviews data...');
    
    try {
        const response = await fetch('reviews_test.tsv');
        if (!response.ok) {
            throw new Error(`Failed to load TSV file: ${response.status} ${response.statusText}`);
        }
        
        const tsvContent = await response.text();
        
        // Parse TSV using Papa Parse
        const result = Papa.parse(tsvContent, {
            header: true,
            delimiter: "\t",
            skipEmptyLines: true
        });
        
        if (result.errors.length > 0) {
            console.warn('Parse warnings:', result.errors);
        }
        
        // Extract review texts from 'text' column and filter invalid entries
        reviews = result.data
            .map(row => row.text)
            .filter(text => text && typeof text === 'string' && text.trim().length > 0);
        
        if (reviews.length === 0) {
            throw new Error('No valid reviews found in the TSV file');
        }
        
        updateStatus(`Loaded ${reviews.length} reviews successfully.`);
    } catch (error) {
        throw new Error(`Error loading reviews: ${error.message}`);
    }
}

// Initialize Transformers.js model
async function initializeModel() {
    updateStatus('Loading sentiment analysis model...');
    
    try {
        // Initialize the pipeline with a sentiment analysis model
        sentimentPipeline = await pipeline('text-classification', 
            'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
        );
        
        updateStatus('Model loaded successfully! Ready for analysis.');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-play-circle"></i> Analyze Random Review';
    } catch (error) {
        throw new Error(`Error loading model: ${error.message}`);
    }
}

// Analyze a random review
async function analyzeRandomReview() {
    clearError();
    
    if (reviews.length === 0) {
        showError('No reviews available for analysis');
        return;
    }
    
    if (!sentimentPipeline) {
        showError('Sentiment model is not ready yet');
        return;
    }
    
    // Set loading state
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    updateStatus('Analyzing review sentiment...');
    
    try {
        // Select random review
        const randomIndex = Math.floor(Math.random() * reviews.length);
        const review = reviews[randomIndex];
        
        // Display review
        reviewPlaceholderEl.style.display = 'none';
        reviewTextEl.style.display = 'block';
        reviewTextEl.textContent = review;
        
        // Hide previous result
        resultContainerEl.style.display = 'none';
        
        // Run sentiment analysis
        const results = await sentimentPipeline(review, { topk: 2 });
        
        // Process results
        let sentiment = 'neutral';
        let confidence = 0;
        let label = '';
        
        if (results && results.length > 0) {
            const primaryResult = results[0];
            label = primaryResult.label;
            confidence = primaryResult.score;
            
            // Determine sentiment category
            if (label === 'POSITIVE' && confidence > 0.5) {
                sentiment = 'positive';
            } else if (label === 'NEGATIVE' && confidence > 0.5) {
                sentiment = 'negative';
            } else {
                sentiment = 'neutral';
            }
        }
        
        // Update UI with results
        updateResults(sentiment, label, confidence);
        updateStatus('Analysis complete!');
        
    } catch (error) {
        showError(`Analysis failed: ${error.message}`);
        console.error('Inference error:', error);
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-redo"></i> Analyze Another Review';
    }
}

// Update UI with sentiment results
function updateResults(sentiment, label, confidence) {
    // Set colors and icons based on sentiment
    let iconClass, icon, color;
    
    switch (sentiment) {
        case 'positive':
            iconClass = 'sentiment-positive';
            icon = 'fa-thumbs-up';
            color = '#10b981';
            break;
        case 'negative':
            iconClass = 'sentiment-negative';
            icon = 'fa-thumbs-down';
            color = '#ef4444';
            break;
        default:
            iconClass = 'sentiment-neutral';
            icon = 'fa-question-circle';
            color = '#f59e0b';
    }
    
    // Update sentiment display
    sentimentIconEl.className = `sentiment-icon ${iconClass}`;
    sentimentIconEl.innerHTML = `<i class="fas ${icon}"></i>`;
    sentimentLabelEl.textContent = label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();
    
    // Update confidence display
    const confidencePercent = Math.round(confidence * 100);
    confidenceTextEl.textContent = `Confidence: ${confidencePercent}%`;
    
    // Update progress bar
    progressFillEl.style.width = `${confidencePercent}%`;
    progressFillEl.style.backgroundColor = color;
    
    // Show result container
    resultContainerEl.style.display = 'block';
}

// Update status message
function updateStatus(message) {
    statusMessageEl.textContent = message;
    console.log(`Status: ${message}`);
}

// Show error message
function showError(message) {
    errorMessageEl.textContent = message;
    errorContainerEl.style.display = 'block';
    console.error(`Error: ${message}`);
}

// Clear error message
function clearError() {
    errorContainerEl.style.display = 'none';
    errorMessageEl.textContent = '';
}

// Event listener for analyze button
analyzeBtn.addEventListener('click', analyzeRandomReview);