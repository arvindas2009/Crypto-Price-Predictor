/**
 * Frontend JavaScript for Crypto Price Predictor
 *
 * Team Zephyrus - OlympAI Hackathon 2025
 * Bright Riders School, Abu Dhabi
 *
 * This file handles all client side interactions for the cryptocurrency
 * price prediction web application. It manages user input, communicates
 * with the Flask backend API, and dynamically updates the UI with results.
 *
 * Main Features:
 * - Input validation and sanitization
 * - Async API communication
 * - Dynamic result rendering
 * - Error handling and user feedback
 * - Mobile responsive navigation
 *
 * Authors: Arvin Das, Kanish Kannan Srinivasan, Muhammed Riswan Navas
 */

// NAVIGATION FUNCTIONS
/*
 * Toggle the visibility of the mobile navigation menu
 * This function is called when the hamburger menu icon is clicked on mobile
 * devices. It adds/removes the 'active' class to show or hide the menu.
 */
function toggleMenu() {
  const menu = document.getElementById('navMenu');
  menu.classList.toggle('active');
}

// INPUT HANDLING FUNCTIONS

/*
 * Set a cryptocurrency symbol in the input field
 * This function is used by the example buttons (BTC-USD, ETH-USD, etc.)
 * @param {string} symbol - The cryptocurrency symbol to set
 */
function setSymbol(symbol) {
  document.getElementById("symbolInput").value = symbol;
}

/*
 * Allows users to press Enter in the input field instead of clicking
 */
document
  .getElementById("symbolInput")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") predict();
  });

// PREDICTION WORKFLOW

/**
 * Main prediction function - Orchestrates the entire prediction workflow
 *
 * This async function handles the complete process of getting a prediction:
 * 1. Validates user input (ensures symbol is not empty)
 * 2. Shows loading indicator to user
 * 3. Sends POST request to Flask backend with the symbol
 * 4. Receives and processes the response
 * 5. Displays results or error messages
 * 6. Cleans up UI state (hides loading, re-enables button)
 */

async function predict() {
  // Get the cryptocurrency symbol from input field
  // trim() removes whitespace, toUpperCase() standardizes format
  const symbol = document
    .getElementById("symbolInput")
    .value.trim()
    .toUpperCase();

  // Validate that user entered a symbol
  if (!symbol) {
    showError("Please enter a crypto symbol");
    return;
  }

  // Update UI to show loading state
  document.getElementById("loading").classList.add("active");
  document.getElementById("results").classList.remove("active");
  document.getElementById("error").classList.remove("active");

  // Disable predict button to prevent multiple simultaneous requests
  document.getElementById("predictBtn").disabled = true;

  try {
    // Send POST request to Flask backend /predict endpoint
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: symbol }),
    });

    // Parse JSON response from server
    const data = await response.json();

    // Check if the request was successful
    // If not, throw an error with the message from backend
    if (!response.ok) {
      throw new Error(data.error || "Something went wrong");
    }

    // Display the successful prediction results
    displayResults(data);

  } catch (error) {
    // If any error occurred (network, server, or validation), show it to user
    showError(error.message);

  } finally {
    // Always execute cleanup regardless of success or failure
    // Hide loading indicator and re-enable the predict button
    document.getElementById("loading").classList.remove("active");
    document.getElementById("predictBtn").disabled = false;
  }
}

// RESULT DISPLAY FUNCTIONS
/*
 * Display prediction results in the user interface
 * This function takes the prediction data from the backend and updates all relevant HTML elements to show the results to the user.

 * Updates include:
 * - Cryptocurrency symbol name
 * - Prediction direction (UP/DOWN)
 * - Confidence percentage
 * - Current market price with correct currency symbol
 * - Model performance metrics (accuracy, Train ROC score)
 * - Analysis visualization charts
 *
 * @param {Object} data - Prediction data object from backend
 * @param {string} data.symbol - Cryptocurrency symbol (e.g., "BTC-USD")
 * @param {string} data.prediction - Prediction result: "UP" or "DOWN"
 * @param {number} data.confidence - Confidence percentage (0-100)
 * @param {number|null} data.current_price - Current market price
 * @param {string} data.currency_symbol - Currency symbol ($, ₹, €, etc.)
 * @param {number} data.accuracy - Model accuracy percentage
 * @param {number} data.best_cv - Cross-validation ROC-AUC score
 * @param {string} data.plot - Base64-encoded PNG image of analysis charts
 */
function displayResults(data) {
  // Update the symbol name header
  document.getElementById("symbolName").textContent = data.symbol;

  // Update prediction text (UP or DOWN)
  const predictionElement = document.getElementById("predictionText");
  predictionElement.textContent =
    data.prediction === "UP" ? "UP" : "DOWN";

  predictionElement.className =
    "prediction-main " + (data.prediction === "UP" ? "up" : "down");

  // Display confidence percentage
  document.getElementById(
    "confidenceText"
  ).textContent = `Confidence: ${data.confidence}%`;

  // Format and display current price with appropriate currency symbol
  // Default to $ if currency symbol not provided
  const currSym = data.currency_symbol || '$';

  // Handle null/undefined price values
  const priceText = (data.current_price !== null && data.current_price !== undefined)
    ? `${currSym}${data.current_price}`
    : 'N/A';
  document.getElementById("currentPrice").textContent = priceText;

  // Display model performance metrics
  document.getElementById("accuracy").textContent = `${data.accuracy}%`;
  document.getElementById("rocScore").textContent = data.best_cv;

  // Display analysis charts as base64-encoded image
  // The backend sends PNG image data encoded in base64 format
  document.getElementById("chartImage").src =
    "data:image/png;base64," + data.plot;

  // Show the results section with animation
  document.getElementById("results").classList.add("active");
}


// ERROR HANDLING FUNCTIONS

/**
 * Display an error message to the user
 * Shows error messages in a dedicated error element when something goes wrong, such as invalid input, network errors, or server-side issues.
 * @param {string} message - The error message to display to the user
 */
function showError(message) {
  const errorElement = document.getElementById("error");
  errorElement.textContent = message;
  errorElement.classList.add("active");
}

// NAVIGATION ENHANCEMENT

/**
 * Enable smooth scrolling for internal navigation links
 *
 * Adds smooth scrolling behavior when users click on navigation links that refers to a section
 * It also closes the mobile menu after navigation for better UX
 */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    // Prevent default jump to section behavior
    e.preventDefault();

    const target = document.querySelector(this.getAttribute('href'));

    if (target) {
      // Smoothly scroll to the target section
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // Close mobile navigation menu if it's open
      document.getElementById('navMenu').classList.remove('active');
    }
  });
});
