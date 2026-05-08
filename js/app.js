// Stress Predictor Application JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Theme Toggle Initialization
    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;
    
    // Check for saved theme preference or system preference
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        html.classList.add('dark-mode');
    }
    
    // Theme toggle switch click handler
    themeToggle.addEventListener('click', function() {
        html.classList.toggle('dark-mode');
        
        // Save preference
        if (html.classList.contains('dark-mode')) {
            localStorage.setItem('theme', 'dark');
        } else {
            localStorage.setItem('theme', 'light');
        }
    });

    const predictionForm = document.getElementById('predictionForm');
    const predictionResultDiv = document.getElementById('predictionResult');
    const stressLevelSpan = document.getElementById('stressLevel');
    const errorMessageDiv = document.getElementById('errorMessage');
    const errorTextP = document.getElementById('errorText');

    // Form submission handler
    predictionForm.addEventListener('submit', async function(event) {
        event.preventDefault();

        // Clear previous results
        predictionResultDiv.style.display = 'none';
        errorMessageDiv.style.display = 'none';

        try {
            // Collect form data
            const formData = new FormData(predictionForm);
            const data = {
                sleep_quality: parseFloat(formData.get('sleep_quality')) || 4,
                exercise_minutes: parseFloat(formData.get('exercise_minutes')) || 0,
                hours_worked: parseFloat(formData.get('hours_worked')) || 0,
                social_interactions: parseFloat(formData.get('social_interactions')) || 0,
                work_load_score: parseFloat(formData.get('work_load_score')) || 5,
                heart_rate: parseFloat(formData.get('heart_rate')) || 70
            };

            // Validate data - ensure all values are valid numbers
            if (Object.values(data).some(val => isNaN(val))) {
                showError('Please enter valid numbers for all fields.');
                return;
            }

            // Validate ranges
            if (data.sleep_quality < 1 || data.sleep_quality > 8) {
                showError('Sleep Quality must be between 1 and 8');
                return;
            }
            if (data.work_load_score < 1 || data.work_load_score > 10) {
                showError('Workload Score must be between 1 and 10');
                return;
            }


            // Try to connect to backend API
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (response.ok) {
                    const result = await response.json();
                    // Handle both string prediction (from API) and raw value
                    displayPrediction(result.raw_value, result.prediction, result.confidence || 0);
                } else {
                    // Fallback to local prediction if backend fails
                    const localPrediction = calculateLocalPrediction(data);
                    displayPrediction(localPrediction, null, 0);
                }
            } catch (error) {
                // Backend not available, use local prediction
                console.log('Backend not available, using local prediction model');
                const localPrediction = calculateLocalPrediction(data);
                displayPrediction(localPrediction, null, 0);
            }

        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred while processing your input. Please try again.');
        }
    });

    // Local prediction calculation for demo
    function calculateLocalPrediction(data) {
        // Simple calculation based on input factors
        let stressScore = 0;

        // Sleep quality (lower is worse)
        if (data.sleep_quality < 3) stressScore += 2;
        if (data.sleep_quality < 2) stressScore += 2;

        // Exercise (less exercise = more stress)
        if (data.exercise_minutes < 20) stressScore += 1;
        if (data.exercise_minutes < 10) stressScore += 1;

        // Work hours (more hours = more stress)
        if (data.hours_worked > 50) stressScore += 2;
        if (data.hours_worked > 80) stressScore += 1;

        // Social interactions (less interaction = higher stress)
        if (data.social_interactions < 2) stressScore += 1;

        // Workload score
        if (data.work_load_score > 7) stressScore += 2;
        if (data.work_load_score > 9) stressScore += 1;

        // Heart rate (elevated = stress indicator)
        if (data.heart_rate > 100) stressScore += 2;
        if (data.heart_rate > 110) stressScore += 1;

        // Normalize to 0-2 scale for Low/Moderate/High
        return Math.min(2, Math.max(0, Math.round(stressScore / 4)));
    }

    // Display prediction results
    function displayPrediction(prediction, predictionText = null, confidence = 0) {
        let stressDescription = '';
        let badgeClass = '';
        let emoji = '';

        // If predictionText is provided from API, use it; otherwise map from prediction value
        if (predictionText) {
            stressDescription = predictionText;
            // Extract emoji and description from prediction text
            if (predictionText.includes('Low')) {
                badgeClass = 'bg-success';
                emoji = '😊';
            } else if (predictionText.includes('Moderate')) {
                badgeClass = 'bg-warning';
                emoji = '😐';
            } else if (predictionText.includes('High')) {
                badgeClass = 'bg-danger';
                emoji = '😟';
            } else {
                badgeClass = 'bg-danger';
                emoji = '😟';
            }
        } else {
            switch (prediction) {
                case 0:
                    stressDescription = '✅ Low Stress';
                    badgeClass = 'bg-success';
                    emoji = '😊';
                    break;
                case 1:
                    stressDescription = '⚠️ Moderate Stress';
                    badgeClass = 'bg-warning';
                    emoji = '😐';
                    break;
                case 2:
                default:
                    stressDescription = '🚨 High Stress';
                    badgeClass = 'bg-danger';
                    emoji = '😟';
            }
        }

        stressLevelSpan.textContent = stressDescription;
        stressLevelSpan.className = `badge ${badgeClass} p-3 fs-5`;

        // Add recommendations
        const recommendations = getRecommendations(prediction);
        const confidenceDisplay = confidence > 0 ? `
            <div class="mt-3">
                <strong>Model Confidence:</strong>
                <div class="progress mt-2" style="height: 25px;">
                    <div class="progress-bar" role="progressbar" style="width: ${confidence}%" aria-valuenow="${confidence}" aria-valuemin="0" aria-valuemax="100">
                        ${confidence.toFixed(1)}%
                    </div>
                </div>
            </div>
        ` : '';

        const resultHTML = `
            <h4 class="alert-heading">${emoji} Stress Assessment Results</h4>
            <p class="mb-3"><strong>Your Predicted Stress Level:</strong></p>
            <div class="stress-badge">
                <span class="badge ${badgeClass} p-3 fs-5">${stressDescription}</span>
            </div>
            ${confidenceDisplay}
            <hr>
            <div class="mt-3">
                <h6>Recommendations:</h6>
                <ul class="mb-0">
                    ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
            <small class="text-muted d-block mt-3">💡 This is a demonstration based on provided input values.</small>
        `;

        predictionResultDiv.innerHTML = resultHTML;
        predictionResultDiv.style.display = 'block';
        predictionResultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Get recommendations based on stress level
    function getRecommendations(prediction) {
        const recommendations = {
            0: [
                'Maintain your current lifestyle habits',
                'Continue regular exercise and social interactions',
                'Keep prioritizing quality sleep',
                'Monitor your stress levels regularly'
            ],
            1: [
                'Increase physical activity or exercise time',
                'Improve sleep quality and consistency',
                'Spend more time with friends and family',
                'Review and reorganize your work schedule'
            ],
            2: [
                'Seek professional help from a healthcare provider',
                'Implement stress-reduction techniques (meditation, yoga)',
                'Reduce work hours if possible',
                'Prioritize sleep (7-9 hours per night)',
                'Engage in regular physical activity',
                'Consider talking to a therapist or counselor'
            ]
        };

        return recommendations[prediction] || recommendations[0];
    }

    // Show error message
    function showError(message) {
        errorTextP.textContent = message;
        errorMessageDiv.style.display = 'block';
        errorMessageDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#' && document.querySelector(href)) {
                e.preventDefault();
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    console.log('Stress Predictor app initialized');
});
