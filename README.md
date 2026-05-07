# 🧠 Stress Predictor

An AI-powered stress prediction application that assesses stress levels based on lifestyle factors and vital signs using machine learning.

## Overview

The Stress Predictor is a Flask-based web application that uses a trained scikit-learn machine learning model to predict stress levels. It analyzes multiple lifestyle and physiological factors to provide personalized stress assessments.

## Features

✨ **Key Features:**
- 🤖 Machine Learning-based stress prediction
- 📊 Multi-factor analysis (sleep, exercise, work hours, social interactions, workload, heart rate)
- 🎨 Responsive Bootstrap 5 UI
- 📱 Mobile-friendly design
- ⚡ Real-time form validation
- 💡 Personalized recommendations based on stress level
- 🔒 Secure Flask backend with JSON API

## Project Structure

```
stress_predictor/
├── app.py                 # Flask application and API
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   └── stress_data.csv   # Training dataset
├── models/
│   └── stress_model.pkl  # Trained ML model
├── static/
│   ├── css/
│   │   ├── bootstrap.min.css
│   │   ├── style.css
│   │   └── tailwind.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html        # Main web interface
└── .gitignore           # Git ignore file
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/akhilbalarushi-a11y/NewRepoA.git
cd stress_predictor
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the model (if not already trained):**
```bash
python train_model.py
```

5. **Run the Flask application:**
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Web Interface
1. Open `http://localhost:5000` in your browser
2. Fill in the stress assessment form with your current values:
   - Sleep Quality (1-5 scale)
   - Daily Exercise (minutes)
   - Hours Worked (per week)
   - Social Interactions (per week)
   - Workload Score (1-10 scale)
   - Heart Rate (BPM)
3. Click "Predict My Stress Level"
4. View your personalized stress assessment and recommendations

### API Usage

**Endpoint:** `POST /predict`

**Request Format:**
```json
{
    "sleep_quality": 7,
    "exercise_minutes": 30,
    "hours_worked": 8,
    "social_interactions": 5,
    "work_load_score": 6,
    "heart_rate": 70
}
```

**Response Format:**
```json
{
    "prediction": 0
}
```

**Prediction Values:**
- `0` = Low Stress
- `1` = Moderate Stress
- `2` = High Stress

## Model Details

### Features Used
- **Sleep Quality**: Scale 1-5 (higher = better)
- **Exercise Minutes**: Daily exercise duration
- **Hours Worked**: Weekly work hours
- **Social Interactions**: Weekly social engagements
- **Work Load Score**: Perceived workload 1-10
- **Heart Rate**: Current heart rate in BPM

### Model Type
- Algorithm: Scikit-learn classifier (RandomForest/Logistic Regression)
- Training Data: `stress_data.csv`
- Model File: `models/stress_model.pkl`
- Format: Joblib serialized pickle file

### Training
The model was trained on historical stress data to predict stress levels based on the input features. See `train_model.py` for training details.

## Technologies Used

**Backend:**
- Flask 3.1.3
- Python 3.12
- Scikit-learn 1.8.0
- Pandas 3.0.2
- Joblib 1.5.3

**Frontend:**
- HTML5
- Bootstrap 5.3.0
- CSS3
- Vanilla JavaScript
- Responsive Design

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` or `production`
- `FLASK_DEBUG`: Set to `True` for debug mode (development only)

### Flask Settings
Edit `app.py` to modify:
- Debug mode
- Port
- Host address
- Model path

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve main web interface |
| POST | `/predict` | Make stress prediction |

## Error Handling

The application handles the following error scenarios:

1. **Missing Model File:**
   - Returns 500 error if `models/stress_model.pkl` not found
   - Training script needs to be run first

2. **Invalid Input Data:**
   - Returns 400 error if required features are missing
   - Returns 400 error if features are invalid type

3. **Prediction Error:**
   - Returns 500 error if prediction processing fails
   - Check model file integrity

## Troubleshooting

### Model Not Found Error
```
Error: Model file not found. Please train and save your model.
```
**Solution:** Run `python train_model.py` to train and save the model.

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:** Change the port in `app.py` or kill the process using port 5000.

### Missing Dependencies
```
ModuleNotFoundError: No module named 'flask'
```
**Solution:** Ensure virtual environment is activated and run `pip install -r requirements.txt`

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment:
1. Set `debug=False` in `app.py`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Deploy with Docker or cloud platform
4. Enable HTTPS/SSL
5. Set up proper logging

### GitHub Pages (Static Site)
The `gh-pages` branch contains a static HTML version at:
```
https://akhilbalarushi-a11y.github.io/NewRepoA/
```

## Disclaimer

⚠️ **Important:** This tool is for **informational and educational purposes only**. It is **NOT** a substitute for professional medical advice. Always consult with qualified healthcare professionals for:
- Accurate stress assessment
- Mental health concerns
- Medical conditions
- Professional treatment recommendations

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Development

### Adding New Features
1. Test new ML models with `train_model.py`
2. Update feature list in `app.py` if needed
3. Update HTML form in `templates/index.html`
4. Update JavaScript in `static/js/script.js`
5. Update CSS styling in `static/css/style.css`

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## License

This project is provided as-is for educational purposes. See LICENSE file for details.

## Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Open an issue on GitHub
3. Contact the maintainer

## Author

**Akhil Balarushi**
- GitHub: [@akhilbalarushi-a11y](https://github.com/akhilbalarushi-a11y)
- Repository: [NewRepoA](https://github.com/akhilbalarushi-a11y/NewRepoA)

## Changelog

### Version 1.0.0 (May 2026)
- Initial release
- Core stress prediction functionality
- Web interface with Bootstrap
- REST API endpoint
- Model serialization and loading
- Responsive mobile design

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Bootstrap 5 Docs](https://getbootstrap.com/docs/5.0/)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

## Status

✅ Active Development
- [x] Core functionality
- [x] Web interface
- [x] API endpoints
- [x] Model training
- [ ] Advanced analytics
- [ ] User authentication
- [ ] Database integration

---

**Last Updated:** May 7, 2026

For the latest updates, visit the [GitHub repository](https://github.com/akhilbalarushi-a11y/NewRepoA).