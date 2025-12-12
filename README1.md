# Complete Blood Count (CBC) Test Results Interpretation System

A Django-based web application for automated extraction, verification, and interpretation of Complete Blood Count (CBC) laboratory test results from medical documents using OCR and AI-powered analysis.

## Features

- **Document Upload & OCR**: Upload CBC test images and automatically extract text using Tesseract OCR
- **Intelligent Test Matching**: Fuzzy matching algorithm to map extracted test labels to canonical test names
- **Result Verification**: User-friendly interface to review and correct OCR-extracted values
- **Manual Entry**: Direct entry of CBC test results for users without physical documents
- **AI Interpretation**: Automated clinical interpretation of test results using Google Generative AI
- **User Authentication**: Secure user signup and login system
- **Result History**: Track and manage patient test results over time
- **Correction Logging**: Maintain audit trail of all result modifications
- **Reference Ranges**: Validate results against clinical reference ranges

## Technology Stack

- **Backend**: Django 5.2
- **Database**: SQLite (development) / PostgreSQL (production-ready)
- **OCR**: Tesseract-OCR, Pytesseract, Pillow
- **AI/ML**: Google Generative AI, RapidFuzz
- **Frontend**: HTML5, Bootstrap 5, Django Crispy Forms
- **Virtual Environment**: Python 3.11+

## Installation

### Prerequisites

1. **Python 3.11+** installed on your system
2. **Tesseract-OCR** installed:
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

### Setup Steps


2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv2
   # Windows
   venv2\Scripts\activate
   # Linux/macOS
   source venv2/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Tesseract path** (Windows users):
   - Update the path in `CBC_App/views.py` if Tesseract is installed in a different location:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

5. **Set up environment variables**:
   - Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_generative_ai_key
   DEBUG=True
   SECRET_KEY=your_django_secret_key
   ```
   - Get your Google API key from [Google AI Studio](https://ai.google.dev/)

6. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

7. **Create superuser** (optional, for admin panel):
   ```bash
   python manage.py createsuperuser
   ```

8. **Run development server**:
   ```bash
   python manage.py runserver
   ```
   - Access the application at `http://localhost:8000`

## Usage

### User Registration & Login
1. Click **Sign Up** to create a new account
2. Fill in your credentials and submit
3. Log in with your username and password

### Upload CBC Document
1. Navigate to **Upload** section
2. Select a CBC test image (JPG, PNG)
3. Submit the form
4. System automatically extracts text using OCR

### Verify Results
1. Review OCR-extracted values on the verification page
2. Correct any misread values if needed
3. Confirm and save the verified results
4. View interpretation and clinical recommendations

### Manual Entry
1. Go to **Manual Entry** section
2. Select test from dropdown or enter test name
3. Input test value, unit, and reference ranges
4. Submit to get AI-powered interpretation

### View History
1. Access **Home** page to see your recent test results
2. Click on any result to view detailed information and interpretation

## Project Structure

```
CBC_Main_Project/
├── manage.py                          # Django management script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── CBC_App/                           # Main Django application
│   ├── models.py                      # Database models
│   ├── views.py                       # View functions and business logic
│   ├── forms.py                       # Django forms
│   ├── urls.py                        # URL routing
│   ├── ocr.py                         # OCR processing
│   ├── admin.py                       # Django admin configuration
│   ├── signals.py                     # Django signals
│   ├── tests.py                       # Unit tests
│   ├── config.yml                     # App configuration
│   ├── migrations/                    # Database migrations
│   ├── templates/                     # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── signup.html
│   │   ├── upload.html
│   │   ├── verify.html
│   │   ├── manual_entry.html
│   │   └── manual_entry_result.html
│   ├── utils/                         # Utility modules
│   │   ├── cbc_interpreter.py         # CBC interpretation logic
│   │   └── interpreter_service.py     # Interpreter service
│   ├── management/                    # Custom management commands
│   └── __pycache__/
├── CBC_Main_Project/                  # Django project configuration
│   ├── settings.py                    # Django settings
│   ├── urls.py                        # Root URL configuration
│   ├── wsgi.py                        # WSGI application
│   ├── asgi.py                        # ASGI application
│   └── __pycache__/
├── templates/                         # Root templates directory
├── static/                            # Static files (CSS, JS)
├── media/                             # User-uploaded files
└── venv2/                             # Virtual environment
```

## Database Models

### UploadDocument
Stores metadata about uploaded CBC test documents
- user: Foreign key to User
- file: Document file
- filename: Original filename
- upload_time: Timestamp
- raw_text: Extracted OCR text
- status: Processing status (pending/processed)

### TestResult
Individual test result extracted from document
- document: Foreign key to UploadDocument
- user: Foreign key to User
- test: Foreign key to CanonicalTest
- label_raw: Raw extracted label
- value_raw: Raw extracted value
- value_numeric: Parsed numeric value
- unit: Measurement unit
- verified: Verification status

### CanonicalTest
Reference database of all possible CBC tests
- code: Standard test code (e.g., "HGB")
- display_name: Human-readable test name
- reference_min/reference_max: Normal range values

### TestInterpretation
AI-generated interpretation of test results
- test_result: Foreign key to TestResult
- flag: Result status (normal/abnormal/critical)
- text: Interpretation text
- meta: Additional metadata

### CorrectionLog
Audit trail of result corrections
- result: Foreign key to TestResult
- corrected_by: User who made correction
- previous_value_raw/new_value_raw: Change tracking
- reason: Reason for correction

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page / document list |
| `/upload/` | GET, POST | Upload CBC document |
| `/verify/<document_id>/` | GET, POST | Verify extracted results |
| `/manual-entry/` | GET | Manual entry form |
| `/manual-entry-process/` | POST | Process manual entry |
| `/login/` | GET, POST | User login |
| `/signup/` | GET, POST | User registration |
| `/logout/` | GET | User logout |
| `/admin/` | GET, POST | Django admin panel |

## Configuration

### Settings (CBC_App/config.yml)
- OCR language settings
- Fuzzy matching thresholds
- UI preferences

### Interpreter Config (CBC_App/interpreter_src/config2.yml)
- AI model parameters
- Reference ranges
- Interpretation rules

## Development

### Running Tests
```bash
python manage.py test CBC_App
```

### Collecting Static Files
```bash
python manage.py collectstatic
```

### Database Migrations
```bash
# Create new migration
python manage.py makemigrations

# Apply migrations
python manage.py migrate
```

### Django Shell
```bash
python manage.py shell
```

## Troubleshooting

### Tesseract Not Found (Windows)
- Ensure Tesseract-OCR is installed
- Update path in `views.py` to match installation directory
- Restart the development server

### OCR Not Working
- Check image quality and format (JPG, PNG recommended)
- Ensure image contains clear text
- Try uploading a different image

### Google API Key Issues
- Verify API key in `.env` file
- Check that Generative AI API is enabled in Google Cloud Console
- Ensure quota limits haven't been exceeded

### Database Locked Error
- Delete `db.sqlite3` and run migrations again
- Ensure only one development server is running
- Close any concurrent database access

## Performance Optimization

- **Image Compression**: Large images are automatically processed
- **Database Indexing**: Frequently queried fields are indexed
- **Caching**: Consider implementing Redis for session caching in production
- **Async Tasks**: OCR processing can be moved to Celery for better scalability

## Security Considerations

- **Authentication**: All sensitive operations require user login
- **CSRF Protection**: Enabled on all POST requests
- **SQL Injection**: ORM prevents SQL injection
- **XSS Protection**: Template auto-escaping enabled
- **Secret Key**: Change `SECRET_KEY` in production

### Production Deployment
For production deployment:
1. Set `DEBUG=False` in settings
2. Use strong `SECRET_KEY`
3. Configure allowed hosts
4. Use PostgreSQL instead of SQLite
5. Enable HTTPS
6. Configure proper email backend for notifications
7. Set up proper logging
8. Use production WSGI server (Gunicorn)

## License

This project is licensed under the MIT License - see LICENSE file for details.



## Acknowledgments

- Tesseract-OCR for text extraction
- Google Generative AI for clinical interpretation
- RapidFuzz for intelligent test matching
- Django community for the excellent framework

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Document upload with OCR
- Result verification system
- Manual entry interface
- AI-powered interpretation
- User authentication
- Result history tracking
#
