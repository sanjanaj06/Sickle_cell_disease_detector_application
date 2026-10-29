"""
app/backend/app.py - Flask web application and REST API for Sickle Cell Detection.
"""
import os
import sys
import tempfile
from flask import Flask, jsonify, render_template, request

# Setup path to import backend modules & config
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

# Add both directories to sys.path so imports work regardless of how it's launched
for path in (BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from app.backend.inference import get_pipeline
except (ModuleNotFoundError, ImportError):
    from inference import get_pipeline

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "frontend", "templates"),
    static_folder=os.path.join(APP_DIR, "frontend", "static"),
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Renders the main clinical dashboard interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify backend status."""
    return jsonify({"status": "healthy", "service": "Sickle Cell Diagnostic API"})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts:
      - 'image': file upload
      - 'age': patient age (years)
      - 'gender': patient biological sex ('male' or 'female')
      - 'pain_crises': number of pain crises per year
      - 'spo2': oxygen saturation percentage
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided in request."}), 400

        file = request.files["image"]
        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        # Parse patient clinical parameters with defaults
        age = float(request.form.get("age", 25.0))
        gender = request.form.get("gender", "male").strip().lower()
        pain_crises = float(request.form.get("pain_crises", 0.0))
        spo2 = float(request.form.get("spo2", 98.0))

        if not (0 <= age <= 120):
            return jsonify({"success": False, "error": "Age must be between 0 and 120."}), 400
        if gender not in ("male", "female"):
            return jsonify({"success": False, "error": "Gender must be 'male' or 'female'."}), 400
        if pain_crises < 0:
            return jsonify({"success": False, "error": "Pain crises per year cannot be negative."}), 400
        if not (0 < spo2 <= 100):
            return jsonify({"success": False, "error": "SpO2 must be between 1 and 100%."}), 400

        # Save temporarily to run inference
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            file.save(temp_path)

        try:
            pipeline = get_pipeline()
            result = pipeline.run(
                image_path=temp_path,
                age=age,
                gender=gender,
                pain_crises_per_year=pain_crises,
                spo2=spo2,
            )
            return jsonify({"success": True, "data": result}), 200
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    print(f"\n==================================================")
    print(f" Sickle Cell AI Diagnostic Server Started on Port {port}")
    print(f" Web UI: http://localhost:{port}/")
    print(f"==================================================\n")
    app.run(host=host, port=port, debug=False)
