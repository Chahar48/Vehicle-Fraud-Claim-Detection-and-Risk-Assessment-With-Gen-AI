from setuptools import setup, find_packages

setup(
    name="fraud-detection-genai",
    version="0.1.0",
    description="AI-based vehicle insurance fraud detection & manual-review automation system",
    author="Mukesh Kumar",
    author_email="chaharmukesh518@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "xgboost",
        "sentence-transformers",
        "faiss-cpu",
        "pillow",
        "pytesseract",
        "pdf2image",
        "PyPDF2",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "streamlit",
        "python-dateutil",
        "pyyaml",
        "opencv-python",
        "pydantic>=2.0",
        "requests",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "run-api=run_api:main",
            "run-ui=ui.reviewer_app:main"
        ]
    }
)
