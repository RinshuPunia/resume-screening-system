# resume-screening-system
An ML-based resume screening system that automatically ranks resumes against a job description using **TF-IDF and Cosine Similarity**. Built with classical Machine Learning  to demonstrate practical NLP techniques for a real-world HR-tech problem — recruiters manually reading through hundreds of resumes.

---

## 🎯 Problem Statement

Recruiters spend significant time manually screening resumes against job requirements. This project automates that process by:
- Matching resumes to a job description based on textual similarity
- Extracting years of experience automatically from resume text
- Ranking candidates so recruiters can focus on the strongest matches first

---

## ✨ Features

- **Flexible JD Input** — paste job description as text or upload as a PDF
- **Multi-format Resume Upload** — supports PDF and image (JPG/PNG) resumes
- **OCR Support** — extracts text from image-based/scanned resumes using Tesseract OCR
- **TF-IDF + Cosine Similarity Matching** — ranks resumes by relevance to the JD
- **Automatic Experience Extraction** — regex-based extraction handling multiple date formats (e.g., `"5+ years"`, `"07/2013 to Current"`, `"January 2013 to January 2014"`)

- **Interactive Dashboard** — built with Streamlit, includes ranking table, expandable candidate details, and score visualization

---

## 🛠️ Tech Stack

| Category | Tools/Libraries |
|---|---|
| Language | Python |
| Web Framework | Streamlit |
| ML/NLP | scikit-learn (TF-IDF, Cosine Similarity), NLTK (stopwords, lemmatization) |
| PDF Processing | pdfplumber |
| OCR | pytesseract, Tesseract OCR, Pillow |
| Visualization | Matplotlib |
| Data Handling | pandas |

---


## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed locally (required only for image-based resume support)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/resume-screening-system.git
cd resume-screening-system

# Install dependencies
pip install -r requirements.txt
```

### Configure Tesseract (Windows users, for image OCR support)
Update the Tesseract path in `app.py` to match your local installation:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```


## ⚠️ Known Limitations


- **Experience extraction** is regex-based and may not perfectly handle every resume format or date style, especially when education dates overlap with work experience sections.
- **TF-IDF similarity is lexical, not semantic** — it matches exact words/phrases rather than meaning, so scores in the 30–60% range are typical for genuinely strong matches (this mirrors how many real-world ATS systems perform).
- **OCR accuracy** for scanned/image resumes depends on image quality and may introduce minor text errors.

---

## 🔮 Future Improvements

- keyword-based skill extraction with **NER** (spaCy / Hugging Face Transformers) for open-ended skill detection
- Use **sentence embeddings** (e.g., Sentence-BERT) for semantic JD-resume matching instead of pure TF-IDF
- Add support for **multiple job description comparison** to suggest the best-fit role for a candidate
- Add a **weighted scoring system** combining similarity, experience, and skill-match ratio into a single ranking score
- Deploy on Streamlit Community Cloud for public access

