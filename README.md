# DRIVER DROWSINESS DETECTION
### Using Temporal Facial Behavior Analysis

## Overview
A vision-based driver monitoring system that analyzes facial behaviour to identify signs of drowsiness. The system uses Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), statistical and temporal feature extraction, and a Random Forest classifier to distinguish between Alert and Drowsy states.

An audible warning mechanism is integrated to alert the driver when drowsiness is detected.

## Key Features
- Eye closure and mouth-opening analysis.
- EAR and MAR-based facial measurements.
- Statistical and temporal feature extraction.
- Random Forest-based drowsiness classification.
- Voice-based warning mechanism.
- Sequential vs. parallel execution comparison.
- Webcam-based live detection interface.

## Technologies Used
- **Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy, Pandas
- **Frontend:** HTML, CSS, JavaScript, Vite
- **Backend:** Python
- **Hardware:** Arduino

## Dataset
**UTA Real-Life Drowsiness Dataset (UTA-RLDD)**

The experiments use a reconstructed feature dataset containing 120 samples:
- Alert: 60 samples
- Drowsy: 60 samples
- Five-fold evaluation

## Experimental Results

| Metric | Result |
|---|---:|
| Accuracy | 80.83% |
| Drowsy Recall | 81.67% |
| F1-score | 0.8099 |
| ROC-AUC | 0.8603 |

**Parallel Execution:** 5.28% runtime reduction compared with sequential execution.

## Execution

**Backend:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.test_live_prediction
```

**Frontend (new terminal):**

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:5173/`

## Project Team
- **N C Deepika** – 26MAI0051
- **Jayant TN** – 26MAI0074

## Reference
M. E. Sahin, "Real-time driver drowsiness detection and classification on embedded systems using machine learning algorithms," *Traitement du Signal*, 2023.

[DOI: 10.18280/ts.400302](https://doi.org/10.18280/ts.400302)

---

Developed as an academic project focused on intelligent driver monitoring and real-time drowsiness detection.
