# Demo - Gradio Web UI

Interactive web demo for the Recommendation System.

## Quick Start

```bash
cd demo

# Install Gradio
pip install gradio

# Run demo
python app.py
```

Open http://localhost:7860 in your browser.

## Features

1. **Recommendations Tab**
   - Enter user ID or click "Random User"
   - Adjust alpha slider (LightGCN vs TF-IDF weight)
   - View user history and recommendations

2. **Compare Methods Tab**
   - Side-by-side comparison of LightGCN, TF-IDF, and Hybrid
   - See how different methods rank items

## Screenshot

The demo shows:
- User interaction history
- Recommendations with scores
- Method comparison

## Requirements

- gradio >= 4.0
- torch
- scipy
- numpy
