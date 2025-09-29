# CNN for Galaxy Classification + Webapp

This repository provides an implementation of a Convolutional Neural Network (CNN) for classifying galaxy morphologies using the Galaxy10 dataset. The project utilizes TensorFlow for building and training the model and follows standard preprocessing steps to ensure efficient data handling and model performance.

The accompanying webapp based on this model can be found [here](https://cnnapp-amisha.streamlit.app/%255D%28https:/cnnapp-amisha.streamlit.app/).

## TL;DR
This repository contains a Jupyter notebook (`Final_CNNN(1).ipynb`) that trains and evaluates a CNN for image classification. The notebook includes data loading, model definition, training loop, evaluation (metrics & confusion matrix), and sample inference cells. Detected frameworks/tools used: TensorFlow/Keras, model checkpoints.

## What’s inside
- `Final_CNNN(1).ipynb` — main notebook (training, evaluation, basic inference).
- `models/` — (optional) suggested folder for checkpoints (not committed).
- `data/` — place small sample images here; full dataset should be downloaded externally.
- `assets/` — add demo GIF / screenshots for portfolio.

## How to run (recommended)
1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3. Launch the notebook:
```bash
jupyter lab Final_CNNN(1).ipynb
# or
jupyter notebook Final_CNNN(1).ipynb
```

## Quick usage notes
- The notebook appears to use: TensorFlow/Keras, model checkpoints.
- If large model weights are used, **do not** commit them. Instead, add a `scripts/download_weights.py` that fetches weights from a release or Hugging Face, and load at runtime.
- Add a small `data/sample/` folder (≤ 5 MB) with a couple sample images so reviewers can run inference cells immediately.
- For portability, consider converting the notebook into a standalone script or Streamlit app (see below).

## Tips to make this portfolio-ready
1. **Add a short README** (this file) and a one-paragraph summary at the top of the notebook (markdown cell).
2. **Include a demo GIF** (`assets/demo.gif`) that shows: training quick pass / upload → inference / Grad-CAM (if present).
3. **Provide `requirements.txt`** listing exact versions detected from the notebook (e.g., PyTorch/TensorFlow, torchvision, scikit-learn, matplotlib, opencv-python, pillow).
4. **Create `scripts/download_weights.py`** to fetch heavy checkpoints and place them in `models/`.
5. **Optional:** Convert core inference cell to `streamlit_app.py` for a live demo (Streamlit Community Cloud).

## Minimal `requirements.txt` suggestion
```
jupyterlab
numpy
pandas
matplotlib
scikit-learn
Pillow
opencv-python
torch        # or tensorflow (replace as detected in notebook)
torchvision  # if using PyTorch
seaborn
```

