# Is There a Cat?

A small web app that finds every cat in an uploaded photo using instance
segmentation, outlines each one, and reports how many there are and roughly
where they're located.

The app runs [YOLOv8-seg](https://docs.ultralytics.com/tasks/segment/)
(via the `ultralytics` package), a pretrained convolutional neural network
that both detects objects and produces a pixel-level mask for each one. We
filter its output to the COCO "cat" class, then use OpenCV to draw the mask
outline, a bounding box, and a number on top of each detected cat, and
compute a rough grid position (e.g. "top-left", "center") from each cat's
bounding-box centroid.

## Run it locally

```bash
git clone <this-repo-url>
cd cat-finder
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The first run will download the small `yolov8n-seg.pt` model weights
(~7 MB) automatically. Then open the local URL
Streamlit prints (usually http://localhost:8501) and upload a photo.
