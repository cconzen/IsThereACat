# Is There a Cat?

A small web app that finds every cat in an uploaded photo using instance
segmentation, outlines each one, and reports how many there are and roughly
where they're located.

The app runs [YOLOv8-seg](https://docs.ultralytics.com/tasks/segment/)
(via the `ultralytics` package), a pretrained CNN
that both detects objects and produces a pixel-level mask for each one.

## Run it online

Check out the app [here](https://isthereacat.streamlit.app/).

## Run it locally

```bash
git clone https://github.com/cconzen/IsThereACat
cd cat-finder
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The first run will download the small `yolov8n-seg.pt` model weights
(~7 MB) automatically. Then open the local URL
Streamlit prints (usually http://localhost:8501) and upload a photo.
