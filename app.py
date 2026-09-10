import io

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO

# COCO class id for "cat"
# https://docs.ultralytics.com/datasets/detect/coco/
CAT_CLASS_ID = 15


PALETTE_BGR = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
]

BASE_HEX = "#f8e8ff"
ACCENT_HEX = "#FFB3DE"
INK_HEX = "#201a33"

ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]

LOCATION_PHRASES = {
    "top-left": "the upper left of the frame",
    "top-centre": "the top, centered",
    "top-right": "the upper right of the frame",
    "middle-left": "the left side, neither low nor high",
    "centre": "the very centre of the frame",
    "middle-right": "the right side, neither low nor high",
    "bottom-left": "the lower left of the frame",
    "bottom-centre": "the bottom, centered",
    "bottom-right": "the lower right of the frame",
}

def roman_numeral(n: int) -> str:
    """Return a roman numeral for 1-20; fall back to the cardinal number."""
    return ROMAN[n - 1] if 1 <= n <= len(ROMAN) else str(n)

def render_wave_background() -> None:

    html = f"""
    <div id="wave-host" style="margin:0;padding:0;">
      <canvas id="wave-canvas" style="display:block;width:100%;height:100%;"></canvas>
    </div>
    <script>
      const frame = window.frameElement;
      if (frame) {{
        frame.style.position = 'fixed';
        frame.style.top = '0';
        frame.style.left = '0';
        frame.style.width = '100vw';
        frame.style.height = '100vh';
        frame.style.border = 'none';
        frame.style.pointerEvents = 'none';
      }}

      const canvas = document.getElementById('wave-canvas');
      const ctx = canvas.getContext('2d');

      function hexToRgb(hex) {{
        const n = parseInt(hex.replace('#', ''), 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
      }}
      const BASE = hexToRgb('{BASE_HEX}');
      const ACCENT = hexToRgb('{ACCENT_HEX}');
      const CELL = 26;      // grid spacing, in pixels
      const SPEED = 0.018;  // how quickly the wave animates

      function resize() {{
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }}
      window.addEventListener('resize', resize);
      resize();

      function lerp(a, b, x) {{ return a + (b - a) * x; }}

      let t = 0;
      function draw() {{
        ctx.fillStyle = '{BASE_HEX}';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const cols = Math.ceil(canvas.width / CELL) + 1;
        const rows = Math.ceil(canvas.height / CELL) + 1;

        for (let y = 0; y < rows; y++) {{
          for (let x = 0; x < cols; x++) {{
            const px = x * CELL;
            const py = y * CELL;
            // two overlapping sine waves, offset in phase, give a
            // drifting, non-repetitive ripple rather than a rigid grid scan
            const wave = Math.sin(x * 0.35 + t) * 0.5
                       + Math.sin(y * 0.28 - t * 0.8 + x * 0.05) * 0.5;
            const intensity = (wave + 1) / 2; // normalize to 0..1
            const alpha = Math.pow(intensity, 3) * 0.5;
            if (alpha > 0.02) {{
              const r = lerp(BASE[0], ACCENT[0], intensity) | 0;
              const g = lerp(BASE[1], ACCENT[1], intensity) | 0;
              const b = lerp(BASE[2], ACCENT[2], intensity) | 0;
              ctx.fillStyle = `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha.toFixed(3)}})`;
              const size = CELL * (0.25 + intensity * 0.55);
              ctx.beginPath();
              ctx.arc(px, py, size / 2, 0, Math.PI * 2);
              ctx.fill();
            }}
          }}
        }}
        t += SPEED;
        requestAnimationFrame(draw);
      }}
      draw();
    </script>
    """
    components.html(html, height=0)



@st.cache_resource(show_spinner=False)
def load_model(option) -> YOLO:
    """Load the pretrained YOLOv8 seg model (downloads on first run)"""
    return YOLO(option)
    #yolov8s-seg
    #yolov8n-seg

def grid_location(cx: float, cy: float, width: int, height: int) -> str:
    """Classify a point into one of nine regions of the image."""
    col = min(int(cx / width * 3), 2)
    row = min(int(cy / height * 3), 2)
    rows = ["top", "middle", "bottom"]
    cols = ["left", "centre", "right"]
    if row == 1 and col == 1:
        return "centre"
    return f"{rows[row]}-{cols[col]}"


def annotate_image(image_bgr: np.ndarray, result) -> tuple[np.ndarray, list[dict]]:
    """Trace, number, and record each detected cat."""
    h, w = image_bgr.shape[:2]
    overlay = image_bgr.copy()
    output = image_bgr.copy()
    locations: list[dict] = []

    if result.masks is None or len(result.boxes) == 0:
        return output, locations

    for i, (box, mask_xy) in enumerate(zip(result.boxes, result.masks.xy)):
        color = PALETTE_BGR[i % len(PALETTE_BGR)]
        poly = mask_xy.astype(np.int32)

        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(output, [poly], isClosed=True, color=color, thickness=2)

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        loc = grid_location(cx, cy, w, h)
        numeral = roman_numeral(i + 1)

        locations.append({
            "id": i + 1,
            "numeral": numeral,
            "confidence": conf,
            "location": loc,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
        })

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{numeral} . {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 1)
        label_y = max(0, y1 - th - 10)
        cv2.rectangle(output, (x1, label_y), (x1 + tw + 12, y1), color, -1)
        cv2.putText(output, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_TRIPLEX,
                    0.6, (250, 246, 238), 1, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
    return output, locations


def render_verdict(locations: list[dict]) -> None:
    if not locations:
        st.markdown('<p class="verdict">Cannot detect any cats; might they be hiding?</p>',unsafe_allow_html=True,)
        return
    if len(locations) == 1:
        verdict = "Indeed, there is one single cat."
    else:
        verdict = f"Why, there are {len(locations)} cats."
    st.markdown(f'<p class="verdict">{verdict}</p>', unsafe_allow_html=True)

    for loc in locations:
        phrase = LOCATION_PHRASES.get(loc["location"], loc["location"])
        st.markdown(
            f'<div class="entry"><span class="num">{loc["numeral"]}.</span>'
            f'A cat observed in {phrase}, identified with {loc["confidence"]:.0%} certainty.</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Is There a Cat?", page_icon="🐱", layout="centered")

    # load stylesheet
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    # procedural background wavey thing
    render_wave_background()

    st.markdown('<h1 class="block-title">Is There a Cat?</p>', unsafe_allow_html=True)
    st.markdown('<p class="block-subtitle">'
        'Might there be a cat hiding within your photograph? Better to check than to feel sorrow.'
        '</p>',unsafe_allow_html=True,)

    option_map = {
        "yolov8n-seg": "standard",
        "yolov8s-seg": "intense",
        "yolov8m-seg": "Super intense"}

    col1, col2 = st.columns([1, 0.15], vertical_alignment="center")

    with col1:
        st.markdown("How intense do you want the search to be?")

    with col2:
        with st.popover(":material/help:", type="secondary"):
            st.markdown("""
            This tool uses **YOLO (You Only Look Once)**, a computer-vision
            model for detecting and segmenting objects in images. 

            ### How do the intensities differ?

            **standard: YOLOv8n-seg (nano)**  
            Fastest and smallest model. Best for quick results.

            **intense: YOLOv8s-seg (small)**  
            More accurate, but also more demanding and might take longer.

            **super intense: YOLOv8m-seg (medium)**  
            High accuracy but also slowest.
            """)


    selection = st.pills(" ",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default="yolov8n-seg"
    )

    #st.write(f"Your selected option: {None if selection is None else selection}")
    yolo_model = selection

    # deco visual separation
    st.markdown('<p class="ornament">❇</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Submit a photograph", type=["jpg", "jpeg", "png", "bmp", "webp"],)

    if not uploaded:
        st.markdown('<p class="caption-line">Analysis will be conducted here.</p>', unsafe_allow_html=True,)
        return

## start YOLO
    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("YOLOing..."):
        model = load_model(yolo_model)
        results = model.predict(source=image_bgr, classes=[CAT_CLASS_ID], conf=0.25, verbose=False)
        result = results[0]
        annotated_bgr, locations = annotate_image(image_bgr, result)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True)
    st.markdown('<p class="caption-line">Figure I. — the specimen, examined.</p>', unsafe_allow_html=True,)

    render_verdict(locations)

    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")

    st.download_button("Save annotated figure", data=buf.getvalue(), file_name="cat_yoloed.png", mime="image/png",)



if __name__ == "__main__":
    main()
