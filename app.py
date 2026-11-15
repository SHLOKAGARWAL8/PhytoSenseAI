
"""
AgroDetect - EfficientNet-B0 only
Phase 3: Grad-CAM
Phase 4: Flask web application

Expected checkpoint:
    models/efficientnet_b0_best.pth

Run locally:
    python app.py

Render start command:
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180
"""

import os
import json
import uuid
import logging
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps, UnidentifiedImageError
from flask import (
    Flask,
    request,
    render_template_string,
    url_for
)
from werkzeug.utils import secure_filename
from torchvision import models, transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "AGRODETECT_MODEL_PATH",
        str(ROOT / "models" / "efficientnet_b0_best.pth")
    )
)

CLASS_MAP_PATH = ROOT / "class_indices.json"

UPLOAD_DIR = ROOT / "static" / "uploads"
HEATMAP_DIR = ROOT / "static" / "heatmaps"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
NUM_CLASSES = 38
MAX_FILE_SIZE = 12 * 1024 * 1024

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reduce CPU thread contention on small Render instances.
torch.set_num_threads(1)

# Prevent simultaneous predictions from competing for memory.
INFERENCE_LOCK = threading.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgroDetect")


# ============================================================
# 2. CLASS MAPPING
# ============================================================

DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_class_names():
    """Load the exact preprocessing class-index mapping."""

    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH, encoding="utf-8") as f:
            mapping = json.load(f)

        if isinstance(mapping, dict):
            names = [
                name for name, index in
                sorted(mapping.items(), key=lambda item: item[1])
            ]
        else:
            names = list(mapping)

        if len(names) != NUM_CLASSES:
            raise ValueError(
                f"class_indices.json contains {len(names)} classes; "
                f"expected {NUM_CLASSES}."
            )

        return names

    train_dir = ROOT / "dataset" / "train"

    if train_dir.exists():
        names = sorted(
            p.name for p in train_dir.iterdir() if p.is_dir()
        )

        if len(names) == NUM_CLASSES:
            return names

    logger.warning(
        "class_indices.json and dataset/train unavailable. "
        "Using the standard PlantVillage class ordering."
    )

    return DEFAULT_CLASSES


CLASS_NAMES = load_class_names()


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. EFFICIENTNET-B0 MODEL
# ============================================================

def extract_state_dict(checkpoint):
    """Handle common PyTorch checkpoint dictionary layouts."""

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")

    state = checkpoint

    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "weights"
    ):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            break

    if not isinstance(state, dict):
        raise ValueError("No valid model state dictionary found.")

    state = {
        key: value
        for key, value in state.items()
        if torch.is_tensor(value)
    }

    if not state:
        raise ValueError("Checkpoint contains no model tensors.")

    # Remove common training wrappers.
    prefixes = ("module.", "_orig_mod.", "model.")

    cleaned = {}

    for key, value in state.items():
        while True:
            matched = False

            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    matched = True
                    break

            if not matched:
                break

        cleaned[key] = value

    return cleaned


def build_model():
    """Construct EfficientNet-B0 with the 38-class output head."""

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model


def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"EfficientNet checkpoint not found: {MODEL_PATH}\n"
            "Place efficientnet_b0_best.pth inside the models folder."
        )

    logger.info("Loading EfficientNet-B0: %s", MODEL_PATH)

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

    state_dict = extract_state_dict(checkpoint)

    model = build_model()

    # Strict loading prevents silently using random or missing weights.
    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    # Grad-CAM needs gradients on its target activation,
    # not on the model's parameters.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    logger.info("EfficientNet-B0 loaded successfully.")
    logger.info("Device: %s", DEVICE)

    return model


MODEL = None
MODEL_ERROR = None

try:
    MODEL = load_model()

except Exception as exc:
    MODEL_ERROR = str(exc)
    logger.exception("Model initialization failed.")


# ============================================================
# 5. GRAD-CAM IMPLEMENTATION
# ============================================================

class GradCAM:
    """
    Memory-conscious Grad-CAM for EfficientNet-B0.

    Uses the final convolutional layer of EfficientNet-B0.
    """

    def __init__(self, model):

        self.model = model
        self.activations = None

        target_layer = None

        for module in model.features.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module

        if target_layer is None:
            raise RuntimeError(
                "Could not locate EfficientNet convolutional layer."
            )

        self.handle = target_layer.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, module, inputs, output):

        # Enable gradients for the selected feature map.
        output.requires_grad_(True)

        self.activations = output

    def generate(self, input_tensor):

        self.activations = None

        with torch.enable_grad():

            logits = self.model(input_tensor)

            if self.activations is None:
                raise RuntimeError(
                    "Grad-CAM did not capture target activations."
                )

            predicted_class = int(
                logits.argmax(dim=1).item()
            )

            gradients = torch.autograd.grad(
                outputs=logits[:, predicted_class].sum(),
                inputs=self.activations,
                retain_graph=False,
                create_graph=False
            )[0]

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            cam = torch.relu(
                (weights * self.activations).sum(
                    dim=1,
                    keepdim=True
                )
            )

            cam = F.interpolate(
                cam,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0].detach().cpu().numpy()

            cam_min = cam.min()
            cam_max = cam.max()

            cam = (
                (cam - cam_min) /
                (cam_max - cam_min + 1e-8)
            )

            return cam, logits.detach()

    def close(self):

        self.handle.remove()
        self.activations = None


# ============================================================
# 6. HEATMAP GENERATION
# ============================================================

def create_heatmap(original, cam, output_path):
    """
    Generate a colored Grad-CAM heatmap and overlay.
    """

    cam_image = Image.fromarray(
        np.uint8(np.clip(cam, 0, 1) * 255)
    )

    cam_image = cam_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    heat = np.asarray(cam_image).astype(np.float32) / 255.0

    # Blue -> cyan/green -> yellow/red style heatmap.
    red = np.clip(1.5 * heat, 0, 1)

    green = np.clip(
        1.5 - np.abs(2 * heat - 1) * 1.5,
        0,
        1
    )

    blue = np.clip(1.0 - 1.5 * heat, 0, 1)

    colored = np.stack(
        [red, green, blue],
        axis=-1
    )

    colored = np.uint8(colored * 255)

    heatmap = Image.fromarray(colored)

    overlay = Image.blend(
        original.convert("RGB"),
        heatmap,
        alpha=0.42
    )

    overlay.save(output_path, quality=95)

    return output_path


# ============================================================
# 7. DISEASE INFORMATION
# ============================================================

def get_guidance(class_name):
    """
    Practical general guidance.
    Chemical treatment should be confirmed locally.
    """

    label = class_name.split("___")[-1]
    label = label.replace("_", " ").strip()

    lower = label.lower()

    if "healthy" in lower:

        return {
            "status": "Healthy",
            "description": "The model classified the leaf as healthy.",
            "remedy": [
                "Continue regular crop monitoring.",
                "Maintain appropriate irrigation.",
                "Keep the growing area free of infected plant debris."
            ],
            "prevention": [
                "Inspect leaves regularly.",
                "Use clean tools.",
                "Maintain suitable plant spacing."
            ]
        }

    if "powdery mildew" in lower:

        remedy = [
            "Remove severely affected leaves where practical.",
            "Improve airflow around plants.",
            "Avoid excessive nitrogen fertilization.",
            "Consult local agricultural guidance for suitable fungicides."
        ]

    elif "rust" in lower:

        remedy = [
            "Remove heavily infected plant debris.",
            "Improve air circulation.",
            "Avoid overhead irrigation where possible.",
            "Seek local guidance on appropriate disease management."
        ]

    elif "bacterial" in lower:

        remedy = [
            "Remove severely infected plant material.",
            "Avoid handling plants when foliage is wet.",
            "Disinfect gardening tools.",
            "Use disease-free planting material."
        ]

    elif "late blight" in lower:

        remedy = [
            "Remove infected plant material promptly.",
            "Avoid overhead irrigation.",
            "Prevent prolonged leaf wetness.",
            "Consult an agricultural specialist about locally approved treatment."
        ]

    elif "early blight" in lower:

        remedy = [
            "Remove affected lower leaves.",
            "Clear infected crop debris.",
            "Use crop rotation.",
            "Maintain good spacing and airflow."
        ]

    elif "scab" in lower:

        remedy = [
            "Remove fallen infected leaves and fruit.",
            "Prune for better airflow.",
            "Use disease-resistant varieties where available.",
            "Follow local disease-management recommendations."
        ]

    elif "mosaic" in lower or "yellow leaf curl" in lower:

        remedy = [
            "Isolate affected plants where practical.",
            "Remove severely infected plants according to local guidance.",
            "Inspect for insect vectors.",
            "Use clean planting material."
        ]

    elif "spider mite" in lower:

        remedy = [
            "Inspect the underside of leaves.",
            "Remove heavily infested leaves.",
            "Monitor mite populations.",
            "Use locally recommended integrated pest management."
        ]

    elif "citrus greening" in lower or "huanglongbing" in lower:

        remedy = [
            "Contact a local agricultural extension specialist.",
            "Inspect for Asian citrus psyllid.",
            "Follow regional quarantine and infected-tree guidance."
        ]

    elif "mold" in lower:

        remedy = [
            "Remove affected leaves.",
            "Improve ventilation.",
            "Avoid excess humidity.",
            "Avoid prolonged leaf wetness."
        ]

    else:

        remedy = [
            "Remove severely affected plant material.",
            "Improve airflow and growing conditions.",
            "Avoid unnecessary overhead watering.",
            "Consult a local agricultural specialist for diagnosis-specific treatment."
        ]

    return {
        "status": "Disease detected",
        "description": f"Predicted condition: {label}.",
        "remedy": remedy,
        "prevention": [
            "Inspect crops regularly.",
            "Remove infected plant debris.",
            "Use healthy planting material.",
            "Maintain appropriate irrigation and spacing.",
            "Monitor nearby plants for similar symptoms."
        ]
    }


# ============================================================
# 8. OPTIMIZED PREDICTION PIPELINE
# ============================================================

def predict_image(image_path):

    if MODEL is None:
        raise RuntimeError(
            "EfficientNet-B0 is not loaded: " + str(MODEL_ERROR)
        )

    # Open and normalize uploaded image.
    with Image.open(image_path) as image:

        original = ImageOps.exif_transpose(image).convert("RGB")

        # Limit unusually large images to reduce memory consumption.
        original.thumbnail(
            (1600, 1600),
            Image.Resampling.LANCZOS
        )

    input_tensor = transform(original).unsqueeze(0).to(DEVICE)

    # One forward pass supplies both prediction and Grad-CAM.
    # The lock prevents simultaneous inference requests.
    with INFERENCE_LOCK:

        gradcam = GradCAM(MODEL)

        try:

            cam, logits = gradcam.generate(input_tensor)

        finally:

            gradcam.close()

    # Calculate prediction probabilities.
    probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_index].item() * 100
    )

    prediction = CLASS_NAMES[predicted_index]

    # Calculate top three predictions.
    top_values, top_indices = torch.topk(
        probabilities,
        k=3
    )

    top_predictions = []

    for value, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):

        top_predictions.append({
            "name": CLASS_NAMES[index],
            "confidence": round(value * 100, 2)
        })

    # Generate Grad-CAM image.
    heatmap_filename = f"{uuid.uuid4().hex}.jpg"

    heatmap_path = HEATMAP_DIR / heatmap_filename

    create_heatmap(
        original,
        cam,
        heatmap_path
    )

    return {
        "prediction": prediction,

        "display_name": prediction.replace(
            "___",
            " - "
        ).replace("_", " "),

        "confidence": round(confidence, 2),

        "top_predictions": top_predictions,

        "guidance": get_guidance(prediction),

        "heatmap": heatmap_filename
    }


# ============================================================
# 9. FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# 10. EMBEDDED FRONTEND
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>AgroDetect | Plant Disease Intelligence</title>

<style>

:root {
    --bg: #f2f7f2;
    --surface: #ffffff;
    --green: #187344;
    --dark: #103b29;
    --text: #183528;
    --muted: #64796d;
    --border: #dce9df;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: radial-gradient(
        circle at 12% 0%,
        #e2f2e5 0,
        transparent 34%
    ), var(--bg);

    color: var(--text);

    font-family: Inter, Segoe UI, Arial, sans-serif;
}

header {
    background: linear-gradient(
        115deg,
        #0d3324,
        #17613d 72%,
        #237a4b
    );

    color: white;

    padding: 20px max(
        5%,
        calc((100% - 1180px) / 2)
    );

    display: flex;

    align-items: center;

    justify-content: space-between;

    box-shadow: 0 8px 28px rgba(11, 48, 31, .16);
}

.brand {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.5px;
}

.brand span {
    color: #8fdda4;
}

header small {
    color: #c5dfcd;
}

.container {
    width: min(1160px, 92%);
    margin: 48px auto;
}

.hero {
    margin-bottom: 30px;

    padding: 30px 34px;

    border-radius: 22px;

    background: linear-gradient(
        120deg,
        rgba(255,255,255,.94),
        rgba(235,247,237,.92)
    );

    border: 1px solid #dcebe0;

    box-shadow: 0 14px 38px rgba(24,75,45,.06);
}

.hero h1 {
    font-size: clamp(30px, 5vw, 46px);

    margin-bottom: 10px;

    color: var(--dark);

    letter-spacing: -1.3px;

    line-height: 1.12;
}

.hero p {
    color: var(--muted);

    line-height: 1.7;

    max-width: 760px;

    font-size: 16px;
}

.panel {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 20px;

    padding: 30px;

    margin-bottom: 24px;

    box-shadow: 0 12px 34px rgba(20, 60, 35, .055);

    transition: transform .18s ease, box-shadow .18s ease;
}

.panel:hover {
    box-shadow: 0 16px 40px rgba(20, 60, 35, .085);
}

.upload {
    border: 2px dashed #8fbea0;

    border-radius: 18px;

    padding: 46px 22px;

    text-align: center;

    background: linear-gradient(
        145deg,
        #f7fcf7,
        #eef8f0
    );
}

.upload h3 {
    color: var(--dark);
    font-size: 21px;
    margin-bottom: 8px;
}

input[type=file] {
    display: block;

    max-width: 100%;

    margin: 20px auto;

    padding: 12px;

    border: 1px solid #d6e6d9;

    border-radius: 10px;

    background: white;
}

button {
    background: linear-gradient(
        110deg,
        #187344,
        #239354
    );

    color: white;

    border: none;

    padding: 14px 28px;

    border-radius: 11px;

    font-weight: 700;

    cursor: pointer;

    box-shadow: 0 6px 16px rgba(24,115,68,.18);

    transition: transform .15s ease, box-shadow .15s ease;
}

button:hover {
    background: linear-gradient(
        110deg,
        #125d36,
        #187344
    );

    transform: translateY(-1px);

    box-shadow: 0 9px 20px rgba(24,115,68,.24);
}

button:disabled {
    background: #9ba99e;
    cursor: not-allowed;
}

.error {
    color: #a32727;

    background: #fff0f0;

    padding: 14px;

    border-radius: 10px;

    margin-bottom: 20px;

    overflow-wrap: anywhere;
}

.notice {
    color: #76551c;

    background: #fff7e5;

    border: 1px solid #f0dfb8;

    padding: 14px;

    border-radius: 10px;

    margin-bottom: 20px;

    overflow-wrap: anywhere;
}

.result-title {
    color: var(--dark);

    font-size: 25px;

    margin-top: 0;

    letter-spacing: -.45px;
}

.confidence {
    display: inline-block;

    font-size: 36px;

    color: var(--green);

    font-weight: 800;

    margin: 8px 0;

    padding: 5px 15px;

    border-radius: 12px;

    background: #eaf7ee;
}

.images {
    display: grid;

    grid-template-columns: repeat(2, 1fr);

    gap: 20px;
}

.images img {
    width: 100%;

    max-height: 440px;

    object-fit: contain;

    background: #f7faf7;

    padding: 7px;

    border-radius: 12px;

    border: 1px solid var(--border);
}

.image-label {
    font-weight: 700;

    color: var(--dark);

    margin-bottom: 10px;
}

.result-grid {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 24px;
}

ul {
    padding-left: 21px;

    line-height: 1.9;
}

.top-item {
    display: flex;

    justify-content: space-between;

    gap: 18px;

    padding: 12px 0;

    border-bottom: 1px solid var(--border);
}

.panel h2 {
    letter-spacing: -.45px;
}

footer {
    padding: 30px;

    text-align: center;

    color: var(--muted);

    border-top: 1px solid var(--border);

    background: rgba(255,255,255,.55);
}

@media(max-width:700px) {

    .container {
        margin: 24px auto;
    }

    .hero {
        padding: 24px 20px;
    }

    .panel {
        padding: 20px;
    }

    header {
        gap: 12px;

        flex-wrap: wrap;

        padding: 16px 5%;
    }

    .images,
    .result-grid {
        grid-template-columns: 1fr;
    }
}

</style>

</head>

<body>

<header>

    <div class="brand">
        Agro<span>Detect</span>
    </div>

    <small>
        Powered by EfficientNet-B0
    </small>

</header>


<main class="container">


<section class="hero">

    <h1>
        Plant Disease Intelligence
    </h1>

    <p>
        Upload a plant leaf image to identify its condition,
        visualize the model's attention using Grad-CAM,
        and explore recommended management practices.
    </p>

</section>


{% if error %}

<div class="notice">

    <strong>Model unavailable</strong>

    <br>

    {{ error }}

</div>

{% endif %}


{% if message %}

<div class="error">
    {{ message }}
</div>

{% endif %}


<section class="panel">

    <form
        action="/predict"
        method="POST"
        enctype="multipart/form-data"
    >

        <div class="upload">

            <h3>
                Upload a leaf photograph
            </h3>

            <p>
                Supported: JPG, JPEG, PNG, WEBP, BMP
            </p>

            <input
                type="file"
                name="image"
                accept="image/*"
                required
                {% if not ready %}disabled{% endif %}
            >

            <br>

            <button
                type="submit"
                {% if not ready %}disabled{% endif %}
            >

                Analyze Plant Image

            </button>

        </div>

    </form>

</section>


{% if result %}


<section class="panel">

    <h2 class="result-title">
        Prediction Results
    </h2>

    <h3>
        {{ result.display_name }}
    </h3>

    <div class="confidence">
        {{ result.confidence }}%
    </div>

    <p>
        Model confidence score
    </p>

</section>


<section class="panel">

    <h2>
        Grad-CAM Visualization
    </h2>

    <div class="images">

        <div>

            <div class="image-label">
                Original Image
            </div>

            <img
                src="{{ original_url }}"
                alt="Uploaded plant leaf"
            >

        </div>


        <div>

            <div class="image-label">
                Grad-CAM Heatmap Overlay
            </div>

            <img
                src="{{ heatmap_url }}"
                alt="Grad-CAM visualization"
            >

        </div>

    </div>

    <p>
        Highlighted regions indicate image areas that contributed
        to the model's selected prediction. They are not a
        definitive biological diagnosis.
    </p>

</section>


<section class="panel">

    <h2>
        Top Predictions
    </h2>

    {% for item in result.top_predictions %}

        <div class="top-item">

            <strong>
                {{ item.name.replace("___", " - ").replace("_", " ") }}
            </strong>

            <span>
                {{ item.confidence }}%
            </span>

        </div>

    {% endfor %}

</section>


<section class="panel">

    <h2>
        Disease Information
    </h2>

    <p>
        {{ result.guidance.description }}
    </p>

    <div class="result-grid">


        <div>

            <h3>
                Management Recommendations
            </h3>

            <ul>

                {% for item in result.guidance.remedy %}

                    <li>
                        {{ item }}
                    </li>

                {% endfor %}

            </ul>

        </div>


        <div>

            <h3>
                Prevention
            </h3>

            <ul>

                {% for item in result.guidance.prevention %}

                    <li>
                        {{ item }}
                    </li>

                {% endfor %}

            </ul>

        </div>


    </div>

</section>


{% endif %}


</main>


<footer>

    AgroDetect | Plant Disease Classification and Explainable AI

    <br>

    EfficientNet-B0 + Grad-CAM

</footer>


</body>
</html>
"""


# ============================================================
# 11. ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(
        HTML,
        ready=MODEL is not None,
        error=MODEL_ERROR,
        result=None,
        message=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    if MODEL is None:

        return render_template_string(
            HTML,
            ready=False,
            error=MODEL_ERROR,
            result=None,
            message="The trained EfficientNet-B0 model is unavailable."
        )

    if "image" not in request.files:

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=None,
            message="Please select an image."
        )

    file = request.files["image"]

    if not file or not file.filename:

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=None,
            message="No image selected."
        )

    if not allowed_file(file.filename):

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=None,
            message="Unsupported image format."
        )

    filename = secure_filename(file.filename)

    unique_name = f"{uuid.uuid4().hex}_{filename}"

    image_path = UPLOAD_DIR / unique_name

    try:

        file.save(image_path)

        # Validate actual image contents.
        with Image.open(image_path) as image:
            image.verify()

        result = predict_image(image_path)

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=result,
            message=None,
            original_url=url_for(
                "static",
                filename=f"uploads/{unique_name}"
            ),
            heatmap_url=url_for(
                "static",
                filename=f"heatmaps/{result['heatmap']}"
            )
        )

    except (UnidentifiedImageError, OSError):

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=None,
            message="The uploaded file is not a valid image."
        )

    except Exception as exc:

        logger.exception("Prediction failed")

        return render_template_string(
            HTML,
            ready=True,
            error=None,
            result=None,
            message=f"Prediction failed: {exc}"
        )


@app.route("/health", methods=["GET"])
def health():

    return {
        "application": "AgroDetect",
        "model": "EfficientNet-B0",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "classes": len(CLASS_NAMES),
        "error": MODEL_ERROR
    }


# ============================================================
# 12. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n====================================")
    print("       AGRODETECT APPLICATION")
    print("====================================")

    print("Model: EfficientNet-B0")
    print("Device:", DEVICE)
    print("Classes:", len(CLASS_NAMES))
    print("Model loaded:", MODEL is not None)

    if MODEL is None:

        print("MODEL ERROR:", MODEL_ERROR)

        print("Place the checkpoint at:", MODEL_PATH)

    print("URL: http://127.0.0.1:5000")

    print("====================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
