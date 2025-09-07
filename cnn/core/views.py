from django.shortcuts import render

# Create your views here.
import json, torch, torch.nn.functional as F
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path

from .training import manager, MODEL_PATH
from .mnist_cnn import SimpleCNN
from .utils import base64png_to_tensor28x28

def index(request):
    return render(request, "core/index.html")

@csrf_exempt
def start_training(request):
    started = manager.start(epochs=8, batch=128, lr=1e-3)
    return JsonResponse({"started": started})

def train_stream(request):
    resp = StreamingHttpResponse(manager.stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    return resp

@csrf_exempt
def predict_digit(request):
    if request.method != "POST":
        return JsonResponse({"error":"POST only"}, status=405)

    data = json.loads(request.body.decode("utf-8"))
    b64 = data.get("image")
    if not b64:
        return JsonResponse({"error":"missing image"}, status=400)

    x = base64png_to_tensor28x28(b64)  # [1,1,28,28]

    # โหลดโมเดล
    model = SimpleCNN()
    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1).squeeze(0)
        pred = int(prob.argmax().item())
        conf = float(prob.max().item())
    return JsonResponse({"pred": pred, "conf": round(conf,4)})
