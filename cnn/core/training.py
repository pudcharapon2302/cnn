import json, queue, threading, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from django.conf import settings
from .mnist_cnn import SimpleCNN

MODEL_PATH = Path(settings.BASE_DIR) / "core" / "model.pth"

class TrainingManager:
    def __init__(self):
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._training = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_training(self):
        with self._lock:
            return self._training

    def _put(self, d): self._q.put(json.dumps(d))

    def _loop(self, epochs=2, batch=64, lr=0.001):
        try:
            self._put({"status":"starting", "device": str(self.device)})

            # (ตามสไลด์) โหลด MNIST ด้วย torchvision + ToTensor
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
                ])  # :contentReference[oaicite:2]{index=2}
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(settings.BASE_DIR / "data", train=True, download=True, transform=tfm),
                batch_size=batch, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(settings.BASE_DIR / "data", train=False, transform=tfm),
                batch_size=1000, shuffle=False)

            model = SimpleCNN().to(self.device)
            opt = optim.Adam(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()

            for epoch in range(1, epochs+1):
                model.train()
                seen, loss_sum, correct = 0, 0.0, 0
                t0 = time.time()

                for step, (x, y) in enumerate(train_loader, 1):
                    x, y = x.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    out = model(x)
                    loss = crit(out, y)
                    loss.backward()
                    opt.step()

                    seen += y.size(0)
                    loss_sum += loss.item() * y.size(0)
                    pred = out.argmax(1)
                    correct += (pred == y).sum().item()

                    if step % 100 == 0:
                        self._put({"status":"progress","epoch":epoch,"step":step,
                                   "loss": round(loss_sum/seen,4),
                                   "acc": round(100*correct/seen,2)})

                # val acc (แบบสั้น)
                model.eval()
                val_ok, val_n = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        p = model(x).argmax(1)
                        val_ok += (p==y).sum().item()
                        val_n  += y.size(0)

                self._put({"status":"epoch_done","epoch":epoch,
                           "train_loss": round(loss_sum/seen,4),
                           "train_acc": round(100*correct/seen,2),
                           "val_acc": round(100*val_ok/val_n,2),
                           "sec": round(time.time()-t0,2)})

            torch.save(model.state_dict(), MODEL_PATH)
            self._put({"status":"done","msg":f"saved {MODEL_PATH.name}"})
        except Exception as e:
            self._put({"status":"error","msg":str(e)})
        finally:
            with self._lock: self._training = False
            self._q.put(None)  # ปิดสตรีม

    def start(self, **kw):
        with self._lock:
            if self._training: return False
            self._training = True
        threading.Thread(target=self._loop, kwargs=kw, daemon=True).start()
        return True

    def stream(self):
        yield "data: " + json.dumps({"status":"connected"}) + "\n\n"
        while True:
            item = self._q.get()
            if item is None:
                yield "data: " + json.dumps({"status":"stream_end"}) + "\n\n"
                break
            yield "data: " + item + "\n\n"

manager = TrainingManager()
