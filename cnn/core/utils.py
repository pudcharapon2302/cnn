import base64, io
from PIL import Image, ImageOps
import torch

def base64png_to_tensor28x28(b64):
    # รับรูปแบบ data:image/png;base64,....
    if "," in b64: b64 = b64.split(",",1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    # MNIST เป็น "ขาวบนพื้นดำ" แต่วาดปกติคือ "ดำบนพื้นขาว" → invert
    img = ImageOps.invert(img)
    # padding ให้จตุรัสแล้ว resize 28x28
    w,h = img.size; side=max(w,h)
    sq = Image.new("L",(side,side),0); sq.paste(img,((side-w)//2,(side-h)//2))
    img28 = sq.resize((28,28), Image.LANCZOS)
    # ToTensor (เหมือน transforms.ToTensor())
    x = torch.from_numpy(255 - torch.ByteTensor(torch.ByteStorage.from_buffer(img28.tobytes())).numpy()).float()
    x = torch.tensor(list(img28.getdata()), dtype=torch.float32).reshape(1,28,28) / 255.0
    return x.unsqueeze(0)  # [1,1,28,28]
