import torch, sys, os
src = sys.argv[1] if len(sys.argv)>1 else "checkpoints/production/G_AB_epoch02_ts.pt"
out = sys.argv[2] if len(sys.argv)>2 else src.replace(".pt","_q.pt")
m = torch.jit.load(src, map_location='cpu')
mq = torch.quantization.quantize_dynamic(m, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
torch.jit.save(mq, out)
print("Saved quantized:", out)