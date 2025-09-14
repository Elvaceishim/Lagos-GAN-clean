import torch, sys
model_path = sys.argv[1] if len(sys.argv)>1 else "checkpoints/production/G_AB_epoch02_ts.pt"
onnx_out = sys.argv[2] if len(sys.argv)>2 else model_path.replace(".pt",".onnx")
img_size = int(sys.argv[3]) if len(sys.argv)>3 else 64
m = torch.jit.load(model_path, map_location='cpu')
m.eval()
dummy = torch.randn(1,3,img_size,img_size)
# unwrap script module if needed
try:
    torch.onnx.export(m, dummy, onnx_out, opset_version=13, input_names=['input'], output_names=['output'])
    print("Exported ONNX:", onnx_out)
except Exception as e:
    print("ONNX export failed:", e)
    raise