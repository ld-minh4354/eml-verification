import onnx, os
m = onnx.load(os.path.join("models", "MNIST", "baseline", "resnet18-MNIST-10.onnx"))
print([n.op_type for n in m.graph.node])
