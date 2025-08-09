import torch
import onnxruntime
import argparse
from models.yolo import YOLOv5s
import numpy as np

def main(args):
    """
    主函数，用于加载PyTorch模型并将其导出为ONNX格式。
    """
    # --- 1. 加载模型 ---
    print("正在加载 PyTorch 模型...")
    try:
        model = YOLOv5s()
        model.eval()  # 必须设置为评估模式
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # --- 2. 创建虚拟输入 ---
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    print(f"创建虚拟输入: shape={dummy_input.shape}")

    # --- 3. 导出 ONNX ---
    print(f"正在导出 ONNX 模型到: {args.output}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"ONNX 导出成功!")
    except Exception as e:
        print(f"ONNX 导出失败: {e}")
        return

    # --- 4. (可选) 验证 ONNX 模型 ---
    if args.validate:
        print("\n正在验证 ONNX 模型...")
        try:
            session = onnxruntime.InferenceSession(args.output, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            
            # 使用 PyTorch 模型进行推理
            with torch.no_grad():
                outputs_torch = model(dummy_input)

            # 使用 ONNX Runtime 进行推理
            outputs_onnx = session.run(None, {input_name: dummy_input.cpu().numpy()})[0]
            
            # 比较输出
            np.testing.assert_allclose(outputs_torch.cpu().numpy(), outputs_onnx, rtol=1e-3, atol=1e-5)
            print("✅ ONNX 模型验证成功：PyTorch 和 ONNX Runtime 输出一致。")
            print(f"   - ONNX 输出 shape: {outputs_onnx.shape}")
            print(f"   - PyTorch 输出 shape: {outputs_torch.shape}")

        except Exception as e:
            print(f"❌ ONNX 模型验证失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export YOLOv5 model to ONNX format')
    parser.add_argument('--img-size', type=int, default=640, help='input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for dummy input')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--output', type=str, default='yolov5s_stamp.onnx', help='output ONNX file path')
    parser.add_argument('--validate', action='store_true', help='validate the ONNX model after export')
    
    args = parser.parse_args()
    main(args)
