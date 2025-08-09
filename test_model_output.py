import torch
from params_yolo import get_default_config

def test_model_output():
    config = get_default_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = config.model_dict[config.model_name]().to(device)
    model.train()  # 设置为训练模式
    
    # 创建假数据
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    print("Testing model output...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output type: {type(output)}")
        
        if isinstance(output, list):
            print(f"Number of outputs: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"Output {i} shape: {out.shape}")
                else:
                    print(f"Output {i} type: {type(out)}")
                    if isinstance(out, list):
                        print(f"  Nested list length: {len(out)}")
                        for j, nested_out in enumerate(out):
                            if hasattr(nested_out, 'shape'):
                                print(f"    Output {i}[{j}] shape: {nested_out.shape}")
                            else:
                                print(f"    Output {i}[{j}] type: {type(nested_out)}")
        elif isinstance(output, tuple):
            print(f"Tuple length: {len(output)}")
            for i, out in enumerate(output):
                if hasattr(out, 'shape'):
                    print(f"Tuple element {i} shape: {out.shape}")
                else:
                    print(f"Tuple element {i} type: {type(out)}")
        else:
            if hasattr(output, 'shape'):
                print(f"Single output shape: {output.shape}")
            else:
                print(f"Single output type: {type(output)}")

if __name__ == "__main__":
    test_model_output()
