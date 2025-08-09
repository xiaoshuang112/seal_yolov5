import torch
from params_yolo import get_default_config

def debug_model_shapes():
    config = get_default_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = config.model_dict[config.model_name]().to(device)
    model.train()
    
    # 创建假数据
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    print("Testing model output shapes...")
    with torch.no_grad():
        output = model(dummy_input)
        
        if isinstance(output, list) and len(output) >= 3:
            if isinstance(output[2], list):
                preds = output[2]
            else:
                preds = [output[2]]
        
        print("Prediction tensor shapes:")
        for i, pred in enumerate(preds):
            print(f"preds[{i}].shape = {pred.shape} (dimensions: {len(pred.shape)})")

if __name__ == "__main__":
    debug_model_shapes()
