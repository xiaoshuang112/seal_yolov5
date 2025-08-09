import os
import cv2
from pathlib import Path

def fix_label_normalization():
    """修复标签坐标归一化问题"""
    print("🔧 开始修复标签坐标归一化...")
    
    label_dir = Path("data/labels")
    image_dir = Path("data/images")
    
    if not label_dir.exists() or not image_dir.exists():
        print("❌ 标签或图片目录不存在")
        return False
    
    label_files = list(label_dir.glob("*.txt"))
    fixed_count = 0
    error_count = 0
    
    for label_file in label_files:
        try:
            # 获取对应的图片文件
            img_name = label_file.stem + ".jpg"
            img_path = image_dir / img_name
            
            if not img_path.exists():
                print(f"⚠️ 找不到对应图片: {img_name}")
                continue
            
            # 读取图片尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ 无法读取图片: {img_name}")
                continue
                
            img_h, img_w = img.shape[:2]
            
            # 读取并修复标签
            new_lines = []
            need_fix = False
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        
                        # 检查是否需要归一化
                        if x > 1.0 or y > 1.0 or w > 1.0 or h > 1.0:
                            # 归一化坐标 (假设原坐标是中心点+宽高格式)
                            x_norm = x / img_w
                            y_norm = y / img_h  
                            w_norm = w / img_w
                            h_norm = h / img_h
                            
                            # 确保在[0,1]范围内
                            x_norm = max(0, min(1, x_norm))
                            y_norm = max(0, min(1, y_norm))
                            w_norm = max(0, min(1, w_norm))
                            h_norm = max(0, min(1, h_norm))
                            
                            new_line = f"{int(cls)} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                            need_fix = True
                        else:
                            new_line = line
                            
                        new_lines.append(new_line)
            
            # 如果需要修复，写回文件
            if need_fix:
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
                
        except Exception as e:
            print(f"❌ 处理标签文件 {label_file.name} 失败: {e}")
            error_count += 1
    
    print(f"✅ 标签修复完成: 修复了 {fixed_count} 个文件, 错误 {error_count} 个")
    return fixed_count > 0

def verify_label_format():
    """验证标签格式是否正确"""
    print("\n🔍 验证标签格式...")
    
    label_dir = Path("data/labels")
    label_files = list(label_dir.glob("*.txt"))[:5]  # 检查前5个文件
    
    all_good = True
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        if x > 1.0 or y > 1.0 or w > 1.0 or h > 1.0:
                            print(f"❌ {label_file.name}:{line_num} 仍有异常坐标: {[x, y, w, h]}")
                            all_good = False
                        elif x < 0 or y < 0 or w <= 0 or h <= 0:
                            print(f"❌ {label_file.name}:{line_num} 坐标范围异常: {[x, y, w, h]}")
                            all_good = False
        except Exception as e:
            print(f"❌ 验证文件 {label_file.name} 失败: {e}")
            all_good = False
    
    if all_good:
        print("✅ 标签格式验证通过")
    else:
        print("❌ 标签格式仍有问题")
    
    return all_good

if __name__ == "__main__":
    print("🚀 开始修复标签归一化问题...")
    print("="*50)
    
    # 1. 修复标签归一化
    success = fix_label_normalization()
    
    if success:
        # 2. 验证修复结果
        verify_label_format()
        
        print("\n🎯 修复完成！")
        print("📋 修复后的标签格式: class x_center y_center width height (归一化到[0,1])")
        print("🚀 现在可以重新训练模型了")
    else:
        print("\n❌ 修复失败，请检查文件路径和权限")
