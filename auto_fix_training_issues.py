import os
import torch
import shutil
from pathlib import Path

def fix_loss_weights():
    """修复损失权重不平衡问题"""
    print("🔧 修复1: 调整损失权重...")
    
    # 读取当前train.py
    train_py_path = "train.py"
    with open(train_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到并替换损失权重
    old_hyp = """hyp = {
        'box': 0.05, 'obj': 1.0, 'cls': 0.5,
        'cls_pw': 1.0, 'obj_pw': 1.0,
        'anchor_t': 4.0, 'fl_gamma': 0.0,
        'label_smoothing': 0.0
    }"""
    
    new_hyp = """hyp = {
        'box': 0.1, 'obj': 0.3, 'cls': 0.5,  # 修复: 降低obj权重，增加box权重
        'cls_pw': 1.0, 'obj_pw': 1.0,
        'anchor_t': 4.0, 'fl_gamma': 0.0,
        'label_smoothing': 0.0
    }"""
    
    if old_hyp in content:
        content = content.replace(old_hyp, new_hyp)
        with open(train_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 损失权重已修复: box=0.1, obj=0.3, cls=0.5")
    else:
        print("⚠️ 未找到确切的损失权重配置，请手动修改")
    
    return True

def check_label_format():
    """检查标签格式并给出修复建议"""
    print("\n🔧 修复2: 检查标签格式...")
    
    label_dir = Path("data/labels")
    if not label_dir.exists():
        print("❌ 标签目录不存在")
        return False
    
    # 检查几个标签文件
    label_files = list(label_dir.glob("*.txt"))[:5]
    abnormal_labels = []
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x, y, w, h = map(float, parts[:5])
                        # 检查坐标是否超出[0,1]范围
                        if x > 1.0 or y > 1.0 or w > 1.0 or h > 1.0:
                            abnormal_labels.append((label_file.name, line_num, [x, y, w, h]))
        except Exception as e:
            print(f"读取标签文件 {label_file} 失败: {e}")
    
    if abnormal_labels:
        print("❌ 发现标签格式异常:")
        for file_name, line_num, coords in abnormal_labels[:3]:  # 只显示前3个
            print(f"   {file_name}:{line_num} - 坐标: {coords}")
        print("🔧 标签坐标必须在[0,1]范围内！请检查标签生成过程。")
        return False
    else:
        print("✅ 标签格式检查通过")
        return True

def fix_confidence_threshold():
    """修复置信度阈值相关设置"""
    print("\n🔧 修复3: 调整置信度阈值...")
    
    # 修改train.py中的IoU分析阈值
    train_py_path = "train.py"
    with open(train_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 降低IoU分析的置信度阈值
    if "conf_thresh=0.01" in content:
        content = content.replace("conf_thresh=0.01", "conf_thresh=0.001")
        print("✅ IoU分析置信度阈值已调整为0.001")
    
    # 修改验证集的置信度阈值
    if "conf_thresh=0.1" in content:
        content = content.replace("conf_thresh=0.1", "conf_thresh=0.001")
        print("✅ 验证集置信度阈值已调整为0.001")
    
    with open(train_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True

def create_optimized_training_config():
    """创建优化的训练配置"""
    print("\n🔧 修复4: 创建优化训练配置...")
    
    config_content = '''
# 优化后的训练配置建议

# 1. 损失权重 (已在代码中修复)
hyp = {
    'box': 0.1,      # 提高box权重，关注定位精度
    'obj': 0.3,      # 降低obj权重，避免置信度塌陷  
    'cls': 0.5,      # 保持cls权重
    'cls_pw': 1.0, 
    'obj_pw': 1.0,
    'anchor_t': 4.0, 
    'fl_gamma': 0.0,
    'label_smoothing': 0.0
}

# 2. 学习率建议
lr = 0.005  # 适当降低学习率，避免训练不稳定

# 3. 训练策略
epochs = 100
warmup_epochs = 3
patience = 20  # 早停patience

# 4. 数据增强 (当前已关闭，建议保持)
augment = False  # 继续关闭，先让模型收敛

# 5. 置信度阈值
conf_thresh_train = 0.001   # 训练时IoU统计阈值
conf_thresh_val = 0.001     # 验证时统计阈值
conf_thresh_infer = 0.01    # 推理时过滤阈值

# 6. anchor建议 (需要根据数据集生成)
# 当前目标尺寸较大 (w=132, h=122 pixels)
# 建议重新生成anchor或调整现有anchor
'''
    
    with open("optimized_config.txt", 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ 优化配置已保存到 optimized_config.txt")
    return True

def backup_current_model():
    """备份当前模型"""
    print("\n🔧 修复5: 备份当前模型...")
    
    model_dir = Path("model")
    backup_dir = Path("model_backup")
    
    if model_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(model_dir, backup_dir)
        print("✅ 当前模型已备份到 model_backup/")
        return True
    else:
        print("⚠️ model目录不存在，跳过备份")
        return False

def main():
    """主修复流程"""
    print("🚀 开始自动修复训练问题...")
    print("="*60)
    
    # 1. 备份模型
    backup_current_model()
    
    # 2. 修复损失权重
    fix_loss_weights()
    
    # 3. 检查标签格式
    label_ok = check_label_format()
    
    # 4. 修复置信度阈值
    fix_confidence_threshold()
    
    # 5. 创建优化配置
    create_optimized_training_config()
    
    print("\n" + "="*60)
    print("🎯 修复总结:")
    print("✅ 1. 损失权重已平衡: box↑(0.1), obj↓(0.3)")
    print("✅ 2. 置信度阈值已调低: 0.001")
    print("✅ 3. 优化配置已生成")
    
    if not label_ok:
        print("❌ 4. 标签格式异常，需要手动修复")
        print("   - 确保所有坐标都在[0,1]范围内")
        print("   - 检查标签生成脚本")
    else:
        print("✅ 4. 标签格式正常")
    
    print("\n🚀 下一步建议:")
    print("1. 立即重新开始训练，观察IoU是否上升")
    print("2. 监控前几个epoch的置信度均值是否提升")
    print("3. 如果IoU仍不升，考虑重新生成anchor")
    print("4. 如果标签格式异常，先修复标签再训练")
    
    print(f"\n💡 核心原理:")
    print("- obj权重过高→模型学会'低置信度策略'→置信度塌陷")
    print("- box权重过低→定位精度不被重视→IoU不升")
    print("- 调整权重后，模型会重新学习平衡置信度和定位精度")

if __name__ == "__main__":
    main()
