import importlib
import sys
from pkg_resources import parse_version

# 定义项目要求的依赖清单 (根据你的 environment.yml)
REQUIRED_PACKAGES = {
    # 核心数据处理
    "numpy": "1.21.0",
    "pandas": "1.3.0",
    "scipy": "1.7.0",
    "sklearn": "1.0.0",  # 对应 scikit-learn
    
    # 梯度提升
    "xgboost": "1.5.0",
    "lightgbm": "3.3.0",
    
    # 深度学习与图计算
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "torch_geometric": "2.3.0",
    "torch_scatter": "2.1.0",
    "torch_sparse": "0.6.0",
    
    # 化学信息学与 NLP
    "rdkit": "2022.09.1",
    "transformers": "4.20.0",
    
    # 可视化与工具
    "matplotlib": "3.5.0",
    "seaborn": "0.11.0",
    "yaml": "6.0",       # 对应 pyyaml
    "tqdm": "4.62.0",
    "shap": "0.41.0",
    "joblib": "1.1.0"
}

def check_dependencies():
    print(f"{'Package':<20} | {'Required':<12} | {'Installed':<12} | {'Status'}")
    print("-" * 65)
    
    all_passed = True
    
    for pkg, min_version in REQUIRED_PACKAGES.items():
        # 处理包名映射
        import_name = pkg
        try:
            module = importlib.import_module(import_name)
            
            # 获取版本号的不同方式
            if pkg == "rdkit":
                installed_version = module.rdBase.rdkitVersion
            elif pkg == "yaml":
                installed_version = module.__version__
            else:
                installed_version = getattr(module, "__version__", "Unknown")
            
            # 版本对比
            if installed_version != "Unknown" and parse_version(installed_version) >= parse_version(min_version):
                status = "✅ OK"
            else:
                status = "⚠️  Low Version"
                all_passed = False
                
            print(f"{pkg:<20} | {min_version:<12} | {installed_version:<12} | {status}")
            
        except ImportError:
            print(f"{pkg:<20} | {min_version:<12} | {'Missing':<12} | ❌ Missing")
            all_passed = False

    print("-" * 65)
    
    # 特殊检查：CUDA 支持
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA GPU 加速: {'✅ 可用 (' + torch.cuda.get_device_name(0) + ')' if cuda_available else '❌ 不可用'}")
    except:
        pass

    if all_passed:
        print("\n🎉 恭喜！当前环境完全满足 QSAR Modeling Pipeline 的运行要求。")
    else:
        print("\n💡 提示：部分依赖缺失或版本过低，请根据上方列表进行安装/更新。")

if __name__ == "__main__":
    check_dependencies()