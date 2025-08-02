# cudaが有効化されているか確認
import torch
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")