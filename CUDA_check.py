import torch

def check_cuda():
    print("Checking CUDA setup...")

    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of available devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
        # Additional check - Allocate a tensor on GPU and perform a simple operation
        try:
            x = torch.randn(3, 3).cuda()
            print("Tensor allocation on GPU successful.")
            y = x * 2
            print("Simple tensor operation successful.")
        except Exception as e:
            print(f"Tensor operation failed: {e}")
    else:
        print("CUDA is not available.")
        
if __name__ == "__main__":
    check_cuda()
