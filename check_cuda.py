import torch

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA capable devices found. PyTorch will use the CPU.")

if __name__ == "__main__":
    check_cuda()