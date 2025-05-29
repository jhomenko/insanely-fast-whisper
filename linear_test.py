import torch

if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using device: {device}")

    try:
        # Create a simple tensor
        input_tensor = torch.randn(1, 10, device="cpu")
        print(f"Input tensor on CPU: {input_tensor.shape}")

        # Move the tensor to XPU
        input_tensor_xpu = input_tensor.to(device)
        print(f"Input tensor on XPU: {input_tensor_xpu.shape}")

        # Create a simple linear layer
        linear_layer = torch.nn.Linear(10, 5).to(device)
        print("Linear layer created on XPU")

        # Perform a forward pass through the linear layer
        output_tensor_xpu = linear_layer(input_tensor_xpu)
        print(f"Output tensor on XPU: {output_tensor_xpu.shape}")

        print("Basic linear operation on XPU successful!")

    except Exception as e:
        print(f"Error during basic linear operation on XPU: {e}")

else:
    print("XPU is not available.")

exit()
