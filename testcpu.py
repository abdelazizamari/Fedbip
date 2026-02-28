import torch
print(torch.version.cuda)

print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU détecté :", torch.cuda.get_device_name(0))
    x = torch.tensor([1.0]).to("cuda")
    print("Appareil du tenseur :", x.device)
    print("CUDA disponible :", torch.cuda.is_available())
    print("GPU count :", torch.cuda.device_count())
    print("Nom du GPU :", torch.cuda.get_device_name(0))
else:
    print("Pas de GPU détecté, utilisation CPU")
