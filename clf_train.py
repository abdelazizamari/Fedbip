import torch
from config import parse_args

import torchvision.models as models
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt   # bibliothèque pour tracer des graphes

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args()

tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)

if args.dataset == 'domainnet':
    from domainnet_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'pacs':
    from pacs_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'officehome':
    from officehome_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'bloodmnist':
    from bloodmnist_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'dermamnist':
    from dermamnist_data import get_dataloader, get_dataloader_domain
elif args.dataset == 'ucm':
    from ucm_data import get_dataloader, get_dataloader_domain

def tokenize_captions(examples, is_train=False):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError("Caption format error")
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    return inputs.input_ids
        
def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [example[1] for example in examples]
    padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    domain_ids = torch.tensor([example[2] for example in examples])
    class_ids = torch.tensor([example[3] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
        "domain_ids": domain_ids,
        "class_ids": class_ids,
    }

def train(seed, train_setting):
    model = models.resnet18(pretrained=args.pretrained)
    print(f"Training with seed {seed} and setting {train_setting}")
    setup_seed(seed)
    categories = args.categories

    # listes pour stocker les résultats de performance
    epoch_list = []   
    acc_list = []    

    num_shots = int(train_setting.split("_")[-1])

    train_dataloader = get_dataloader(
        args, args.train_batch_size, None,
        train_setting, tokenize_captions, collate_fn, num_shot=num_shots)

    test_dataloader = get_dataloader(
        args, args.train_batch_size, None,
        'test', tokenize_captions, collate_fn, num_shot=-1)

    num_epochs = 50
    num_classes = len(categories)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(batch['pixel_values'].to(device))
            labels = batch['class_ids'].to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # ÉVALUATION DU MODÈLE
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0

            for batch in test_dataloader:
                inputs = batch["pixel_values"].to(device)
                outputs = model(inputs)
                labels = batch['class_ids'].to(device)
                preds = torch.argmax(outputs, dim=1)

                total_correct += preds.eq(labels).sum().item()
                total_samples += inputs.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}: Accuracy = {round(accuracy, 3)}")

        # sauvegarde des résultats pour le graphe
        epoch_list.append(epoch + 1)
        acc_list.append(accuracy)

    # création du graphe
    plt.figure()   # crée une nouvelle figure
    plt.plot(epoch_list, acc_list, marker='o')   # trace la courbe accuracy vs epoch
    plt.xlabel("Epoch")   # nom de l'axe horizontal
    plt.ylabel("Accuracy")   # nom de l'axe vertical
    plt.title(f"Accuracy vs Epochs\nSeed={seed}, Setting={train_setting}")  # titre du graphe
    plt.grid(True)   # ajoute une grille
    plt.savefig(f"accuracy_{train_setting}_seed{seed}.png")
    plt.show()

# condition principale Python
if __name__ == "__main__":   
    for seed in [0, 1, 2]:
        if not isinstance(args.train_type, list):
            args.train_type = [args.train_type]
        for train_setting in args.train_type:
            print(train_setting)
            train(seed, train_setting)