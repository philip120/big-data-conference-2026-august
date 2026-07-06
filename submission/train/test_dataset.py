from train.load_dataset import load_matlab_nl_dataset
from train.semantic_adapter import code_to_nodes
from train.dataset import CodeNLDataset


if __name__ == "__main__":
    raw = load_matlab_nl_dataset("train")

    # start tiny on purpose
    dataset = CodeNLDataset(raw[:10], code_to_nodes)

    print("Dataset size:", len(dataset))

    for i in range(min(3, len(dataset))):
        ex = dataset[i]
        print("\n--- EXAMPLE", i, "---")
        print("CODE NODES:")
        for n in ex["code_nodes"][:5]:
            print(" ", n)
        print("\nNL:")
        print(ex["nl_text"][:300])
