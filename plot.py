import matplotlib.pyplot as plt


def plot_train_losses(filename: str, label: str, to_save: bool = False):
    iter_nums, losses = _get_iter_nums_and_losses(filename)
    plt.plot(iter_nums, losses, label=label)
    plt.legend()
    if to_save:  # last one to plot
        plt.xlabel("Iteration Number")
        plt.ylabel("Training Loss")
        plt.savefig(f"plots/plot_owt_train_losses.png", bbox_inches="tight", dpi=400)


def _get_iter_nums_and_losses(filename: str):
    iter_nums = []
    losses = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if not line.startswith("iter "):
                continue
            stripped_line = line.strip()
            line_splits = stripped_line.split(",")
            iter_loss_raw = line_splits[0]
            iter_loss_raw_splits = iter_loss_raw.split(":")
            assert len(iter_loss_raw_splits) == 2, f"{iter_loss_raw_splits}"
            iter_num = int(iter_loss_raw_splits[0][5:])
            loss = float(iter_loss_raw_splits[1][5:])
            iter_nums.append(iter_num)
            losses.append(loss)
    return iter_nums, losses


if __name__ == "__main__":
    plot_train_losses("log_ddp_gpt2_8gpus_96bsz_40gradacc.out", "gpt2")
    plot_train_losses("log_ddp_gpt2medium_8gpus_96bsz_40gradacc.out", "gpt2-medium")
    plot_train_losses(
        "log_softmoe_gpt2_8gpus_32bsz_4gradacc.out", "gpt-softmoe", to_save=True
    )
