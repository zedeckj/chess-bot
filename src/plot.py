import matplotlib.pyplot as plt
import json
import numpy as np

def retrieve_losses(model_name : str) -> tuple[list[list], list]:
    with open(f"losses/{model_name}.json", "r") as f:
        losses_dict = json.loads(f.read())
    testing_losses = losses_dict["testing"]
    training_losses = losses_dict["training"]
    return training_losses, testing_losses

def plot_testing(modelname : str, data : list[float], training_round : int):

    plt.figure(figsize=(12, 6))
    plt.plot(data, marker='o')
    if training_round == 0:
        plt.title(f'Testing loss for {modelname}\n')
    else:
        plt.title(f'Testing loss for {modelname} during round {training_round}\n')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{modelname}-{training_round}-testing.png")


def plot_training(modelname : str, learning_rate : str, batch_size : str, data : list[list], training_round : int, extra_info = ""):

    data = [[value for value in row if value != None] for row in data]

    flattened_data = [value for row in data for value in row]
    row_lengths = [len(row) for row in data]
    row_boundaries = [sum(row_lengths[:i+1]) - 1 for i in range(len(row_lengths) - 1)]


    plt.figure(figsize=(12, 6))
    plt.plot(flattened_data, marker='o')


    for i, boundary in enumerate(row_boundaries):
        ymin, ymax = plt.ylim()

        label_y_position = ymin + (ymax - ymin) * 0.9
        plt.axvline(x=boundary + 0.5, color='red', linestyle='--',)
        plt.text(boundary + 0.5, label_y_position, f'Epoch {i + 1}',
             color='red', rotation=90, ha='right', va='center')


    if training_round == 0:
        plt.title(f'Training loss for {modelname}\nwith lr = {learning_rate}, batch size = {batch_size}\n{extra_info}')
    else:
        plt.title(f'Training loss for {modelname} during round {training_round}\nwith lr = {learning_rate}, batch size = {batch_size}\n{extra_info}')
    plt.xlabel('Training Sets Used (~500k Boards Each)')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{modelname}-{training_round}.png")

def make_plots():
    training_losses1, testing_losses1 = retrieve_losses("autoencoder-backup")
    training_losses2, testing_losses2 = retrieve_losses("autoencoder2")
    training_losses3, testing_losses3 = retrieve_losses("autoencoder3")
    training_losses4, testing_losses4 = retrieve_losses("autoencoder5")
    training_losses5, testing_losses5 = retrieve_losses("autoencoder6")
    training_losses_c, testing_losses_c = retrieve_losses("autoencoder-complex")
    plot_training("autoencoder256", "1-e4", "2048", training_losses1, 1)
    plot_training("autoencoder256", "1-e5", "2048", training_losses2, 2)
    plot_training("autoencoder256", "1-e3", "32", training_losses3, 3)
    plot_training("autoencoder256", "1-e5", "2048", training_losses4, 4, "With the inclusion of `piece_count_limit_loss`")
    plot_training("autoencoder256", "1-e5", "2048", training_losses5, 5, "With the inclusion of the updated `piece_count_loss_fn`")
    plot_training("autoencoder128", "1-e5", "2048", training_losses_c, 0, "Included `piece_count_loss_fn`")
    
    plot_testing("autoencoder256", testing_losses1, 1)
    plot_testing("autoencoder256", testing_losses2, 2)
    plot_testing("autoencoder256", testing_losses3, 3)
    plot_testing("autoencoder256", testing_losses4, 4)
    plot_testing("autoencoder256", testing_losses5, 5)
    plot_testing("autoencoder128", testing_losses_c, 0)


if __name__ == "__main__":
    make_plots()