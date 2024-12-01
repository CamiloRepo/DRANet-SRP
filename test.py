from __future__ import print_function
from args import get_args
from trainer import Trainer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)

    # Initialize an empty list to store accuracy values and conversion names
    accuracy_over_time = []
    conversions = []

    # Set up the trainer for testing
    trainer.set_default()
    trainer.set_networks()
    trainer.load_networks(opt.load_step)

    # Run the evaluation for all conversions
    for cv in trainer.test_converts:
        acc = trainer.eval(cv)
        if acc != 0:  # Append only if accuracy is returned
            accuracy_over_time.append(acc)
            conversions.append(cv)

    # Check if the accuracy list is empty
    if len(accuracy_over_time) == 0:
        print("No accuracy values recorded. Please check the eval method and ensure accuracies are being returned properly.")

    # Plot the accuracy using a bar plot
    if accuracy_over_time:
        plt.figure(figsize=(10, 6))
        plt.bar(conversions, accuracy_over_time, color='b')
        plt.xlabel('Conversions')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy for Different Test Conversions')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('accuracy_bar_plot.png')
        plt.show()
