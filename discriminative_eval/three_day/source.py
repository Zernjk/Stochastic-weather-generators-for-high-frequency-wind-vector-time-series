from dataset import DatasetGenerator
from model import WindModelLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import datetime
from torch.optim.lr_scheduler import LambdaLR
import random
import argparse
import matplotlib.pyplot as plt
import tempfile

def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# Use it at the beginning of your script
set_seed(42)  # You can choose any seed value


def train_model(timestamp, model, train_dataset, validation_dataset, test_dataset, epochs, batch_size, learning_rate, device, save_every, gen_name, seed, filter_type='none', log_file=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_batch_size = batch_size * 4  # Larger batch size for evaluation
    validation_loader = DataLoader(validation_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    filter_suffix = f"_{filter_type}" if filter_type != 'none' else ""
    os.makedirs(f'model_checkpoints/{gen_name}_seed{seed}{filter_suffix}', exist_ok=True)

    train_losses = []
    train_disc_scores = []
    validation_disc_scores = []
    test_disc_scores = []

    # Track best validation discriminative score and corresponding test score
    best_val_disc_score = -1.0
    best_val_epoch = -1
    best_val_test_disc_score = -1.0
    best_test_predictions = []
    best_test_real_accuracy = -1.0
    best_test_gen_accuracy = -1.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # Convert labels to float
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))

        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']

        log_message = f'Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_losses[-1]:.4f}'
        print(log_message)
        if log_file:
            log_file.write(log_message + '\n')
            log_file.flush()

        # Save model every save_every epochs
        if (epoch + 1) % save_every == 0:
            model.eval()

            # Training discriminative score (on training data)
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    outputs = model(inputs)
                    predicted = (outputs.data > 0.5).float()
                    train_total += labels.size(0)
                    train_correct += (predicted.squeeze() == labels).sum().item()

            train_accuracy = train_correct / train_total
            train_disc_score = abs(0.5 - train_accuracy)
            train_disc_scores.append(train_disc_score)

            # Validation phase
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    outputs = model(inputs)
                    predicted = (outputs.data > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted.squeeze() == labels).sum().item()

            val_accuracy = val_correct / val_total
            val_disc_score = abs(0.5 - val_accuracy)
            validation_disc_scores.append(val_disc_score)

            # Test phase
            test_correct = 0
            test_total = 0
            all_test_predictions = []
            all_test_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).float()
                    outputs = model(inputs)
                    predicted = (outputs.data > 0.5).float()
                    test_total += labels.size(0)
                    test_correct += (predicted.squeeze() == labels).sum().item()
                    all_test_predictions.extend(predicted.squeeze().cpu().numpy().astype(int).tolist())
                    all_test_labels.extend(labels.cpu().numpy().astype(int).tolist())

            test_accuracy = test_correct / test_total
            test_disc_score = abs(0.5 - test_accuracy)
            test_disc_scores.append(test_disc_score)

            # Calculate first half (real data) and second half (generated data) accuracy
            n_test = len(all_test_predictions)
            n_real = n_test // 2
            real_predictions = all_test_predictions[:n_real]
            real_labels = all_test_labels[:n_real]
            gen_predictions = all_test_predictions[n_real:]
            gen_labels = all_test_labels[n_real:]
            assert real_labels == [1] * n_real and gen_labels == [0] * n_real

            real_correct = sum(p == l for p, l in zip(real_predictions, real_labels))
            gen_correct = sum(p == l for p, l in zip(gen_predictions, gen_labels))
            real_accuracy = real_correct / len(real_predictions) if real_predictions else 0
            gen_accuracy = gen_correct / len(gen_predictions) if gen_predictions else 0

            # Update best validation score if current is higher
            if val_disc_score > best_val_disc_score:
                best_val_disc_score = val_disc_score
                best_val_epoch = epoch + 1
                best_val_test_disc_score = test_disc_score
                best_test_predictions = all_test_predictions.copy()
                best_test_real_accuracy = real_accuracy
                best_test_gen_accuracy = gen_accuracy

            # Log current and best scores
            log_message = f'Train: Accuracy={train_accuracy:.4f}, Disc Score={train_disc_score:.4f}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')

            log_message = f'Validation: Accuracy={val_accuracy:.4f}, Disc Score={val_disc_score:.4f}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')

            log_message = f'Test: Accuracy={test_accuracy:.4f}, Disc Score={test_disc_score:.4f}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')

            log_message = f'Test Breakdown: Real Accuracy={real_accuracy:.4f}, Generated Accuracy={gen_accuracy:.4f}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')

            log_message = f'Best Val Disc Score={best_val_disc_score:.4f} at Epoch {best_val_epoch}, Corresponding Test Disc Score={best_val_test_disc_score:.4f}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')
                log_file.flush()

            checkpoint_path = f'model_checkpoints/{gen_name}_seed{seed}{filter_suffix}/lstm_classifier_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            log_message = f'Epoch {epoch+1}: Model saved to {checkpoint_path}'
            print(log_message)
            if log_file:
                log_file.write(log_message + '\n')
                log_file.flush()

    # Log final metrics
    if validation_disc_scores and test_disc_scores:
        log_message = f'Final Metrics:\n'
        log_message += f'  Final Train Disc Score: {train_disc_scores[-1]:.4f}\n'
        log_message += f'  Final Validation Disc Score: {validation_disc_scores[-1]:.4f}\n'
        log_message += f'  Final Test Disc Score: {test_disc_scores[-1]:.4f}\n'
        log_message += f'  Best Val Disc Score: {best_val_disc_score:.4f} at Epoch {best_val_epoch}\n'
        log_message += f'  Corresponding Test Disc Score: {best_val_test_disc_score:.4f}\n'
        log_message += f'  Corresponding Test Real Accuracy: {best_test_real_accuracy:.4f}\n'
        log_message += f'  Corresponding Test Generated Accuracy: {best_test_gen_accuracy:.4f}'
        print(log_message)
        if log_file:
            log_file.write(log_message + '\n')
            log_file.flush()

        # Save test predictions at best validation epoch
        predictions_path = f'model_checkpoints/{gen_name}_seed{seed}{filter_suffix}/best_val_test_predictions.npy'
        np.save(predictions_path, np.array(best_test_predictions))
        log_message = f'Test predictions (at best val epoch) saved to {predictions_path}'
        print(log_message)
        if log_file:
            log_file.write(log_message + '\n')
            log_file.flush()

    return train_losses, train_disc_scores, validation_disc_scores, test_disc_scores, best_val_disc_score, best_val_epoch, best_val_test_disc_score, best_test_predictions, best_test_real_accuracy, best_test_gen_accuracy


def plot_training_metrics(train_losses, train_disc_scores, validation_disc_scores, test_disc_scores, save_every, gen_name, random_seed, filter_type='none'):
    """Plot training loss and discriminative scores, save to temp file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot discriminative scores
    eval_epochs = [i * save_every for i in range(1, len(validation_disc_scores) + 1)]
    ax2.plot(eval_epochs, train_disc_scores, 'b-', label='Train Disc Score')
    ax2.plot(eval_epochs, validation_disc_scores, 'g-', label='Validation Disc Score')
    ax2.plot(eval_epochs, test_disc_scores, 'r-', label='Test Disc Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discriminative Score')
    ax2.set_title(f'Discriminative Scores (filter: {filter_type})')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Save to temp file with filter_type in filename
    filter_suffix = f"_{filter_type}" if filter_type != 'none' else ""
    temp_file = f"./figures/{gen_name}_{random_seed}{filter_suffix}.png"
    plt.savefig(temp_file, dpi=150)
    plt.close()

    print(f"Training plot saved to {temp_file}")
    return temp_file


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train wind model with specified parameters')
    parser.add_argument('--real_name', type=str, default='training',
                      help='Name of the real dataset (default: training)')
    parser.add_argument('--gen_name', type=str, default='validation',
                      help='Name of the generated dataset (default: validation)')
    parser.add_argument('--dataset_seed', type=int, required=True,
                      help='Random seed for dataset generation')
    parser.add_argument('--cuda', type=str, default='3',
                      help='CUDA device number to use (default: 3), use "cpu" to force CPU usage')
    parser.add_argument('--filter_type', type=str, default='none', choices=['none', 'low', 'high'],
                      help='Frequency filter type: none (default), low (low-pass), high (high-pass)')
    parser.add_argument('--cutoff_freq', type=int, default=48,
                      help='Cutoff frequency in cycles/day (default: 48 = 30-min period)')

    args = parser.parse_args()

    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_suffix = f"_{args.filter_type}" if args.filter_type != 'none' else ""
    log_filename = f'./log/training_log_{args.gen_name}_seed{args.dataset_seed}{filter_suffix}.txt'
    # log_filename = f'./log/training_log_{timestamp}.txt'
    log_file = open(log_filename, 'w')
    
    # Log and print arguments
    args_dict = vars(args)
    args_message = "Arguments used:"
    print(args_message)
    log_file.write(args_message + '\n')
    for arg, value in args_dict.items():
        arg_message = f"  {arg}: {value}"
        print(arg_message)
        log_file.write(arg_message + '\n')
    print()  # Empty line for better readability
    log_file.write('\n')
    
    # Set device based on argument
    if args.cuda.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    log_message = f"Using device: {device}"
    print(log_message)
    log_file.write(log_message + '\n')
    
    dataset_generator = DatasetGenerator(
        real_name=args.real_name,
        gen_name=args.gen_name,
        random_seed=args.dataset_seed,
        # filter_type=args.filter_type,
        # cutoff_freq=args.cutoff_freq,
    )
    train_dataset = dataset_generator.get_dataset("training")
    validation_dataset = dataset_generator.get_dataset("validation")
    test_dataset = dataset_generator.get_dataset("testing")
    
    # print the length of the train and test datasets to log file
    log_message = f"Length of train dataset: {len(train_dataset)}"
    print(log_message)
    log_file.write(log_message + '\n')
    log_message = f"Length of validation dataset: {len(validation_dataset)}"
    print(log_message)
    log_file.write(log_message + '\n')
    log_message = f"Length of test dataset: {len(test_dataset)}"
    print(log_message)
    log_file.write(log_message + '\n')


    # Initialize the model and move it to GPU
    model = WindModelLSTM()
    # model = WindModelFNN()
    model = model.to(device)

    # Log training parameters
    learning_rate = 0.001
    epochs = 1000
    batch_size = 64
    save_every = 5
    log_message = f"Training with learning_rate={learning_rate}, epochs={epochs}"
    print(log_message)
    log_file.write(log_message + '\n')

    # Log filter settings
    if args.filter_type != 'none':
        filter_period = 1440 // args.cutoff_freq
        log_message = f"Filter: {args.filter_type}-pass, cutoff={args.cutoff_freq} cycles/day (period={filter_period} min)"
    else:
        log_message = "Filter: none (using full frequency spectrum)"
    print(log_message)
    log_file.write(log_message + '\n')

    train_losses, train_disc_scores, validation_disc_scores, test_disc_scores, best_val_disc_score, best_val_epoch, best_val_test_disc_score, best_test_predictions, best_test_real_accuracy, best_test_gen_accuracy = train_model(
        timestamp, model, train_dataset, validation_dataset, test_dataset,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, device=device, save_every=save_every, gen_name=args.gen_name, seed=args.dataset_seed,
        filter_type=args.filter_type, log_file=log_file,
    )

    # Plot training metrics
    plot_training_metrics(train_losses, train_disc_scores, validation_disc_scores, test_disc_scores, save_every, gen_name=args.gen_name, random_seed=args.dataset_seed, filter_type=args.filter_type)

    # Close the log file
    log_file.close()
    print(f"Training log saved to {log_filename}")


if __name__ == "__main__":
    main()
    