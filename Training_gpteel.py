from src.the_model.languague_model.language import LanguageModel
from src.SetAndLoader import DataFrameDataset, CommentsDataset
from src.vocab_and_tokenize.tokenizer import vocabulario
from src.the_model.wrapper import AutoregressiveWrapper
from torch.utils.data import random_split
from src.Training import Trainer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import argparse
import torch
import os



def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='cyblx trainer parser')

    parser.add_argument(
    '-ds',                          "--dataset",
                                    metavar = '',
                                    type    = str, 
                                    default = "./dataset/eel_comments.csv",
                                    help    = 'location of the text dataframe with directory.')
    
    parser.add_argument(
    '-ep',                          "--epochs",
                                    metavar = '',
                                    type    = int, 
                                    default = 100,
                                    help    = 'Number of times the model trains on the entire dataset.')
    
    parser.add_argument(
    '-bs',                          "--batch_size",
                                    metavar = '',
                                    type    = int, 
                                    default = 32,
                                    help    = "Defines the number of samples processed together in one batch.")

    parser.add_argument(
    '-ed',                          "--embed_dim",
                                    metavar = '',
                                    type    = int, 
                                    default = 300,
                                    help    = 'Sets the size of the representation vector for each word.')

    parser.add_argument(
    '-ml',                          "--max_length",
                                    metavar = '',
                                    type    = int, 
                                    default = 25,
                                    help    = "Sets the maximum length of input sequences for the model.")
    
    parser.add_argument(
    '-ec',                          "--stop_crit",
                                    metavar = '',
                                    type    = int, 
                                    default = 20,
                                    help    = "Stops training when performance stops improving based on this criteria.")

    parser.add_argument(
    '-lr',                          "--learn_rate",
                                    metavar = '',
                                    type    = float,
                                    default = 0.0001,
                                    help    = "Sets the rate at which the model's weights are updated during training.")

    parser.add_argument(
    '-sd',                          "--save_dir",
                                    metavar = '',
                                    type    = str,
                                    default = './modelos',
                                    help    = "Specifies the directory path where model checkpoints and outputs are saved.")

    parser.add_argument(
    '-nh',                          "--num_heads",
                                    metavar = '',
                                    type    = int,
                                    default = 12,
                                    help    = "Sets the number of attention heads in a multi-head attention mechanism.")
    
    parser.add_argument(
    '-nl',                          "--num_layers",
                                    metavar = '',
                                    type    = int,
                                    default = 8,
                                    help    = "Defines the number of layers in the neural network architecture.")

    parser.add_argument(
    '-dr',                          "--dropout",
                                    metavar = '',
                                    type    = float,
                                    default = 0.1,
                                    help    = "Specifies the fraction of neurons to drop during training to prevent overfitting.")
    
    parser.add_argument(
    '-ef',                          "--no_expand_paths",
                                    action  = 'store_false',
                                    default = True,
                                    help    = "Determines whether to expand file paths relative to the save directory.")


    parser.add_argument(
    '-cu',                          "--no_cuda",
                                    action  = 'store_true', 
                                    default = False,
                                    help    = "Enables GPU acceleration for training if set to True.")
    
    parser.add_argument(
    '-wl',                          "--word_level",
                                    action  = 'store_true', 
                                    default = False,
                                    help    = "Indicates whether to process the input at the character level instead of the word level.")
    

    args = parser.parse_args()
    return args

def main(args):

    # Mude sua fonte de conjunto de dados
    print("\n Loading data source \n", flush = True)
    if args.dataset == "./dataset/eel_comments.csv":
        corpus = pd.read_csv(args.dataset)['comentariosESugestoesDoAluno']#.sample(frac = 0.02) # Sample fraction for scripts test 
    else:
        corpus = pd.read_csv(args.dataset)['comentario'].sample(frac = 0.025) # Change if necessary
    
    # handle dirs
    print("Creating directories", flush = True)
    def handle_dirs(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    mode = "/word" if args.word_level else "/char"
    if args.no_expand_paths:
        handle_dirs(args.save_dir)       
        if args.word_level:
            handle_dirs(args.save_dir + mode)
        else:
            handle_dirs(args.save_dir + mode)
        model_file = os.path.join(args.save_dir + f'{mode}')
        print("Expanded filepaths: ", flush = True)
        print("\t{}".format(args.save_dir), flush = True)
        print("\t{}".format(mode), flush = True)

    # handle cuda
    print("\nChecking CUDA", flush = True)
    if args.no_cuda:
        cuda = torch.device("cpu")
    else:
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}\n".format(torch.cuda.get_device_name()), flush = True)


    print("\n ***** Handling Vocabulary ***** \n", flush = True)
    corpus = DataFrameDataset(corpus)
    tokenizer = vocabulario(character_level = args.word_level)

    num_items = len(corpus)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    subsets_splits = random_split(corpus, [num_train, num_val])

    print(f"Vectorising level: {'WORD' if args.word_level else 'CHARACTER'}\n", flush = True)
    vecs = []
    names = ['train','val']
    for subset,name in zip(subsets_splits,names):
        print(f'Vectorizing {name}:', flush = True)
        vec = CommentsDataset(subset, tokenizer, args.max_length)
        vecs.append(vec)
        print('\n', flush = True)

    AllDatasets = dict(zip(names, vecs))
    train = AllDatasets['train'].create_dataloader(args.batch_size, shuffle = True)
    val = AllDatasets['val'].create_dataloader(args.batch_size, shuffle = False)
    tokenizer.to_serializable(model_file,'GPTeel_vocab.json')

    print("\n ***** Handling Model ***** \n", flush = True)
    model = AutoregressiveWrapper(LanguageModel(
                cuda                    = cuda,
                number_of_tokens        = len(tokenizer),
                max_sequence_length     = args.max_length,
                embedding_dimension     = args.embed_dim,
                number_of_layers        = args.num_layers,
                number_of_heads         = args.num_heads,
                feed_forward_dimension  = None,
                dropout_rate            = args.dropout
    )).to(cuda)

    # Train the model
    model_file = os.path.join(model_file,'GPTeel')
    # Learning control

    trainer = Trainer(model, 
                        cuda                    = cuda,
                        early_stopping_criteria = args.stop_crit, 
                        model_state_file        = model_file, 
                        tokenizer               = tokenizer,
                        lr                      = args.learn_rate,
                        optimizer               = None,
                        criterion               = None,
                        scheduler               = None)

    trainer.train(train, val, args.epochs)

    def plot_metrics(train_state: dict, file: str):
        """Plot loss and accuracy metrics."""

        matplotlib.use('tkAgg')  # Use TkAgg backend for interactive plots
        epochs = range(1, len(train_state['train_loss']) + 1)

        # Plot loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_state['train_loss'], 'bo-', label='Train Loss')
        plt.plot(epochs, train_state['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_state['train_accuracy'], 'bo-', label='Train Accuracy')
        plt.plot(epochs, train_state['val_accuracy'], 'ro-', label='Validation Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        # Save and plot the figure
        plt.savefig(f'{file}_acc_loss.png')  # Save as PNG file
        plt.show()
        plt.close()  # Close the figure to free memory
        print(f'\nGrafico salvo: {file}_acc_loss.png', flush = True)

    # Example call
    plot_metrics(trainer.train_state, model_file)

    print('** Treinamento finalizado **', flush = True)
    print("*** Check Generate_gpteel.py --help to use the model *** \n", flush = True)

if __name__ == "__main__":
    args = _parse_args()
    main(args)
