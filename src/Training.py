from src.vocab_and_tokenize.tokenizer import vocabulario
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import time
import json
import torch
import os

class Trainer:
    def __init__(self,
                    model:                      nn.Module, 
                    cuda:                       torch.device,
                    early_stopping_criteria:    int, 
                    model_state_file:           str, 
                    tokenizer:                  vocabulario,
                    lr:                         float = 0.0001, 
                    optimizer:                  Optional[Optimizer] = None, 
                    criterion:                  Optional[nn.Module] = None, 
                    scheduler:                  Optional[lr_scheduler._LRScheduler] = None):
        
        super().__init__()
        
        self.model = model
        self.cuda = cuda
        self.tokenizer = tokenizer
        self.early_stopping_criteria = early_stopping_criteria
        self.model_state_file = model_state_file

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_index)
        else:
            self.criterion = criterion

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer= self.optimizer,
                                                                         mode='min', factor=0.5,
                                                                         patience=1)
        else:
            self.scheduler = scheduler
        
        self.train_state = Trainer.make_train_state(learning_rate = lr, model_state_file = model_state_file)

    def train(self, train_loader: DataLoader, val_loader : DataLoader, epochs : int):

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model Parameters", flush=True)
        print(f'The model has {num_params:,} trainable parameters \n', flush=True)
        print("Models Arguments", flush=True)
        print('Called: model = %s, opt = %s(lr = %f), epochs = %d, device = %s \n        stop_criteria = %s, criterion = %s, scheduler = %s' % \
        (type(self.model).__name__,
        type(self.optimizer).__name__,
        self.optimizer.param_groups[0]['lr'], 
        epochs, 
        self.cuda, 
        self.early_stopping_criteria,
        type(self.criterion).__name__,
        type(self.scheduler).__name__), flush=True)

        self.train_state["device"] = str(self.cuda)
        self.train_state["batch_size"] = train_loader.batch_size
        self.train_state["trainable_parameters"] = num_params
        self.train_state["optimizer"] = str(type(self.optimizer).__name__)
        self.train_state["scheduler"] = str(type(self.scheduler).__name__)
        self.train_state["criterion"] = str(type(self.criterion).__name__)
        self.train_state["stop_early_criteria"] = self.early_stopping_criteria

        print("\n***** STARTING TRAINING *****\n", flush = True)
        start_time_sec = time.time()
        train_bar = tqdm(desc=f'Training_GPT_EEL',
                              total=len(train_loader), 
                              position=1, 
                              leave=False)

        val_bar = tqdm(desc=f'Validating_GPT_EEL',
                              total=len(val_loader), 
                              position=1, 
                              leave=False)


        for epoch in range(epochs):
            
            self.train_state['epoch_index'] = epoch
            train_bar.reset()
            train_losses = []
            train_accuracies = []

            for batch_idx, (input_tensor,mask_tensor) in enumerate(train_loader):
                self.model.train()
                # Compute the model output
                # Training input
                # Model_output shape(batch_size, sequence_length, number tokens)
                # Target shape (batch_size, sequence_length)
                input_tensor   = input_tensor.to(self.cuda)
                mask_tensor    = mask_tensor.to(self.cuda)

                model_output, target = self.model.forward(
                    x=input_tensor,
                    mask=mask_tensor
                )
                #Compute the Losses
                # The lossess is compute on the model output an the target
                # output transpose shape (batch_size, number tokens, sequence_length)
                loss = self.criterion(model_output.transpose(1, 2), target)

                # Backpropagate the loss.
                loss.backward()

                # Clip the gradients. This is used to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                # Update the model parameters. This is done by taking a step in the direction of the gradient.
                self.optimizer.step()

                # Reset the gradients. This is done so that the gradients from the previous batch
                # are not used in the next step.
                self.optimizer.zero_grad()

                # Append the loss to the list of losses, so that the average loss can be computed for this epoch.
                train_losses.append(loss.item())

                accuracy = Trainer.calculate_accuracy(model_output, target)
                train_accuracies.append(accuracy)
                
                train_bar.set_postfix({'loss': round(np.average(train_losses),3),
                                       "acc(%)" : round(np.average(train_accuracies),3),
                                       "epoch" : epoch + 1})
                train_bar.update()
        
            # Append the loss to the list of losses, so that the average loss can be computed for this epoch.
            # average or mean

            self.train_state['train_loss'].append(np.average(train_losses))
            self.train_state['train_accuracy'].append(np.average(train_accuracies))

            # Validation
            self.model.eval()
            val_bar.reset()
            val_losses = []
            val_accuracies = []

            with torch.no_grad():
                for batch_idx, (input_tensor,mask_tensor) in enumerate(val_loader):
                    # Compute the model output
                    # Training input
                    input_tensor   = input_tensor.to(self.cuda)
                    mask_tensor    = mask_tensor.to(self.cuda)

                    model_output, target = self.model.forward(
                        x=input_tensor,
                        mask=mask_tensor
                    )

                    #Compute the Losses
                    # The lossess is compute on the model output an the target
                    loss = self.criterion(model_output.transpose(1, 2), target)

                    # Append the loss to the list of losses, so that the average loss can be computed for this epoch.
                    val_losses.append(loss.item())
                    accuracy = Trainer.calculate_accuracy(model_output, target)
                    val_accuracies.append(accuracy)

                    val_bar.set_postfix({'loss': round(np.average(val_losses),3),
                       "acc(%)" : round(np.average(val_accuracies),3),
                       "epoch" : epoch + 1})
                    val_bar.update()

            self.train_state['val_loss'].append(np.average(val_losses))
            self.train_state['val_accuracy'].append(np.average(val_accuracies))


            self.scheduler.step(self.train_state['val_loss'][-1])

            # Update training state with early stopping
            self.train_state = Trainer.update_train_state(self.early_stopping_criteria, 
                                                            self.model,
                                                            self.train_state,
                                                            self.optimizer,
                                                            self.scheduler)

            if self.train_state['stop_early']:
                print(f"Early stopping at epoch {epoch}", flush = True)
                break
        
        # END OF TRAINING LOOP
        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print('\n\nTime total:     %5.2f sec' % (total_time_sec), flush=True)
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec), flush=True)

        Trainer.save_the_time('total_time',
                                self.train_state['model_filename'] + '_train_state.json',
                                total_time_sec)
        Trainer.save_the_time('time_per_epoch',
                                self.train_state['model_filename'] + '_train_state.json', 
                                time_per_epoch_sec)
        
        if os.path.exists(self.train_state['model_filename'] + '_train_state_temporary.json'):
            os.remove(self.train_state['model_filename'] + '_train_state_temporary.json')        

        print(f'\nModelo e status salvo: {self.train_state['model_filename']}', flush = True)
    
    @staticmethod
    def make_train_state(learning_rate: float, model_state_file: str):
        return {
            "trainable_parameters" : 0,
            "batch_size" : None,
            "optimizer" : None,
            "scheduler" : None,
            "criterion" :   None,
            "stop_early_criteria": None,
            "stop_early": False,
            "early_stopping_step": 0,
            "early_stopping_best_val": 1e8,
            "learning_rate": learning_rate,
            "epoch_index": 0,
            "model_filename": model_state_file,
            "device" : None,
            "total_time" : None,
            "time_per_epoch" : None,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    @staticmethod
    def update_train_state(early_stopping_criteria: int, model: nn.Module, train_state: dict, optimizer: Optimizer, 
                           scheduler: lr_scheduler._LRScheduler):

        """Update training state with early stopping and model checkpointing."""
        if train_state['epoch_index'] == 0 or train_state['val_loss'][-1] < train_state['early_stopping_best_val']:
            
            ## Save cheeckpoints the dictionary to a JSON file and the best model to a pt file
            Trainer.save_state_and_model(train_state, model, optimizer, scheduler, 
                                         json_filename=train_state['model_filename'] + '_train_state.json', 
                                         model_filename=train_state['model_filename'] + '.pt')

            #Restart stopping_step
            train_state['early_stopping_step'] = 0
            #update the best value loss
            train_state['early_stopping_best_val'] = train_state['val_loss'][-1]
        else:
            
            #Save the actual train state for consulting during the training
            with open(train_state['model_filename'] +'_train_state_temporary.json', 'w') as json_file:
                json.dump(train_state, json_file, indent=4)

            train_state['early_stopping_step'] += 1

        train_state['stop_early'] = train_state['early_stopping_step'] >= early_stopping_criteria
        return train_state
    
    @staticmethod
    def save_state_and_model(train_state: dict, model: nn.Module, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler,
                             json_filename: str, model_filename: str):

        # Save the train state to a JSON file
        with open(json_filename, 'w') as json_file:
            json.dump(train_state, json_file, indent=4)

        # Save the scheduler and optimizer state state to a pth file
        torch.save(scheduler.state_dict(), train_state['model_filename'] + '_scheduler_state.pth')
        torch.save(optimizer.state_dict(), train_state['model_filename'] + '_optimizer_state.pth')

        # Save the PyTorch model to a .pt file
        model.save_checkpoint(model_filename)
    
    @staticmethod
    def calculate_accuracy(output: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcula a acurácia a partir da saída e dos targets."""

        # Obter as previsões de tokens mais prováveis ao longo da dimensão vocab_size
        _, predicted_tokens = torch.max(output, dim=-1)  # Shape: (batch_size, sequence_length)

        # Certificar que os shapes são iguais, se necessário, ajuste
        if predicted_tokens.shape[1] != targets.shape[1]:
            min_length = min(predicted_tokens.shape[1], targets.shape[1])
            predicted_tokens = predicted_tokens[:, :min_length]
            targets = targets[:, :min_length]

        # Comparar as previsões com os valores reais dos tokens
        correct_predictions = (predicted_tokens == targets).float()  # Shape: (batch_size, sequence_length)

        # Máscara para ignorar tokens de padding (assumindo 0 como padding)
        mask = (targets != 0).float()  # Shape: (batch_size, sequence_length)
        correct_predictions *= mask  # Aplicar a máscara

        # Calcular a acurácia média por token
        total_correct = correct_predictions.sum()
        total_valid = mask.sum()

        # Verificar se há tokens válidos para evitar divisão por zero
        if total_valid == 0:
            return 0.0

        accuracy = (total_correct / total_valid) * 100  # Média ponderada

        return accuracy.item()
    
    @staticmethod
    def save_the_time(param, file, seconds: float):
        # Define time units in seconds
        SECONDS_IN_MINUTE = 60
        SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
        SECONDS_IN_DAY = SECONDS_IN_HOUR * 24
        SECONDS_IN_YEAR = SECONDS_IN_DAY * 365  # Approximate, ignoring leap years

        # Calculate time units
        years = seconds // SECONDS_IN_YEAR
        seconds %= SECONDS_IN_YEAR
        days = seconds // SECONDS_IN_DAY
        seconds %= SECONDS_IN_DAY
        hours = seconds // SECONDS_IN_HOUR
        seconds %= SECONDS_IN_HOUR
        minutes = seconds // SECONDS_IN_MINUTE
        seconds %= SECONDS_IN_MINUTE

        # Create result list with only non-zero values
        result = []
        if years > 0:
            result.append(f"{round(years,2)} years")
        if days > 0 or (years == 0 and hours == 0 and minutes == 0 and seconds > 0):
            result.append(f"{round(days,2)} days")
        if hours > 0 or (years == 0 and days == 0 and minutes == 0 and seconds > 0):
            result.append(f"{round(hours,2)} hours")
        if minutes > 0 or (years == 0 and days == 0 and hours == 0 and seconds > 0):
            result.append(f"{round(minutes,2)} minutes")
        if seconds > 0 or not result:  # Ensure seconds is included if no other unit is relevant
            result.append(f"{round(seconds,2)} seconds")

        # update State Json
        with open(file, 'r+') as file:
            data = json.load(file)
            data[str(param)] = ', '.join(result)
            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()