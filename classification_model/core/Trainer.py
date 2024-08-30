import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import os, pdb
import copy
import numpy as np
import wandb

from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

from core.apply import number_of_correct, get_likely_index, log_confusion_matrix

class CNNTrainer:
    def __init__(self, cfg, model, labels, device):
        self.cfg = cfg
        self.model = model
        self.wandb = cfg.wandb_
        self.wandb.watch(self.model, log="all")
        self.log_interval = cfg.log_interval
        self.labels = labels
        self.device = device

    def train(self, train_dataloader, test_dataloader):
        min_loss = float('inf')
        best_result = dict()
        best_score = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        tqdm_bar = tqdm(range(self.cfg.epoch))      
        for epoch in tqdm_bar:
            final_loss = 0
            self.model.train()

            tqdm_bar_batch = tqdm(enumerate(train_dataloader))
            for batch_idx, (data, target) in tqdm_bar_batch:

                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)

                loss = F.nll_loss(output.squeeze(), target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                final_loss += loss.detach().cpu().item()          
                # print training stats
                if batch_idx % self.cfg.log_interval == 0:
                    # print(f"Train Epoch: {epoch} Batch: {batch_idx} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} ({self.cfg.log_interval * batch_idx / len(train_dataloader.dataset):.0f}%)]\tLoss: {loss.item():.6f}")
                    self.wandb.log({f"loss": loss})

                # update progress bar
                tqdm_bar_batch.set_description(f'Epoch: {epoch},  Batch: {batch_idx}, loss: {loss/len(target)}')
                # record loss
            tqdm_bar.set_description(f'Epoch: {epoch}, loss: {final_loss/len(train_dataloader.dataset)}')



            accuracy = self.evaluate(test_dataloader, epoch, task="validate")
            self.wandb.log({f"final_loss": final_loss, "accuracy": accuracy})
            self.scheduler.step()

            if best_score < accuracy:
                best_score = accuracy
                best_model = copy.deepcopy(self.model)
                model_path = f"{self.cfg.model_path}_model-{self.cfg.model}_epoch-{epoch}_batch-{self.cfg.batch_size}_data-{self.cfg.data_type}_lr-{self.cfg.lr}.torch"
                if not os.path.exists(os.path.basename(model_path)):
                    os.mkdir(os.path.basename(model_path))
                torch.save(best_model.state_dict(), model_path)

        return best_result
   
    #replaced log_confusion_matrix with apply.py
    
    def evaluate(self, test_loader, epoch, task="test"):
        with torch.no_grad():
            self.model.eval()
            correct = 0
            y_pred = []
            y_true = []

            tqdm_bar_batch = tqdm(test_loader)
            for data, target in test_loader:

                data = data.to(self.device)
                target = target.to(self.device)

                # apply transform and model on whole batch directly on device
                #data = transform(data)
                output = self.model(data)

                try:
                    pred = get_likely_index(output)     
                except:
                    try:
                        output = output.squeeze()
                        pred = get_likely_index(output)
                    except:
                        import pdb; pdb.set_trace()       
                correct_pred =  number_of_correct(pred, target)
                correct += correct_pred
                y_pred.append(pred.cpu())
                y_true.append(target.cpu())

                # update progress bar
                tqdm_bar_batch.set_description(f"\nTest Epoch: {epoch}\tAccuracy: {correct_pred}/{len(target)} ({100. * correct_pred / len(target):.0f}%)\n")
            
            # tqdm_bar.set_description(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
            preds = torch.cat(y_pred).detach()
            labels = torch.cat(y_true).detach()
            
            log_confusion_matrix(preds, labels, epoch, self.cfg.labels)


            acc = accuracy_score(labels, preds)
            recall = recall_score(labels, preds, average='weighted')
            precision = precision_score(labels, preds, average='weighted')
            f1 = f1_score(labels, preds, average='weighted')


            result = {}
            result[f'{task}/accuracy'] = acc
            result[f'{task}/recall'] = recall
            result[f'{task}/f1'] = f1
            result[f'{task}/precision'] = precision

            self.wandb.log(result)
            # print(f"\nTest Epoch: {epoch}\tAccuracy: {acc})\n")
        return acc
    

    #replaced number_of_correct, get_likely_index to the one in apply.py

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))


    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]

    def predict(self, tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(self.device)
        #tensor = transform(tensor)
        tensor = self.model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        tensor = self.index_to_label(tensor.squeeze())
        return tensor
    

class ResnetTrainer:
    def __init__(self, cfg, my_model):
        self.cfg = cfg
        self.wandb = cfg.wandb_
        self.wandb.watch(my_model, log="all")
        
    
    def train_epoch(self, ml_model, train_loader, epoch, optimizer):
        ml_model.train()

        losses = []
        #import pdb; pdb.set_trace()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="batch_idx"):
            data = data.to(self.cfg.device)

            target = target.to(self.cfg.device)
            output = ml_model(data)
            loss = F.nll_loss(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % self.cfg.log_interval == 0:
                print(f"Train Epoch: {epoch}\tLoss: {loss.item():.4f}")
                self.wandb.log({f"Loss": loss})

            losses.append(loss.item())

        return losses

    def train(self, ml_model, train_loader, validation_loader, test_loader):

        if self.cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(ml_model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif self.cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(ml_model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
            scheduler = None
        else:
            raise ValueError(f"Unknown optimizer {optimizer}, use adam/sgd")
    
        best_score = 0
        best_model = copy.deepcopy(ml_model)
        for epoch in tqdm(range(self.cfg.epoch), desc=f"best_score_{best_score}"):
            print(f"--- start epoch {epoch} ---")
            self.train_epoch(ml_model, train_loader, epoch, optimizer)
            if scheduler:
                scheduler.step()
            result = self.evaluate(ml_model, validation_loader, "validation")
            print(f"Validation accuracy: {result['validation/accuracy']:.5f}")
            if best_score < result["validation/accuracy"]:
                best_score = result["validation/accuracy"]
                best_model = copy.deepcopy(ml_model)
                model_path = f"{self.cfg.model_path}_model-{self.cfg.model}_epoch_{epoch}_nfft_{self.cfg.nfft}_batch-{self.cfg.batch_size}_data-{self.cfg.data_type}_lr-{self.cfg.lr}.torch"
                if not os.path.exists(os.path.basename(model_path)):
                    os.mkdir(os.path.basename(model_path))
                torch.save(best_model.state_dict(), model_path)
            self.wandb.log({f"best_score": best_score})
            self.wandb.log({f"Validation_score": best_score})
            self.wandb.log(result)
        print(f"Top validation accuracy: {best_score:.5f}")
        result = self.evaluate(best_model, test_loader, self.cfg.device, "test")
        self.wandb.log({f"best_test_score": best_score})
        self.wandb.log(result)

    def evaluate(self, model, data_loader, task="validation"):
        model.eval()
        correct = 0
        y_pred = []
        y_true = []
        for data, target in data_loader:
            data = data.to(self.cfg.device)
            target = target.to(self.cfg.device)

            pred = model(data)
            pred = get_likely_index(pred)

            correct += number_of_correct(pred, target)

            y_pred.append(pred.cpu())
            y_true.append(target.cpu())

        score = correct / len(data_loader.dataset)

        preds = torch.cat(y_pred).detach()
        labels = torch.cat(y_true).detach()

        acc = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')

        log_confusion_matrix(preds, labels, 0, self.cfg.labels)

        result = {}
        result[f'{task}/accuracy'] = acc
        result[f'{task}/recall'] = recall
        result[f'{task}/f1'] = f1
        result[f'{task}/precision'] = precision

        return result
    @staticmethod
    def apply_to_wav(model, waveform: torch.Tensor, sample_rate: float, device: str):
        model.eval()
        mel_spec = core.combined_dataloader.prepare_wav(waveform, sample_rate)
        mel_spec = torch.unsqueeze(mel_spec, dim=0).to(device)
        res = model(mel_spec)

        probs = torch.nn.Softmax(dim=-1)(res).cpu().detach().numpy()
        predictions = []
        for idx in np.argsort(-probs):
            label = core.combined_dataloader.idx_to_label(idx)
            predictions.append((label, probs[idx]))
        return predictions
    
    def apply_to_file(self, model, wav_file: str):
        waveform, sample_rate = torchaudio.load(wav_file)
        return self.apply_to_wav(model, waveform, sample_rate, self.cfg.device)