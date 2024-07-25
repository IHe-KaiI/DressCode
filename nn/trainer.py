# Training loop func
from pathlib import Path
import time
import traceback

import torch
import wandb as wb
import torch.nn as nn

# My modules
import data

import net_blocks as blocks
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import datetime

SOS = 2001
EOS = 2002
PAD = 2003
C_outline    = 50
C_rotation   = 1000
C_transl     = 1000
C_stitch_tag = 1000

class Trainer():
    def __init__(
            self, 
            setup, experiment_tracker, dataset=None, data_split={}, 
            with_norm=True, with_visualization=False, resume=None):
        """Initialize training and dataset split (if given)
            * with_visualization toggles image prediction logging to wandb board. Only works on custom garment datasets (with prediction -> image) conversion"""
        self.experiment = experiment_tracker
        self.datawraper = None
        self.standardize_data = with_norm
        self.log_with_visualization = with_visualization
        
        # training setup
        self.setup = setup

        if dataset is not None:
            self.use_dataset(dataset, data_split)
        
        self.resume = None
        if resume is not None:
            self.resume = True

    def init_randomizer(self, random_seed=None):
        """Init randomizatoin for torch globally for reproducibility. 
            Using this function ensures that random seed will be recorded in config
        """
        # see https://pytorch.org/docs/stable/notes/randomness.html
        if random_seed:
            self.setup['random_seed'] = random_seed
        elif not self.setup['random_seed']:
            self.setup['random_seed'] = int(time.time())

        torch.manual_seed(self.setup['random_seed'])
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def use_dataset(self, dataset, split_info):
        """Use specified dataset for training with given split settings"""
        self.datawraper = data.DatasetWrapper(dataset)
        self.datawraper.load_split(split_info)
        self.datawraper.new_loaders(self.setup['batch_size'], shuffle_train=True)

        if self.standardize_data:
            self.datawraper.standardize_data()

        return self.datawraper

    def fit(self, model):
        """Fit provided model to reviosly configured dataset"""
        if not self.datawraper:
            raise RuntimeError('Trainer::Error::fit before dataset was provided. run use_dataset() first')

        self.device = model.device_ids[0] if hasattr(model, 'device_ids') else self.setup['devices'][0]
        
        self._add_optimizer(model)
        self._add_scheduler(len(self.datawraper.loaders.train)) 
        self.es_tracking = []  # early stopping init

        start_epoch = self._start_experiment(model)
        print('Trainer::NN training Using device: {}'.format(self.device))

        if self.log_with_visualization:
            # to run parent dir -- wandb will automatically keep track of intermediate values
            # Othervise it might only display the last value (if saving with the same name every time)
            self.folder_for_preds = Path('./wandb') / 'intermediate_preds'
            self.folder_for_preds.mkdir(exist_ok=True)
        
        self._fit_loop(model, self.datawraper.loaders.train, self.datawraper.loaders.validation, start_epoch=start_epoch)

        print("Trainer::Finished training")
        # self.experiment.stop() -- not stopping the run for convenice for further processing outside of the training routines

    # ---- Private -----
    def _fit_loop(self, model, train_loader, valid_loader, start_epoch=0):
        """Fit loop with the setup already performed. Assumes wandb experiment was initialized"""
        model.to('cuda')
        log_step = wb.run.step - 1
        best_valid_loss = None
        
        CLIP_embedding = blocks.StableDiffusion(torch.device('cuda'), False, True)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD)
        now_time = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
        writer = SummaryWriter(log_dir = f'./tensorboard/{now_time}')

        for epoch in range(start_epoch, wb.config.trainer['epochs']):
            model.train()
            for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

                captions_batch, gt = batch['captions'], batch['ground_truth']
                batch_size = len(captions_batch)

                prompts_batch = []
                for i in range(batch_size):
                    prompts_batch.append(CLIP_embedding.get_text_embeds(captions_batch[i])[0])
                prompts_batch = torch.stack(prompts_batch)

                indices_value = gt["indices_value"].to('cuda').to(torch.long)
                indices_axis  = gt["indices_axis"].to('cuda').to(torch.long)
                indices_pos   = gt["indices_pos"].to('cuda').to(torch.long)

                with torch.autocast("cuda", torch.float16):
                    CLIP_feature  = model.proj_feature_txt(prompts_batch)
                    logits_text = model(indices_value.clone(), indices_axis.clone(), indices_pos.clone(), CLIP_feature)

                loss = criterion(logits_text[:, :-1].reshape(-1, model.vocab_size), indices_value[:, 1:].reshape(-1))
                loss_dict = {}
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # logging
                log_step += 1
                loss_dict.update({'epoch': epoch, 'batch': i, 'loss': loss, 'learning_rate': self.optimizer.param_groups[0]['lr']})
                
                # We use Tensorboard here instead of WB
                # wb.log(loss_dict, step=log_step)
                writer.add_scalar('Training Loss', loss, log_step)

            model.eval()
            with torch.no_grad():
                
                losses = []
                for batch in tqdm(valid_loader):
                    captions_batch, gt = batch['captions'], batch['ground_truth']
                    batch_size = len(captions_batch)

                    prompts_batch = []
                    for i in range(batch_size):
                        prompts_batch.append(CLIP_embedding.get_text_embeds(captions_batch[i])[0])
                    prompts_batch = torch.stack(prompts_batch)

                    indices_value = gt["indices_value"].to('cuda').to(torch.long)
                    indices_axis  = gt["indices_axis"].to('cuda').to(torch.long)
                    indices_pos   = gt["indices_pos"].to('cuda').to(torch.long)

                    with torch.autocast("cuda", torch.float16):
                        CLIP_feature  = model.proj_feature_txt(prompts_batch)
                        logits_text = model(indices_value.clone(), indices_axis.clone(), indices_pos.clone(), CLIP_feature)

                    
                    loss = criterion(logits_text[:, :-1].reshape(-1, model.vocab_size), indices_value[:, 1:].reshape(-1))
                    losses.append(loss)

                valid_loss = sum(losses) / len(losses)  

            # Checkpoints: & compare with previous best
            if best_valid_loss is None or valid_loss < best_valid_loss:  
                best_valid_loss = valid_loss
                self._save_checkpoint(model, epoch, best=True)  
            else:
                self._save_checkpoint(model, epoch)

            # Base logging
            print('Epoch: {}, Validation Loss: {}'.format(epoch, valid_loss))

            # We use Tensorboard here instead of WB
            # wb.log({'epoch': epoch, 'valid_loss': valid_loss, 'best_valid_loss': best_valid_loss}, step=log_step)
            writer.add_scalar('Validate Loss', valid_loss, epoch)

            # check for early stoping
            if self._early_stopping(loss, best_valid_loss, self.optimizer.param_groups[0]['lr']):
                print('Trainer::Stopped training early')
                break

    def _start_experiment(self, model):
        self.experiment.init_run({'trainer': self.setup})
        if self.resume or wb.run.resumed:
            start_epoch = self._restore_run(model)
            self.experiment.checkpoint_counter = start_epoch
            print('Trainer::Resumed run {} from epoch {}'.format(self.experiment.cloud_path(), start_epoch))

            if self.device != wb.config.trainer['devices'][0]:
                # device doesn't matter much, so we just inform but do not crash
                print('Trainer::Warning::Resuming run on different device. Was {}, now using {}'.format(
                    wb.config.trainer['devices'][0], self.device))
        else:
            start_epoch = 0
            # record configurations of data and model
            self.datawraper.save_to_wandb(self.experiment)

        wb.watch(model, log='all')
        return start_epoch

    def _add_optimizer(self, model):
        
        if self.setup['optimizer'] == 'SGD':
            # future 'else'
            print('Trainer::Using default SGD optimizer')
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])
        elif self.setup['optimizer'] == 'Adam':
            # future 'else'
            print('Trainer::Using Adam optimizer')
            print('Trainer::lr = ', self.setup['learning_rate'])
            model.to(self.device)  # see https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.setup['learning_rate'], weight_decay=self.setup['weight_decay'])

    def _add_scheduler(self, steps_per_epoch):
        if 'lr_scheduling' in self.setup:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, 
                max_lr=self.setup['learning_rate'],
                epochs=self.setup['epochs'],
                steps_per_epoch=steps_per_epoch,
                cycle_momentum=False  # to work with Adam
            )
        else:
            self.scheduler = None
            print('Trainer::Warning::no learning scheduling set')

    def _restore_run(self, model):
        """Restore the training process from the point it stopped at. 
            Assuming 
                * Current wb.config state is the same as it was when run was initially created
                * All the necessary training objects are already created and only need update
                * All related object types are the same as in the resuming run (model, optimizer, etc.)
                * Self.run_id is properly set
            Returns id of the next epoch to resume from. """
                                                    
        # get latest checkoint info
        print('Trainer::Loading checkpoint to resume run..')
        checkpoint = self.experiment.get_checkpoint_file_offline()  # latest

        # checkpoint loaded correctly
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # https://discuss.pytorch.org/t/how-to-save-and-load-lr-scheduler-stats-in-pytorch/20208

        # new epoch id
        return checkpoint['epoch'] + 1

    def _early_stopping(self, last_loss, last_tracking_loss, last_lr):
        """Check if conditions are met to stop training. Returns a message with a reason if met
            Early stopping allows to save compute time"""

        # loss goes into nans
        if torch.isnan(last_loss):
            self.experiment.add_statistic('stopped early', 'Nan in losses', log='Trainer::EarlyStopping')
            return True

        # Target metric is not improving for some time
        self.es_tracking.append(last_tracking_loss.item())
        if len(self.es_tracking) > (wb.config.trainer['early_stopping']['patience'] + 1):  # number of last calls to consider plus current -> at least two
            self.es_tracking.pop(0)
            # if all values fit into a window, they don't change much
            if abs(max(self.es_tracking) - min(self.es_tracking)) < wb.config.trainer['early_stopping']['window']:
                self.experiment.add_statistic(
                    'stopped early', 'Metric have not changed for {} epochs'.format(wb.config.trainer['early_stopping']['patience']), 
                    log='Trainer::EarlyStopping')
                return True
        # do not check untill wb.config.trainer['early_stopping'].patience # of calls are gathered

        # Learning rate vanished
        if last_lr < 1e-6:
            self.experiment.add_statistic('stopped early', 'Learning Rate vanished', log='Trainer::EarlyStopping')
            return True
        
        return False
            
    def _save_checkpoint(self, model, epoch, best=False):
        """Save checkpoint that can be used to resume training"""
        
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        self.experiment.save_checkpoint(
            checkpoint_dict,
            aliases=['best'] if best else [], 
            wait_for_upload=best
        )
        
       
