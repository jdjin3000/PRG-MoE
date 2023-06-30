import logging
import os
import datetime

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import module.model as M
from module.evaluation import log_metrics, FocalLoss
from module.preprocessing import get_data, tokenize_conversation

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class LearningEnv:
    def __init__(
        self, 
        gpus, 
        train_data, 
        valid_data, 
        test_data, 
        split_directory, 
        max_seq_len, 
        log_directory, 
        model_name, 
        port, 
        contain_context, 
        data_label, 
        **kwargs,
    ):
        self.gpus = gpus
        self.single_gpu = len(self.gpus) == 1

        self.train_dataset = train_data
        self.valid_dataset = valid_data
        self.test_dataset = test_data

        self.split_directory = split_directory
        self.split_num = None

        self.contain_context = contain_context
        self.max_seq_len = max_seq_len

        self.start_time = datetime.datetime.now()
        self.log_directory = log_directory

        self.options = kwargs

        self.model_name = model_name
        self.port = port

        self.split_performance = None

        self.data_label = data_label

        self.best_performance = [0, 0, 0]  # p, r, f1

    def __set_model__(
        self,
        pretrained_model,
        dropout,
        n_speaker,
        n_emotion,
        n_cause,
        n_expert,
        guiding_lambda,
        **kwargs,
    ):
        self.n_cause = n_cause

        model_args = {
            "dropout": dropout,
            "n_speaker": n_speaker,
            "n_emotion": n_emotion,
            "n_cause": n_cause,
            "n_expert": n_expert,
            "guiding_lambda": guiding_lambda,
        }

        if pretrained_model is not None:
            model = getattr(M, self.model_name)(**model_args)
            model.load_state_dict(torch.load(pretrained_model))
        else:
            model = getattr(M, self.model_name)(**model_args)

        return model

    def set_model(self, allocated_gpu):
        if not self.single_gpu:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{self.port}",
                world_size=len(self.gpus),
                rank=allocated_gpu,
            )

        torch.cuda.set_device(allocated_gpu)

        model = self.__set_model__(**self.options).cuda(allocated_gpu)

        if self.single_gpu:
            self.distributed_model = model
        else:
            self.distributed_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[allocated_gpu], find_unused_parameters=True
            )

    def set_logger_environment(self, file_name_list, logger_name_list):
        for file_name, logger_name in zip(file_name_list, logger_name_list):
            for handler in logging.getLogger(logger_name).handlers[:]:
                logging.getLogger(logger_name).removeHandler(handler)
            self.set_logger(file_name, logger_name)

    def set_logger(self, file_name, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.log_directory:
            if not os.path.exists(f'{self.log_directory}'):
                os.makedirs(f'{self.log_directory}')
            file_handler = logging.FileHandler(f'{self.log_directory}/{file_name}')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_dataloader(self, dataset_file, batch_size, num_worker, shuffle=True, contain_context=False):
        device = "cuda:0"
        data = get_data(dataset_file, device, self.max_seq_len, contain_context)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = data[0]
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = data[1:]

        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
        dataloader_params = {
            "dataset": dataset_,
            "batch_size": batch_size,
            "shuffle": shuffle
        }

        if not self.single_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_, shuffle=shuffle)
            dataloader_params.update({
                "sampler": train_sampler,
                "num_workers": num_worker,
                "pin_memory": False
            })

        return DataLoader(**dataloader_params)


    def init_stopper(self):
        self.stopper[0] = 0


    def multiprocess_work(self, test, training_iter, batch_size, learning_rate, patience, num_worker, **kwargs):
        stopper = torch.zeros(1)
        stopper.share_memory_()

        if self.split_directory:
            self.set_logger_environment([f"{self.model_name}-split_average-{self.start_time}.log"], ["split_logger"])
            logger = logging.getLogger("split_logger")

            split_performance = torch.zeros((3, len(os.listdir(self.split_directory)), 5))
            split_performance.share_memory_()

            for split_dir in os.listdir(self.split_directory):
                self.split_num = split_dir.split("_")[-1]
                base_file_name = os.path.join(self.split_directory, split_dir, f"split_{self.split_num}")
                self.train_dataset = f"{base_file_name}_train.json"
                self.valid_dataset = f"{base_file_name}_valid.json"
                self.test_dataset = f"{base_file_name}_test.json"
                self.start_time = datetime.datetime.now()

                if self.single_gpu:
                    self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test)
                else:
                    torch.multiprocessing.spawn(self.child_process, args=(training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test), nprocs=len(self.gpus))

            mean_performance = torch.mean(split_performance, dim=1)
            logger.info(f"Emotion Classification Test Average Performance | {len(os.listdir(self.split_directory))} trials | {mean_performance[0]}\n")
            logger.info(f"Binary Cause Classification Test Average Performance | {len(os.listdir(self.split_directory))} trials | {mean_performance[1]}\n")
            logger.info(f"MultiClass Cause Classification Test Average Performance | {len(os.listdir(self.split_directory))} trials | {mean_performance[2]}\n")

        else:
            if self.single_gpu:
                self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test)
            else:
                torch.multiprocessing.spawn(self.child_process, args=(training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test), nprocs=len(self.gpus))


    def child_process(self, allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test=False):
        batch_size = int(batch_size / len(self.gpus))
        num_worker = int(num_worker / len(self.gpus))

        self.set_model(allocated_gpu) 

        if allocated_gpu == 0:
            logger_name_list = ['train', 'valid', 'test']

            if self.n_cause == 2:
                file_name_list = [f'{self.model_name}-binary_cause-{_}-{self.start_time}.log' for _ in logger_name_list]
            else:
                file_name_list = [f'{self.model_name}-multiclass_cause-{_}-{self.start_time}.log' for _ in logger_name_list]

            self.set_logger_environment(file_name_list, logger_name_list)

        self.stopper = stopper
        self.split_performance = split_performance


        if test:
            self.valid(allocated_gpu, batch_size, num_worker, saver=None, option='test')
        else:
            self.train(allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker)
            self.valid(allocated_gpu, batch_size, num_worker, saver=None, option='test')

        if not self.single_gpu:
            torch.distributed.barrier()


    def train(self, allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker):
        def get_pad_idx(utterance_input_ids_batch):
            batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
            check_pad_idx = torch.sum(utterance_input_ids_batch.view(-1, max_seq_len)[:, 2:], dim=1).cpu()

            return check_pad_idx

        def get_pair_pad_idx(utterance_input_ids_batch, window_constraint=3, emotion_pred=None):
            batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
            
            check_pad_idx = get_pad_idx(utterance_input_ids_batch)

            if emotion_pred is not None:
                emotion_pred = torch.argmax(emotion_pred, dim=1)
                
            check_pair_window_idx = list()
            for batch in check_pad_idx.view(-1, max_doc_len):
                pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))
                for end_t in range(1, len(batch.nonzero()) + 1):
                    if emotion_pred is not None and emotion_pred[end_t - 1] == 6:
                        continue
                    
                    pair_window_idx[max(0, int((end_t - 1) * end_t / 2), int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                check_pair_window_idx.append(pair_window_idx)
            
            return torch.stack(check_pair_window_idx)

        if allocated_gpu == 0:
            self.init_stopper()
            logger = logging.getLogger('train')

        optimizer = optim.Adam(self.distributed_model.parameters(), lr=learning_rate)

        if self.n_cause == 2:
            model_name_suffix = 'binary_cause'
        else:
            model_name_suffix = 'multiclass_cause'

        if not os.path.exists("model/"):
            os.makedirs("model/")
        saver = ModelSaver(path=f"model/{self.model_name}-{model_name_suffix}-{self.data_label}-{self.start_time}.pt", single_gpu=self.single_gpu)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)

        train_dataloader = self.get_dataloader(self.train_dataset, batch_size, num_worker)

        for i in range(training_iter):
            self.distributed_model.train()
            
            loss_avg, count= 0, 0
            emo_pred_y_list, emo_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, cau_pred_y_list, cau_true_y_list = [list() for _ in range(6)]

            for utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch in tqdm(train_dataloader, desc=f"Train | Epoch {i+1}"):
                batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
                
                check_pad_idx = get_pad_idx(utterance_input_ids_batch)

                prediction = self.distributed_model(
                                                    utterance_input_ids_batch, 
                                                    utterance_attention_mask_batch, 
                                                    utterance_token_type_ids_batch, 
                                                    speaker_batch
                                                    )

                if len(prediction) != 2:
                    emotion_prediction, binary_cause_prediction = prediction
                else:
                    emotion_prediction, binary_cause_prediction = prediction
                
                check_pair_window_idx = get_pair_pad_idx(
                                                        utterance_input_ids_batch, 
                                                        window_constraint=3, 
                                                        emotion_pred=emotion_prediction
                                                        )
                check_pair_pad_idx = get_pair_pad_idx(
                                                    utterance_input_ids_batch, 
                                                    window_constraint=1000
                                                    )

                emotion_prediction = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
                binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                binary_cause_prediction_all = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                
                emotion_label_batch = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]

                if self.n_cause == 2:
                    pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                    pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                else:
                    pair_cause_label_batch = torch.argmax(pair_cause_label_batch.view(-1, self.n_cause), dim=1).view(batch_size, -1)

                    pair_binary_cause_label_batch_window = pair_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                    pair_binary_cause_label_batch_all = pair_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]

                criterion_emo = FocalLoss(gamma=2)
                criterion_cau = FocalLoss(gamma=2)

                loss_emo = criterion_emo(emotion_prediction, emotion_label_batch.to(allocated_gpu))
                
                if torch.sum(check_pair_window_idx) == 0:
                    loss = loss_emo
                else:
                    loss_cau = criterion_cau(binary_cause_prediction_window, pair_binary_cause_label_batch_window.to(allocated_gpu))
                    loss = 0.2 * loss_emo + 0.8 * loss_cau

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cau_pred_y_list_all.append(binary_cause_prediction_all), cau_true_y_list_all.append(pair_binary_cause_label_batch_all)
                cau_pred_y_list.append(binary_cause_prediction_window), cau_true_y_list.append(pair_binary_cause_label_batch_window)
                emo_pred_y_list.append(emotion_prediction), emo_true_y_list.append(emotion_label_batch)

                loss_avg += loss.item()
                count += 1

            loss_avg = loss_avg / count

            # Logging Performance
            if allocated_gpu == 0:
                p_cau, r_cau, f1_cau = log_metrics(logger, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg, n_cause=self.n_cause, option='train')
            self.valid(allocated_gpu, batch_size, num_worker, saver)
            
            if not self.single_gpu:
                torch.distributed.barrier()

            self.valid(allocated_gpu, batch_size, num_worker, saver, option='test')

            if self.stopper or (i == training_iter - 1):
                return
            
            scheduler.step()

    def valid(self, allocated_gpu, batch_size, num_worker, saver=None, option='valid'):
        def get_pad_idx(utterance_input_ids_batch):
            batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
            check_pad_idx = torch.sum(utterance_input_ids_batch.view(-1, max_seq_len)[:, 2:], dim=1).cpu()

            return check_pad_idx

        def get_pair_pad_idx(utterance_input_ids_batch, window_constraint=3, emotion_pred=None):
            batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
            
            check_pad_idx = get_pad_idx(utterance_input_ids_batch)

            if emotion_pred is not None:
                emotion_pred = torch.argmax(emotion_pred, dim=1)
                
            check_pair_window_idx = list()
            for batch in check_pad_idx.view(-1, max_doc_len):
                pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))
                for end_t in range(1, len(batch.nonzero()) + 1):
                    if emotion_pred is not None and emotion_pred[end_t - 1] == 6:
                        continue
                    
                    pair_window_idx[max(0, int((end_t - 1) * end_t / 2), int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                check_pair_window_idx.append(pair_window_idx)
            
            return torch.stack(check_pair_window_idx)


        if allocated_gpu == 0:
            logger = logging.getLogger(option)
        
        if option == 'valid':
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset

        with torch.no_grad():
            valid_dataloader = self.get_dataloader(dataset, batch_size, num_worker, shuffle=False, contain_context=self.contain_context)

            self.distributed_model.eval()
            loss_avg, count= 0, 0
            emo_pred_y_list, emo_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, cau_pred_y_list, cau_true_y_list = [list() for _ in range(6)]

            for utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch in tqdm(valid_dataloader, desc=f"{option}"):
                batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape

                check_pad_idx = get_pad_idx(utterance_input_ids_batch)

                prediction = self.distributed_model(
                                                    utterance_input_ids_batch, 
                                                    utterance_attention_mask_batch, 
                                                    utterance_token_type_ids_batch, 
                                                    speaker_batch
                                                    )

                if len(prediction) != 2:
                    emotion_prediction, binary_cause_prediction = prediction
                else:
                    emotion_prediction, binary_cause_prediction = prediction

                check_pair_window_idx = get_pair_pad_idx(
                                                        utterance_input_ids_batch, 
                                                        window_constraint=3, 
                                                        emotion_pred=emotion_prediction
                                                        )
                check_pair_pad_idx = get_pair_pad_idx(
                                                    utterance_input_ids_batch, 
                                                    window_constraint=1000
                                                    )

                emotion_prediction = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
                binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                binary_cause_prediction_all = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                
                emotion_label_batch = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]

                if self.n_cause == 2:
                    pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                    pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                else:
                    pair_cause_label_batch = torch.argmax(pair_cause_label_batch.view(-1, self.n_cause), dim=1).view(batch_size, -1)

                    pair_binary_cause_label_batch_window = pair_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                    pair_binary_cause_label_batch_all = pair_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]

                criterion_emo = FocalLoss(gamma=2)
                criterion_cau = FocalLoss(gamma=2)

                loss_emo = criterion_emo(emotion_prediction, emotion_label_batch.to(allocated_gpu))
                
                if torch.sum(check_pair_window_idx) == 0:
                    loss = loss_emo
                else:
                    loss_cau = criterion_cau(binary_cause_prediction_window, pair_binary_cause_label_batch_window.to(allocated_gpu))
                    loss = 0.2 * loss_emo + 0.8 * loss_cau

                cau_pred_y_list_all.append(binary_cause_prediction_all), cau_true_y_list_all.append(pair_binary_cause_label_batch_all)
                cau_pred_y_list.append(binary_cause_prediction_window), cau_true_y_list.append(pair_binary_cause_label_batch_window)
                emo_pred_y_list.append(emotion_prediction), emo_true_y_list.append(emotion_label_batch)

                loss_avg += loss.item()
                count += 1

            loss_avg = loss_avg / count

            if allocated_gpu == 0:
                p_cau, r_cau, f1_cau = log_metrics(logger, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg, n_cause=self.n_cause, option=option)
            del valid_dataloader

            if option == 'valid' and allocated_gpu == 0:
                saver(self.distributed_model)

                return 0
            
            if option == 'test' and allocated_gpu == 0:
                f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                if self.best_performance[-1] < f1_cau:
                    self.best_performance = [p_cau, r_cau, f1_cau]
                
                p, r, f1 = self.best_performance
                logger.info(f'\n[current best performance] precision: {p} | recall: {r} | f1-score: {f1}\n')

    def run(self, **kwargs):
        self.multiprocess_work(**kwargs)

    def infer(self, conversation): # conversation: {doc_id:content}, content [[{...}, {...}, ...]]
        with torch.no_grad(): 
            self.distributed_model.eval()
            
            # inputs: (utterance_input_ids, utterance_attention_mask, utterance_token_type_ids, speaker_info)
            inputs = tokenize_conversation(conversation, 0, self.max_seq_len)
            emotion_prediction, binary_cause_prediction = self.distributed_model(*inputs)

        return emotion_prediction, binary_cause_prediction


class ModelSaver:
    def __init__(self, path='checkpoint.pt', single_gpu=None):
        self.path = path
        self.single_gpu = single_gpu

    def __call__(self, model):
        state_dict = model.module.state_dict() if not self.single_gpu else model.state_dict()
        torch.save(state_dict, self.path)
