from numpy import hamming
from .preprocessing import *
import module.model as M
from .evaluation import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging, datetime
import sklearn

import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class learning_env:
    def __init__(self, gpus, train_data, valid_data, test_data, split_directory, max_seq_len, log_directory, model_name, port, contain_context, data_label, **kwargs) -> None:
        self.gpus = gpus
        self.single_gpu = True if len(self.gpus) == 1 else False

        self.train_dataset, self.valid_dataset, self.test_dataset = train_data, valid_data, test_data

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

        self.best_performance = [0, 0, 0] # p, r, f1


    def __set_model__(self, pretrained_model, dropout, n_speaker, n_emotion, n_cause, n_expert, guiding_lambda, **kwargs):
        self.n_cause = n_cause

        model_args = {'dropout':dropout, 'n_speaker':n_speaker, 'n_emotion':n_emotion, 'n_cause':n_cause, 'n_expert':n_expert, 'guiding_lambda':guiding_lambda}

        if pretrained_model != None:
            model = getattr(M, self.model_name)(**model_args)
            model.load_state_dict(torch.load(pretrained_model))
            return model
        else:
            model = getattr(M, self.model_name)(**model_args)
            
            return model

    def set_model(self, allocated_gpu):
        if not self.single_gpu:
            torch.distributed.init_process_group(
                backend='gloo',
                init_method=f'tcp://127.0.0.1:{self.port}',
                world_size=len(self.gpus),
                rank=allocated_gpu)

        torch.cuda.set_device(allocated_gpu)

        model = self.__set_model__(**self.options).cuda(allocated_gpu)
        
        if self.single_gpu:
            self.distributed_model = model
        else:
            self.distributed_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[allocated_gpu], find_unused_parameters=True)

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
            if not os.path.exists(f'log/{self.log_directory}'):
                os.makedirs(f'log/{self.log_directory}')
            file_handler = logging.FileHandler(f'log/{self.log_directory}/{file_name}')
        else:
            file_handler = logging.FileHandler(f'log/{file_name}')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_dataloader(self, dataset_file, batch_size, num_worker, shuffle=True, contain_context=False):
        (utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t), speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = get_data(dataset_file, f"cuda:0", self.max_seq_len, contain_context)
        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
        if self.single_gpu:
            return DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset_, shuffle=shuffle)
            pin_memory = False

            return DataLoader(dataset=dataset_, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_worker, sampler=train_sampler)


    def init_stopper(self):
        self.stopper[0] = 0

    def multiprocess_work(self, test, training_iter, batch_size, learning_rate, patience, num_worker, **kwargs):
        stopper = torch.zeros(1)
        stopper.share_memory_()

        if self.split_directory:
            self.set_logger_environment([f'{self.model_name}-split_average-{self.start_time}.log'], ['split_logger'])
            logger = logging.getLogger('split_logger')

            split_performance = torch.zeros((3, len(os.listdir(self.split_directory)), 5))
            split_performance.share_memory_()

            for _ in os.listdir(self.split_directory):
                self.split_num = _.split('_')[-1]

                base_file_name = os.path.join(self.split_directory, _, f"split_{_.split('_')[-1]}")
                self.train_dataset, self.valid_dataset, self.test_dataset = base_file_name + '_train.json', base_file_name + '_valid.json', base_file_name + '_test.json'
                
                self.start_time = datetime.datetime.now()

                if self.single_gpu:
                    self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test)
                else:
                    torch.multiprocessing.spawn(self.child_process, nprocs=len(self.gpus), args=(training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test))
            
            logger.info(f"Emotion Classificaiton Test Average Performance | {len(os.listdir(self.split_directory))} trials | {torch.mean(split_performance[0], dim = 0)}\n")
            logger.info(f"Binary Cause Classificaiton Test Average Performance | {len(os.listdir(self.split_directory))} trials | {torch.mean(split_performance[1], dim = 0)}\n")
            logger.info(f"MultiClass Cause Classificaiton Test Average Performance | {len(os.listdir(self.split_directory))} trials | {torch.mean(split_performance[2], dim = 0)}\n")
            
        else:
            if self.single_gpu:
                self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test)
            else:
                torch.multiprocessing.spawn(self.child_process, nprocs=len(self.gpus), args=(training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test))

    # ##############################################################################################################################################################################################

    def child_process(self, allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test=False):
        batch_size = int(batch_size / len(self.gpus))
        num_worker = int(num_worker / len(self.gpus))

        self.set_model(allocated_gpu) 

        if allocated_gpu == 0:
            logger_name_list = ['train', 'valid', 'test']

            if self.n_cause == 2:
                file_name_list = [f'{self.model_name}-binary_cause-{_}-{self.start_time}.log' for _ in ['train', 'valid', 'test']]
            else:
                file_name_list = [f'{self.model_name}-multiclass_cause-{_}-{self.start_time}.log' for _ in ['train', 'valid', 'test']]

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

            if emotion_pred != None:
                emotion_pred = torch.argmax(emotion_pred, dim=1)
                
                check_pair_window_idx = list()
                for batch in check_pad_idx.view(-1, max_doc_len):
                    pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))
                    for end_t in range(1, len(batch.nonzero()) + 1):
                        if emotion_pred[end_t - 1] == 6:
                            continue
                        
                        pair_window_idx[max(0, int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                    check_pair_window_idx.append(pair_window_idx)
                check_pair_window_idx = torch.stack(check_pair_window_idx)

                return check_pair_window_idx
            else:
                check_pair_window_idx = list()
                for batch in check_pad_idx.view(-1, max_doc_len):
                    pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))
                    for end_t in range(1, len(batch.nonzero()) + 1):
                        pair_window_idx[max(0, int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                    check_pair_window_idx.append(pair_window_idx)
                check_pair_window_idx = torch.stack(check_pair_window_idx)

                return check_pair_window_idx

        if allocated_gpu == 0:

            self.init_stopper()

            logger = logging.getLogger('train')

        optimizer = optim.Adam(self.distributed_model.parameters(), lr=learning_rate)

        if self.n_cause == 2:
            saver = model_saver(path=f"model/{self.model_name}-binary_cause-{self.data_label}-{self.start_time}.pt", single_gpu=self.single_gpu)
        else:
            saver = model_saver(path=f"model/{self.model_name}-multiclass_cause-{self.data_label}-{self.start_time}.pt", single_gpu=self.single_gpu)

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

                prediction = self.distributed_model(utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch)

                if len(prediction) != 2:
                    emotion_prediction, binary_cause_prediction = prediction
                else:
                    emotion_prediction, binary_cause_prediction = prediction
                
                check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, window_constraint=3, emotion_pred=emotion_prediction)
                check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, window_constraint=1000)

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
                loss_cau = criterion_cau(binary_cause_prediction_window, pair_binary_cause_label_batch_window.to(allocated_gpu))
                loss = 0.2 * loss_emo + 0.8 * loss_cau

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cau_pred_y_list_all.append(binary_cause_prediction_all), cau_true_y_list_all.append(pair_binary_cause_label_batch_all)

                cau_pred_y_list.append(binary_cause_prediction_window), cau_true_y_list.append(pair_binary_cause_label_batch_window)

                emo_pred_y_list.append(emotion_prediction), emo_true_y_list.append(emotion_label_batch)

                loss_avg = loss_avg + loss.item(); count += 1

            loss_avg = loss_avg / count

            # Logging Performance
            if allocated_gpu == 0:
                label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
                logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
                report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
                acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
                logger.info(f'\nemotion: train | loss {loss_avg}\n')

                if self.n_cause == 2:
                    label_ = np.array(['No Cause', 'Cause'])

                    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)
                    _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

                    report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)
                    acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                    logger.info(f'\nbinary_cause: train | loss {loss_avg}\n')
                    logger.info(f'\nbinary_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')
                else:
                    label_ = np.array(['no-context', 'inter-personal', 'self-contagion', 'no cause'])
                    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)

                    p_no_context, p_inter_personal, p_self_contagion = report_dict['no-context']['precision'], report_dict['inter-personal']['precision'], report_dict['self-contagion']['precision']

                    p_cau = (report_dict['no-context']['support'] * report_dict['no-context']['precision'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['precision'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['precision']) / \
                            (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )
                    report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)

                    r_no_context, r_inter_personal, r_self_contagion = report_dict['no-context']['recall'], report_dict['inter-personal']['recall'], report_dict['self-contagion']['recall']

                    acc_cau = report_dict['accuracy']
                    r_cau = (report_dict['no-context']['support'] * report_dict['no-context']['recall'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['recall'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['recall']) / \
                            (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )

                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                    logger.info(f'\nmulticlass_cause: train | loss {loss_avg}\n')

                    logger.info(f'\nmulticlass_cause: no-context    | precision: {p_no_context} | recall: {r_no_context} | f1-score: {2 * p_no_context * r_no_context / (p_no_context + r_no_context) if p_no_context + r_no_context != 0 else 0}\n')
                    logger.info(f'multiclass_cause: inter-personal  | precision: {p_inter_personal} | recall: {r_inter_personal} | f1-score: {2 * p_inter_personal * r_inter_personal / (p_inter_personal + r_inter_personal) if p_inter_personal + r_inter_personal != 0 else 0}\n')
                    logger.info(f'multiclass_cause: self-contagion  | precision: {p_self_contagion} | recall: {r_self_contagion} | f1-score: {2 * p_self_contagion * r_self_contagion / (p_self_contagion + r_self_contagion) if p_self_contagion + r_self_contagion != 0 else 0}\n')

                    logger.info(f'\nmulticlass_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')

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

            if emotion_pred != None:
                emotion_pred = torch.argmax(emotion_pred, dim=1)
                
                check_pair_window_idx = list()
                for batch in check_pad_idx.view(-1, max_doc_len):
                    pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))

                    for end_t in range(1, len(batch.nonzero()) + 1):
                        if emotion_pred[end_t - 1] == 6:
                            continue
                        pair_window_idx[max(0, int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                    check_pair_window_idx.append(pair_window_idx)
                check_pair_window_idx = torch.stack(check_pair_window_idx)

                return check_pair_window_idx
            else:
                check_pair_window_idx = list()
                for batch in check_pad_idx.view(-1, max_doc_len):
                    pair_window_idx = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2))

                    for end_t in range(1, len(batch.nonzero()) + 1):
                        pair_window_idx[max(0, int(end_t * (end_t + 1) / 2) - window_constraint):int(end_t * (end_t + 1) / 2)] = 1
                    check_pair_window_idx.append(pair_window_idx)
                check_pair_window_idx = torch.stack(check_pair_window_idx)

                return check_pair_window_idx


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

                prediction = self.distributed_model(utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch)

                if len(prediction) != 2:
                    emotion_prediction, binary_cause_prediction = prediction
                else:
                    emotion_prediction, binary_cause_prediction = prediction

                check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, window_constraint=3, emotion_pred=emotion_prediction)
                check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, window_constraint=1000)

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
                loss_cau = criterion_cau(binary_cause_prediction_window, pair_binary_cause_label_batch_window.to(allocated_gpu))

                loss = 0.2 * loss_emo + 0.8 * loss_cau

                cau_pred_y_list_all.append(binary_cause_prediction_all), cau_true_y_list_all.append(pair_binary_cause_label_batch_all)
                cau_pred_y_list.append(binary_cause_prediction_window), cau_true_y_list.append(pair_binary_cause_label_batch_window)
                emo_pred_y_list.append(emotion_prediction), emo_true_y_list.append(emotion_label_batch)

                loss_avg = loss_avg + loss.item(); count += 1

            loss_avg = loss_avg / count
            if allocated_gpu == 0:
                label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
                logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
                report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
                acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
                logger.info(f'\nemotion: {option} | loss {loss_avg}\n')

                if self.n_cause == 2:
                    label_ = np.array(['No Cause', 'Cause'])

                    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)
                    _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

                    report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)
                    acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']

                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                    logger.info(f'\nbinary_cause: valid | loss {loss_avg}\n')
                    logger.info(f'\nbinary_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')
                else:
                    label_ = np.array(['no-context', 'inter-personal', 'self-contagion', 'no cause'])
                    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)

                    p_no_context, p_inter_personal, p_self_contagion = report_dict['no-context']['precision'], report_dict['inter-personal']['precision'], report_dict['self-contagion']['precision']

                    p_cau = (report_dict['no-context']['support'] * report_dict['no-context']['precision'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['precision'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['precision']) / \
                            (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )

                    report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)

                    r_no_context, r_inter_personal, r_self_contagion = report_dict['no-context']['recall'], report_dict['inter-personal']['recall'], report_dict['self-contagion']['recall']

                    acc_cau = report_dict['accuracy']
                    r_cau = (report_dict['no-context']['support'] * report_dict['no-context']['recall'] + report_dict['inter-personal']['support'] * report_dict['inter-personal']['recall'] + report_dict['self-contagion']['support'] * report_dict['self-contagion']['recall']) / \
                            (report_dict['no-context']['support'] + report_dict['inter-personal']['support'] + report_dict['self-contagion']['support'] )

                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                    logger.info(f'\nmulticlass_cause: valid | loss {loss_avg}\n')

                    logger.info(f'\nmulticlass_cause: no-context    | precision: {p_no_context} | recall: {r_no_context} | f1-score: {2 * p_no_context * r_no_context / (p_no_context + r_no_context) if p_no_context + r_no_context != 0 else 0}\n')
                    logger.info(f'multiclass_cause: inter-personal  | precision: {p_inter_personal} | recall: {r_inter_personal} | f1-score: {2 * p_inter_personal * r_inter_personal / (p_inter_personal + r_inter_personal) if p_inter_personal + r_inter_personal != 0 else 0}\n')
                    logger.info(f'multiclass_cause: self-contagion  | precision: {p_self_contagion} | recall: {r_self_contagion} | f1-score: {2 * p_self_contagion * r_self_contagion / (p_self_contagion + r_self_contagion) if p_self_contagion + r_self_contagion != 0 else 0}\n')

                    logger.info(f'\nmulticlass_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')

            if not self.single_gpu:
                torch.distributed.barrier()

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
            
            utterance_input_ids, utterance_attention_mask, utterance_token_type_ids, speaker_info = tokenize_conversation(conversation, 0, self.max_seq_len)

            # Running Model
            emotion_prediction, binary_cause_prediction = self.distributed_model(utterance_input_ids, utterance_attention_mask, utterance_token_type_ids, speaker_info)

        return emotion_prediction, binary_cause_prediction


class model_saver:
    def __init__(self, path='checkpoint.pt', single_gpu=None):
        self.path = path
        self.single_gpu = single_gpu

    def __call__(self, model):
        if self.single_gpu:
            torch.save(model.state_dict(), self.path)
        else:
            torch.save(model.module.state_dict(), self.path)