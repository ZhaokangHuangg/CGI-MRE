import os

import torch
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from .metrics import eval_result

from transformers import CLIPProcessor

class Trainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None, model=None, args=None, logger=None, writer=None,lr=None,weight_decay=None,lr2=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.re_dict = re_dict
        self.model = model
        self.logger = logger
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        self.writer = writer
        self.lr2=lr2
        self.lr=lr
        self.weight_decay=weight_decay
        self.step = 0
        self.args = args

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
            self.before_multimodal_train(lr,weight_decay)
        self.model.to(self.args.device)

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  lr = {} weight_decay={}".format(self.lr,self.weight_decay))
        self.logger.info("  lr2 = {} ".format(self.lr2))
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")


        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            train_loss = []
            a_loss=[]
            b_loss=[]
            c_loss=[]
            train_acc = []
            for epoch in range(1, self.args.num_epochs+1):

                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                #  # 在这里，batch 是一个包含一批训练数据的 Tensor，大小为 (batch_size, feature_dim)，可能还有一个标签维度
                #     # 对于图像数据，大小可能是 (batch_size, channels, height, width)，而标签可能为 (batch_size, num_classes)
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    # logits([32,23])  labels[32]

                    # zs
                    # (loss, logits), labels = self._step(batch, mode="train")
                    (loss, logits,a,b,c), labels = self._step(batch, mode="train")
                    # 记录
                    train_loss.append(loss.item())
                    a_loss.append(a.item())
                    b_loss.append(b.item())
                    c_loss.append(c.item())

                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        # self.writer.log({'avg_loss': avg_loss,'epoch ': epoch,'total_loss':loss,'contrastive_loss':a,'cmd_loss':b,'recon_loss':c})

                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    if self.dev_data:
                        self.evaluate(epoch)   # generator to dev.
                    if self.test_data:
                        self.test(epoch)
            
            pbar.close()
            self.pbar = None
            # 记录
            with open(os.path.join(self.args.cwd,"train_loss.txt" ),'a') as train_los:
                train_los.write(str(train_loss))
            with open(os.path.join(self.args.cwd,"train_loss_a.txt"),'a') as train_los:
                train_los.write(str(a_loss))
            with open(os.path.join(self.args.cwd,"train_loss_b.txt"),'a') as train_los:
                train_los.write(str(b_loss))
            with open(os.path.join(self.args.cwd,"train_loss_c.txt"),'a') as train_los:
                train_los.write(str(c_loss))

            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    (loss, logits,a,b,c), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    # 记录


                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()

                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                # acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                acc, micro_f1 , pre, recall = round(result['acc']*100, 4), round(result['micro_f1']*100, 4), round(result['micro_p']*100, 4), round(result['micro_r']*100, 4)

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}.,pre:{},recall:{}"\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc, pre, recall))
                # self.writer.log({'eva_f1': micro_f1, 'eva_accuracy': acc,'epoch': epoch,'total_loss':loss,'contrastive_loss':a,'cmd_loss':b,'recon_loss':c})
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1 # update best metric(f1 score)
                #     if self.args.save_path is not None:
                #         torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                #         self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        train_loss = []
        a_loss = []
        b_loss = []
        c_loss = []
        f1=[]
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    (loss, logits,a,b,c), labels = self._step(batch, mode="dev")    # logits: batch, 3     return outputs/labels
                    total_loss += loss.detach().cpu().item()
                    # 真实的label和预测的labels
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    # 记录
                    train_loss.append(loss.item())
                    a_loss.append(a.item())
                    b_loss.append(b.item())
                    c_loss.append(c.item())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)

                with open(os.path.join(self.args.cwd,"test_loss.txt"),'a') as train_los:
                    train_los.write(str(train_loss))
                with open(os.path.join(self.args.cwd,"test_loss_a.txt"),'a') as train_los:
                    train_los.write(str(a_loss))
                with open(os.path.join(self.args.cwd,"test_loss_b.txt"),'a') as train_los:
                    train_los.write(str(b_loss))
                with open(os.path.join(self.args.cwd,"test_loss_c.txt"),'a') as train_los:
                    train_los.write(str(c_loss))
                # acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                # total_loss = 0
                # self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}"\
                #             .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch, micro_f1, acc))
                # self.writer.log({'test_f1': micro_f1, 'test_accuracy': acc})

                acc, micro_f1, pre, recall = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4), round(
                    result['micro_p'] * 100, 4), round(result['micro_r'] * 100, 4)
                total_loss = 0
                self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}.,pre:{},recall:{}" \
                    .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch, micro_f1, acc, pre,
                            recall))
                # self.writer.log({'test_f1': micro_f1, 'test_accuracy': acc,'epoch': epoch,'total_loss':loss,'contrastive_loss':a,'cmd_loss':b,'recon_loss':c})
                # self.writer.log({'epoch': epoch})

                f1.append(micro_f1)

                with open(os.path.join(self.args.cwd,"f1.txt"),'a') as train_los:
                    train_los.write(str(f1))

                if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                    self.best_test_metric = micro_f1
                    self.best_test_epoch = epoch
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
        
        self.model.train()
    
    def predict(self):
        self.model.eval()
        self.logger.info("\n***** Running predicting *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels ,imgids = [], [],[]
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Predicting")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device

                    (loss, logits,a,b,c), labels,imgid = self._step_pre(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)

                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    #添加 imgid pred_label
                    imgids.extend(imgid)

                    pbar.update()

                    if  torch.equal(preds,labels):
                        print(imgid)

                # evaluate done
                pbar.close()
                sk_result = classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                # save predict results
                import os
                with open(os.path.join(self.args.cwd,'data/txt/result.txt'), 'w', encoding="utf-8") as wf:
                    wf.write(sk_result)
                    print('Successful write!!')

                my_dict = dict(zip(imgids, pred_labels))

                #打印结果
                with open(os.path.join(self.args.cwd,"实验结果.txt" ),"w", encoding="utf-8" )as file:
                    # 遍历实验结果字典
                    for name, result in my_dict.items():
                        # 将名称和数字写入文件，每个结果占一行
                        file.write(f"{name}: {result}\n")
                    print('2222Successful write!!')

                my_dict = dict(zip(imgids, true_labels))
                #打印结果
                with open(os.path.join(self.args.cwd,"实验结果zhen.txt" ),"w", encoding="utf-8" )as file:
                    # 遍历实验结果字典
                    for name, result in my_dict.items():
                        # 将名称和数字写入文件，每个结果占一行
                        file.write(f"{name}: {result}\n")
                    print('2222Successful write!!')



        self.model.train()

    def _step(self, batch, mode="train"):
        # deepke
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs,imgid = batch

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
        return outputs, labels

    def _step_pre(self, batch, mode="train"):
    # deepke
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs,imgid = batch

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
        return outputs, labels,imgid


        # hzk
    def step_hzk(self,batch,mode="train"):
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs = batch
        o = (self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs),1)
        return o,labels

    def before_multimodal_train(self,lr,weight_decay):
        optimizer_grouped_parameters = []

        # params = {'lr':self.args.lr, 'weight_decay':1e-2}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'model' in name and 'HZK2' not in name and 'ScaledDotProductAttention' not in name:
        #         params['params'].append(param)
        # optimizer_grouped_parameters.append(params)

        # params = {'lr':self.lr, 'weight_decay':self.weight_decay}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'HZK2' in name and 'ScaledDotProductAttention' not in name:
        #         params['params'].append(param)
        # optimizer_grouped_parameters.append(params)

        # params = {'lr':self.lr2, 'weight_decay':self.weight_decay}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'ScaledDotProductAttention' in name:
        #         params['params'].append(param)
        # optimizer_grouped_parameters.append(params)

        # 对一些参数指定学习
        no_decay = ["bias", "LayerNorm.weight"]
        ifa_param_optimizer = list(self.model.ifa_model.named_parameters())
        hzk2_param_optimizer = list(self.model.HZK2.named_parameters())
        sdp_param_optimizer = list(self.model.ScaledDotProductAttention.named_parameters())
        linear_param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in ifa_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay, 'lr': self.args.lr},
            {'params': [p for n, p in ifa_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.args.lr},

            {'params': [p for n, p in hzk2_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay, 'lr': self.lr},
            {'params': [p for n, p in hzk2_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.lr},

            {'params': [p for n, p in sdp_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay, 'lr': self.lr},
            {'params': [p for n, p in sdp_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.lr},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.weight_decay, 'lr': self.lr2},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.lr2}
        ]

        # AdamW优化器
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8)
        # 优化器，预热，总训练
        self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)


        # for name, par in self.model.named_parameters():
        #     print(name, par.requires_grad)
