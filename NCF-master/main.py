import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import random
#from tensorboardX import SummaryWriter
import scipy.sparse as sp
import model
import config
import evaluate
import precess_data
seed_value=42

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num ,item_num = precess_data.load_dataset()

# # construct the train and test datasets
# train_dataset = precess_data.NCFData(
# 		train_data, item_num, train_mat, args.num_ng, True)
# test_dataset = precess_data.NCFData(
# 		test_data, item_num, train_mat, 0, False)
# train_loader = data.DataLoader(train_dataset,
# 		batch_size=args.batch_size, shuffle=True, num_workers=4)
# test_loader = data.DataLoader(test_dataset,
# 		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
# ########################### CREATE MODEL #################################
# GMF_model = None
# MLP_model = None
# posion_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
# model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
#不知道干嘛的
np.random.seed(seed_value)
#seed_datasets = np.random(100000,size = 2*user_num)
############################### Start generating poison ###############################
#所有物品的初始概率都是1，用来挑选target的填充物，因此要减1
#填充物列表
filler_items=[]
prob = torch.ones(item_num-1).cuda()
m0 = len(filler_items)
#就生成两个虚假用户
steps = [1,1]
#开始生成虚假用户
target_item = 100
print("Start generating posion model")
for i,step in enumerate(steps):
	candidate_users,candidate_items = [],[]
	print("start 2 fake user")
	for j in range(step):
		# candidate_users.extend([user_num+m0+i+j]*(item_num-1))
		# tmp=list(range(item_num))
		# tmp.remove(target_item)
		# candidate_items.extend(tmp)
		train_data, test_data, user_num ,item_num, train_mat = precess_data.fake_insert(user_num,target_item)
		train_dataset = precess_data.NCFData(train_data, item_num, train_mat, args.num_ng, True)
		test_dataset = precess_data.NCFData(test_data, item_num, train_mat, 0, False)
		train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=4)
		test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)
	########################### CREATE MODEL #################################
	GMF_model = None
	MLP_model = None
	posion_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, config.model, GMF_model, MLP_model)
	posion_model.cuda()
	optimizer = optim.Adam(posion_model.parameters(), lr=args.lr)
	posion_model.train()
	###写到这里了，先预先训练posion——model
	best_ndcg, best_hr = 0, 0
	#for epoch in range(args.epochs):
	#将假用户插入到模型进行训练，并保存最佳hr时的模型
	for epoch in range(5):
		posion_model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item, label in train_loader:
			user = user.cuda()
			item = item.cuda()
			label = label.float().cuda()
			posion_model.zero_grad()
			prediction = posion_model(user, item)
			loss_function = nn.BCEWithLogitsLoss()
			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()

		posion_model.eval()
		HR, NDCG = evaluate.metrics(posion_model, test_loader, args.top_k)
		if HR > best_hr:
			best_hr, best_ndcg = HR, NDCG
			torch.save(posion_model.state_dict(),'{}{}.pth'.format(config.model_path, config.model))
		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
			  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
		print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
	posion_model.load_state_dict(torch.load('{}{}.pth'.format(config.model_path, config.model)))
	torch.save(posion_model.state_dict(),'{}{}.pth'.format(config.model_path, config.model))
	target_hr,target_ndcg = evaluate.target_hr(posion_model,test_loader,args.top_k,target_item)
	print('Initial target HR: {:.4f}, best HR: {:.4f}'.format(np.mean(target_hr), np.mean(target_ndcg)))
# writer = SummaryWriter() # for visualization

# ########################### TRAINING #####################################
# count, best_hr = 0, 0
# for epoch in range(args.epochs):
# 	model.train() # Enable dropout (if have).
# 	start_time = time.time()
# 	train_loader.dataset.ng_sample()
#
# 	for user, item, label in train_loader:
# 		user = user.cuda()
# 		item = item.cuda()
# 		label = label.float().cuda()
#
# 		model.zero_grad()
# 		prediction = model(user, item)
# 		# 加权
# 		w = [1.0,41.0]
# 		weight = torch.zeros(label.shape)
# 		for i in range(label.shape[0]):
# 			weight[i] = w[int(label[i])]
# 		weight = weight.cuda()
# 		loss_function = nn.BCEWithLogitsLoss(weight=weight)
# 		loss = loss_function(prediction, label)
# 		loss.backward()
# 		optimizer.step()
# 		# writer.add_scalar('data/loss', loss.item(), count)
# 		count += 1
#
# 	model.eval()
# 	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
#
# 	elapsed_time = time.time() - start_time
# 	print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
# 			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
# 	print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
#
# 	if HR > best_hr:
# 		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
# 		if args.out:
# 			if not os.path.exists(config.model_path):
# 				os.mkdir(config.model_path)
# 			torch.save(model,
# 				'{}{}.pth'.format(config.model_path, config.model))
#
# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
# 									best_epoch, best_hr, best_ndcg))
