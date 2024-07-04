import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,Subset,random_split
from selfonn import SelfONN1DLayer
from tqdm import tqdm
from models import models
from utils1 import ECGDataset,get_patient_ids,calc_accuracy
from sklearn.metrics import classification_report,confusion_matrix
from cgd import CGD
from torch import nn, optim
def calc_accuracyMSE(output,target): return ((output.squeeze()-target)*(output.squeeze()-target)).float().mean()
# parse paths
# patient_ids=['78']
patient_ids=['1']

# patient_ids=[]
# for i in range(0,5):
#     patient_ids.append(str((i+1)))
# patient_ids=['75']
data_path = "mats/sub128v4_{patient_id}_{split}"
num_epochs = 100
q_range= [1]
validation_loss=np.zeros((1,num_epochs+1))
training_loss=[]
predicted = []
actual = []

for q in q_range:
 for mode in ['model_full']:
    model_path = Path("q"+str(q)+'_'+mode+'_ch12_sgdonly')
    out_path = Path("q"+str(q)+'_'+mode+'_ch12_sgdonly'); out_path.mkdir(exist_ok=True)
    print(out_path)
    #print(model._modules['0'].q)
    count_train = 0
    count_test = 0
    for patient_id in patient_ids:
        print("Patient: ", patient_id)
        train_ds = ECGDataset(data_path,patient_id,"train")
        train_ds = random_split(train_ds,[int(0.8*len(train_ds)),len(train_ds)-int(0.8*len(train_ds))])
        test_ds = ECGDataset(data_path,patient_id,"test")
        train_dl = DataLoader(train_ds[0],batch_size=10,shuffle=True)
        val_dl = DataLoader(train_ds[1],batch_size=10)
        # train_dl = DataLoader(train_ds,batch_size=8,shuffle=True)
        test_dl = DataLoader(test_ds,batch_size=10)
        # TRAINING 
        best_val_loss = 5e9
        best_train_loss = 1e9
        # training_loss[:,0]=1e9
        predicted =[]
        actual =[]
        for run in range(1):
            # define model
            model = models[mode](q)
            model = model.cuda()
            # break
            optim1 =  optim.Adam(model.parameters(),lr=0.0001)
            epochs = tqdm(range(num_epochs))
            # learning_rate=0.1
            for epoch in epochs:
                # optim = torch.optim.CGD(model.parameters(), lr=learning_rate, momentum=0.9)
                train_acc = []
                val_acc = []

                train_loss = []
                model.train()
                for batch in (train_dl):
                    optim1.zero_grad()
                    data = batch[0].cuda()
                    label = batch[1].cuda()
                    output = model(data)
                    # loss = torch.nn.MSELoss()(output.squeeze(-1),label)
                    loss = nn.MSELoss()
                    outputs = loss(label, output.squeeze(-1))
                    outputs.backward()
                    # loss.backward()
                    optim1.step()
                    # train_loss.append(loss.item())
                    train_acc.append(torch.nn.MSELoss()(output.data,label.data).item())
                train_loss  = np.mean(train_acc)
                # optim.setLR(loss_now)               
                for batch in (val_dl):
                    data2 = batch[0].cuda()
                    label2 = batch[1].cuda()
                    output2 = model(data2)
                    # loss = torch.nn.MSELoss()(output.squeeze(-1),label)
                    outputs = loss(label2, output2.squeeze(-1))
                    val_acc.append(torch.nn.MSELoss()(output2.data,label2.data).item())
                loss_now = np.mean(val_acc)
                epochs.set_postfix({'loss':loss_now}) 
                training_loss.append(loss_now)
                if loss_now<best_val_loss:
                    # print("Ep")
                    best_val_loss = loss_now
                    torch.save(model,out_path.joinpath('patient_{0}.pth'.format(patient_id)))                  
# plt.plot(training_loss)
data_path = "mats/sub128v4_{patient_id}_{split}"
for q in q_range:
    for mode in ['model_full']:
        model_path = Path("q"+str(q)+'_'+mode+'_ch12_sgdonly')
        print(model_path)
        for patient_id in patient_ids:
            predicted = []
            actual = []
            print("Patient: ", patient_id)
            # define model
            model = torch.load(model_path.joinpath('patient_{0}.pth'.format(patient_id)))
            model.eval()
            # dataloading
            test_ds = ECGDataset(data_path,patient_id,"test")
            test_dl = DataLoader(test_ds,batch_size=1000)
            # evaluation
            with torch.no_grad():
                for batch in (test_dl):
                    data = batch[0].cuda()
                    label = batch[1].cuda()
                    output = model(data).squeeze()
                    predicted += output.argmax(-1).cpu().numpy().tolist()
                    actual += label.cpu().numpy().tolist()               
            np.savez(model_path.joinpath("predictions{0}.npz".format(patient_id)),actual,predicted)                
import pandas as pd
mod_name="full"
acc=[]
f11=[]
f12=[]
for q in q_range:
    aaa=[]
    aaa=pd.DataFrame(data=aaa)
    ppp=[]
    ppp=pd.DataFrame(data=ppp)
    for mode in [mod_name]:
        # model_path = Path("q"+str(q)+'_paper_struct_ch12_sgdonly')
        model_path = Path("q"+str(q)+'_model_full_ch12_sgdonly')
        print("==== {0} ====".format(model_path))
        for patient_id in patient_ids:
            print(patient_id)
            with np.load(model_path.joinpath("predictions{0}.npz".format(patient_id))) as data:
                actual = data['arr_0']
                actual=actual.argmax(-1)
                actual=pd.DataFrame(data=actual)

                predicted = data['arr_1']
                predicted=pd.DataFrame(data=predicted)
                a=classification_report(actual,predicted,digits=4,output_dict=True)
                aa=a["accuracy"]
                acc.append(aa)
                f111=a["0"]["recall"]
                f11.append(f111)
                f112=a["1"]["recall"]
                f12.append(f112)
                # # aaa=pd.concat([aaa,actual])
                # # ppp=pd.concat([ppp,predicted])
                # print(patient_id)
                print(classification_report(actual,predicted,digits=4))
                print(confusion_matrix(actual,predicted))   
                
