import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
import pickle
from sklearn.metrics import confusion_matrix
import patchify

#Dir and parameters
layer=4
ModelDir='ML/pebble_unet_nadir9obl4/'

train_indexL=np.load('ML/kfold_train_indexL.npy', allow_pickle=True)
test_indexL=np.load('ML/kfold_test_indexL.npy', allow_pickle=True)

tt=[0,1,2,3,4,5,6,7,8,13,15,17,24]
val=[9,10,11,12,14,16,18,19,20,21,22,23]

# Load the images
fn_img = glob.glob("ML/output*/")
fn_img.sort()
imgall=[]

for fn in fn_img:
    #import image with depth
    image = np.load(fn+'stats.npy')
    imageD=image.reshape(1242, 2208, 5)[:,:,0]
    imageD-=np.min(imageD)
    imageD/=np.max(imageD)
    imageD=imageD*255
    image = cv2.imread(fn+'imgL.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.dstack([image,imageD])
    image=image.astype('uint8')
    imgall.append(image)
maskall=[]
mask01=[]
i=0
for fn in fn_img[:13]:
    #mask
    mask=np.load(fn+f'TF_mask{i:02}.npy')
    i+=1
    maskall.append(mask)
    mt=mask.copy()
    mt[mt>0]=1
    mask01.append(mt.astype('uint8'))

fn_msk = glob.glob("ML/napari/*_napari.npz")
fn_msk.sort()
for fn in fn_msk[2:]:
    #import image with depth
    mask=np.load(fn)
    maskall.append(mask['arr_0'])
    mt=mask['arr_0'].copy()
    mt[mt>0]=1
    mask01.append(mt.astype('uint8'))

pty_train=[]
pty_mask=[]

for i in tt:
    #patchify
    pty_img = patchify.patchify(imgall[i], (128,128,layer),step=118)
    pty_msk = patchify.patchify(mask01[i], (128,128),step=118)
    pty_train.append(pty_img.reshape(pty_img.shape[0]*pty_img.shape[1],128,128,layer))
    pty_mask.append(pty_msk.reshape(pty_img.shape[0]*pty_img.shape[1],128,128))

pty_train=np.vstack(pty_train)
pty_mask=np.vstack(pty_mask)
label = np.array(pty_mask, dtype="int")
train = np.array(pty_train)
label = np.expand_dims(label, axis=-1)

desired_shape = (2304, 2304, layer)
#resize for prediction
imgallSQ=[]
for img in imgall:
    # Create a new array with the desired shape, filled with zeros
    new_rgb_depth_image = np.zeros(desired_shape, dtype=img.dtype)

    # Copy the original image into the new array
    new_rgb_depth_image[0:img.shape[0], 
                        0:img.shape[1], :] = img[:,:,:layer]

    new_rgb_depth_image=np.expand_dims(new_rgb_depth_image,axis=0)
    imgallSQ.append(new_rgb_depth_image)


#kfold
TR=2#takes all metrics for 0.7 threshold
Allmetrics=[[],[],[],[]]
Allmetricsval=[[],[],[],[]]
for kfi in range(8):
    modelname=f'pebble_unet_nadir9obl4_kfold{kfi}_best_weights.h5'

    x_train, x_test, y_train, y_test = train[train_indexL[kfi]], train[test_indexL[kfi]], label[train_indexL[kfi]], label[test_indexL[kfi]]

    model = tf.keras.models.load_model(
        ModelDir+'checkpoints/'+modelname, compile=True
    )
    print(modelname, 'loaded')

    y_pred = model.predict(x_test)

    y_predall=[]
    for x in imgallSQ:
        y_pred = model.predict(x)
        y_predall.append(y_pred)
    #save predicted mask
    for fn,i in zip(fn_img,range(len(y_predall))):
        np.save(fn+modelname[:-3]+'.npy',y_predall[i][0,:1242,:2208])

    #acuracy
    tL=[0.5,0.6,0.7,0.8,0.9]
    RcL=[]
    RcLval=[]
    for t in tL:
        m = tf.keras.metrics.Recall(thresholds=t)
        m.update_state(mask01, np.vstack(y_predall)[:,:1242,:2208,0])
        RcL.append(m.result().numpy())
        m = tf.keras.metrics.Recall(thresholds=t)
        m.update_state(mask01[val], np.vstack(y_predall)[val][:,:1242,:2208,0])
        RcLval.append(m.result().numpy())
    Allmetrics[0].append(RcL[TR])
    Allmetricsval[0].append(RcLval[TR])
    PcL=[]
    PcLval=[]
    for t in tL:
        m = tf.keras.metrics.Precision(thresholds=t)
        m.update_state(mask01, np.vstack(y_predall)[:,:1242,:2208,0])
        PcL.append(m.result().numpy())
        m = tf.keras.metrics.Precision(thresholds=t)
        m.update_state(mask01[val], np.vstack(y_predall)[val][:,:1242,:2208,0])
        PcLval.append(m.result().numpy())
    Allmetrics[1].append(PcL[TR])
    Allmetricsval[1].append(PcLval[TR])  
    iouL=[]
    iouLval=[]
    for t in tL:
        m = tf.keras.metrics.BinaryIoU(threshold=t)
        m.update_state(mask01, np.vstack(y_predall)[:,:1242,:2208,0])
        iouL.append(m.result().numpy())
        m = tf.keras.metrics.BinaryIoU(threshold=t)
        m.update_state(mask01[val], np.vstack(y_predall)[val][:,:1242,:2208,0])
        iouLval.append(m.result().numpy())
    Allmetrics[2].append(iouL[TR])
    Allmetricsval[2].append(iouLval[TR])
    f1L=[]
    f1Lval=[]
    for t in tL:
        m = tf.keras.metrics.F1Score(threshold=t)
        m.update_state(mask01, np.vstack(y_predall)[:,:1242,:2208,0])
        f1L.append(m.result().numpy())
        m = tf.keras.metrics.F1Score(threshold=t)
        m.update_state(mask01[val], np.vstack(y_predall)[val][:,:1242,:2208,0])
        f1Lval.append(m.result().numpy())
    Allmetrics[3].append(f1L[TR])
    Allmetricsval[3].append(f1Lval[TR])
        
    plt.plot(tL,RcL, label='Recall')
    plt.plot(tL,PcL,label='Precision')
    plt.plot(tL,iouL,label='BinaryIoU')
    plt.plot(tL,f1L,label='F1Score')
    plt.plot(tL,RcLval, label='Recall test only', linestyle='--')
    plt.plot(tL,PcLval,label='Precision test only', linestyle='--')
    plt.plot(tL,iouLval,label='BinaryIoU test only', linestyle='--')
    plt.plot(tL,f1Lval,label='F1Score test only', linestyle='--')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid()
    plt.title(f'{modelname[:-3]}\nMetrics vs. threshold')
    plt.savefig(ModelDir+'training_history/'+modelname[:-3]+'_modelaccuracy.png')
    plt.show()



    fig, (ax1,ax2,ax3)=plt.subplots(1,3, figsize=(15, 5))
    ax=[ax1,ax2,ax3]
    for a,T in enumerate([0.5,0.7,0.9]):
        ACL=[]

        for i in val:
            confusion = confusion_matrix(mask01[i].flatten(), (y_predall[i][0,:1242,:2208,0]>T).flatten())
            accuracy = (confusion[0,0] + confusion[1,1]) / np.sum(confusion)
            precision = confusion[1,1] / np.sum(confusion[:,1])
            recall = confusion[1,1] / np.sum(confusion[1])
            f1 = 2*confusion[1,1]  / (np.sum(confusion[1]) + np.sum(confusion[:,1]))
            ACL.append([precision,recall,f1, accuracy])

            
        
        ax[a].plot(np.array(ACL)[:,3], label='accuracy')
        ax[a].plot(np.array(ACL)[:,0], label='precision')
        ax[a].plot(np.array(ACL)[:,1], label='recall')
        ax[a].plot(np.array(ACL)[:,2], label='f1')
        ax[a].set_ylim(0.2,1)
        ax[a].set_xlabel('image')
        ax[a].set_ylabel('ratio')
        ax[a].grid()
        ax[a].legend()
        ax[a].set_title(f'Threshold: {T}')

    plt.suptitle(f'{modelname[:-3]}\nAccuracy, precision, recall and f1 score')
    plt.tight_layout()
    plt.savefig(ModelDir+'training_history/'+modelname[:-3]+'_accuracy.png')
    plt.show()

plt.plot(Allmetrics[0], label='Recall')
plt.plot(Allmetrics[1],label='Precision')
plt.plot(Allmetrics[2],label='BinaryIoU')
plt.plot(Allmetrics[3],label='F1Score')
plt.plot(Allmetricsval[0], label='Recall test only', linestyle='--')
plt.plot(Allmetricsval[1],label='Precision test only', linestyle='--')
plt.plot(Allmetricsval[2],label='BinaryIoU test only', linestyle='--')
plt.plot(Allmetricsval[3],label='F1Score test only', linestyle='--')
plt.xlabel('kfold')
plt.legend()
plt.grid()
plt.title(f'kfold consistency')
plt.savefig(ModelDir+'training_history/kfold_consistency.png')