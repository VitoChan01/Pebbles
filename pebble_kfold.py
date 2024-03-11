import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
import patchify

layer=4
kfi = 5

def conv_block(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(
        input
    )
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = tf.keras.models.Model(inputs, outputs, name="U-Net")
    return model

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
    #print(fn+f'TF_mask{i:02}.npy')
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
    #print(fn)
    mask=np.load(fn)
    maskall.append(mask['arr_0'])
    mt=mask['arr_0'].copy()
    mt[mt>0]=1
    mask01.append(mt.astype('uint8'))

pty_train=[]
pty_mask=[]
#newmaskall=[]
tt=[0,1,2,3,4,5,6,7,8,13,15,17,24]
val=[9,10,11,12,14,16,18,19,20,21,22,23]
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

train_indexL=np.load('ML/kfold_train_indexL.npy', allow_pickle=True)
test_indexL=np.load('ML/kfold_test_indexL.npy', allow_pickle=True)

x_train, x_test, y_train, y_test = train[train_indexL[kfi]], train[test_indexL[kfi]], label[train_indexL[kfi]], label[test_indexL[kfi]]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

input_shape = x_train.shape[1:]
batch_size = 64
learning_rate = 0.01
epochs = 100

model = build_unet(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)],
)

modelname=f'pebble_unet_nadir9obl4_kfold{kfi}_best_weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    modelname,  # specify the file to save the best weights
    monitor='val_loss',        # monitor validation loss
    save_best_only=True,        # save only the best weights
    mode='min',                 # mode can be 'min' or 'max' depending on the metric
    verbose=1
)

print('Model: '+modelname)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]      # include the ModelCheckpoint callback
)

def learning_progress(history,k):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    tree_iou = history.history[k[1]]
    val_tree_iou = history.history[k[3]]
    fg, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].semilogy(epochs, loss, "y", label="Training loss")
    ax[0].plot(epochs, val_loss, "r", label="Validation loss")
    ax[0].set_title("Training and validation loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(epochs, tree_iou, "y", label="Training IoU")
    ax[1].plot(epochs, val_tree_iou, "r", label="Validation IoU")
    ax[1].set_title("Training and validation IoU")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("IoU")
    ax[1].set_ylim((-0.01, 1.01))
    ax[1].grid()
    ax[1].legend()
    plt.suptitle(modelname[:-3])
    plt.savefig('ML/training_history/'+modelname[:-3]+'.png')
    plt.show()
learning_progress(history,list(history.history.keys()))

