# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.optimizers import SGD
from matplotlib import pyplot as plt

import os

from create_model import create_base_network, in_dim, tripletize, std_triplet_loss,create_trivial
from generators import triplet_generator
import testing as T

import config as C

last = C.last

def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)

def lock_layers(model,i):
    n = i*5
    print(len(model.layers))
    if n >=len(model.layers):
        n=len(model.layers)

    for layer in model.layers[:-n]:
        layer.trainable = False
    for layer in model.layers[-n:]:
        layer.trainable = True
    return model

def save_loss(history,stage):
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("loss/"+str(stage)+'_'+'loss_plot'+'.png')
    #plt.show()
    plt.close()

# Use log to file
logger = CSVLogger(C.logfile, append=True, separator='\t')

def train_step(i):
    
    checkpoint = ModelCheckpoint("models/trained_stage_"+str(i)+"_best.model", monitor='val_loss', save_best_only=True)
    history = model.fit_generator(
            triplet_generator(C.batch_size, None, C.train_dir), steps_per_epoch=1500, epochs=C.iterations,
            callbacks=[logger,checkpoint],
            validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=100)
    save_loss(history, i)

if last==0:
    log('Creating base network from scratch.')
    if not os.path.exists('models'):
        os.makedirs('models')
    base_model = create_base_network(in_dim)
else:
    log('Loading model:'+save_name(last))
    base_model = load_model(save_name(last))

model = tripletize(base_model)
model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
              loss=std_triplet_loss())

def avg(x):
    return sum(x)/len(x)

vs = T.get_vectors(base_model, C.val_dir)
cents = {}
for v in vs:
    cents[v] = T.centroid(vs[v])

for i in range(last+1, last+11):
    log('Starting iteration '+str(i)+'/'+str(last+10)+' lr='+str(C.learn_rate))
    train_step(i)
    C.learn_rate = C.learn_rate * C.lr_decay
    base_model.save(save_name(i))

    vs = T.get_vectors(base_model, C.val_dir)

    X_data,y_data = T.vectors_to_points(vs,64)
    print("doing PCA")
    explained_variance = T.do_pca(X_data,y_data,"pca/",i,features=3)
    print(explained_variance)
    c = T.count_nearest_centroid(vs)
    log('Summarizing '+str(i))
    with open('log/summarize.'+str(i)+'.log', 'w') as sumfile:
        T.summarize(vs, outfile=sumfile)
    with open('log/clusters.'+str(i)+'.log', 'w') as cfile:
        T.confusion_counts(c, outfile=cfile)
    c_tmp = {}
    r_tmp = {}
    for v in vs:
        c_tmp[v] = T.centroid(vs[v])
        r_tmp[v] = T.radius(c_tmp[v], vs[v])
    c_rad = [round(100*r_tmp[v])/100 for v in vs]
    c_mv = [round(100*T.dist(c_tmp[v],cents[v]))/100 for v in vs]
    log('Centroid radius: '+str(c_rad))
    log('Centroid moved: '+str(c_mv))
    log('PCA explained variance: '+str(explained_variance))
    cents = c_tmp

    with open(C.logfile, 'a') as f:
        T.accuracy_counts(c, outfile=f)
    # todo: avg cluster radius, avg cluster distances
    log('Avg centr rad: %.2f move: %.2f' % (avg(c_rad), avg(c_mv)))
