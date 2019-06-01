import config as C

from PIL import Image
import numpy as np
import sys
import pandas as pd
from sklearn.decomposition import PCA
from generators import paste
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from create_model import create_base_network, in_dim, tripletize, std_triplet_loss,create_trivial
from generators import triplet_generator

import config as C

FILENAME="mlp"

def class_file(model, fname):
    img = np.array(Image.open(fname))/256
    return model.predict(np.expand_dims(paste(img), axis=0))

def dist(x,y):
    return np.linalg.norm(x-y)

# Calculate histogram of all distances from v in v1 to w in v2
def dist_hist(v1,v2):
    return [dist(v,w) for v in v1 for w in v2]

def centroid(vs):
    x0 = np.zeros_like(vs[0])
    for x in vs:
        x0 = np.add(x0,x)
    return x0/len(vs)

import os

# return the set of classes with vectors from model
def get_vectors(model, tdir=C.test_dir):
    classes = os.listdir(tdir)
    res = {}
    for c in classes:
           images = os.listdir(os.path.join(tdir,c))
           res[c] = [class_file(model, os.path.join(tdir,c,f)) for f in images]
    return res

# radius of a cluster, i.e. average or max distance from centroid
def radius(c, vs, avg=True):
    ds = [dist(c,v) for v in vs]
    if avg:
        return sum(ds)/len(ds)
    else:
        return max(ds)

# for backwards compatibility
def centroid_distances(vectors, outfile=sys.stdout):
    # load a bunch of images, calculate outputs
    res = {}
    for c in vectors:
           ds = dist_hist(vectors[c],vectors[c])
           res[c] = centroid(vectors[c])
           print(c[:25].ljust(26),' average radius: %.4f worst case diameter: %.4f' % (sum(ds)/len(ds), max(ds)), file=outfile)
    print('\nCentroid distances',file=outfile)
    for c in res:
        print(c[25].ljust(26),end=': ', file=outfile)
        for n in res:
            print('  %.3f' % (dist(res[c],res[n])), end='', file=outfile)
        print(file=outfile)

# average radius and centroid distances for all test classes
def summarize(vectors, outfile=sys.stdout):
    cents = {}
    rads  = {}
    for c in vectors:
        cents[c] = centroid(vectors[c])
        rads[c] = radius(cents[c], vectors[c])
    for c in vectors:
        print(c[:25].ljust(26),' r=%.3f ' % rads[c], end='', file=outfile)
        for n in vectors:
            print('  %.3f' % dist(cents[c],cents[n]), end='', file=outfile)
        print(file=outfile)

# assign each input to nearest centroid and tally
def count_nearest_centroid(vectors):
    cents = {}
    for c in vectors:
        cents[c] = centroid(vectors[c])
    counts = {}
    for c in vectors:
        counts[c] = {}
        for x in vectors:
            counts[c][x] = 0
        for v in vectors[c]:
            # find closest centroid, and bump its count
            nearest = None
            mindist = 9999999
            for ct in cents:
                d = dist(v,cents[ct])
                if d < mindist:
                    nearest = ct
                    mindist = d
            counts[c][nearest] = counts[c][nearest] + 1 
    return counts

def accuracy_counts(cts, outfile=sys.stdout):
    correct = 0
    total   = 0
    for v in cts:
        correct = correct + cts[v][v]
        for w in cts:
            total = total + cts[v][w]
    print('Accuracy: %.3f' % (correct/total), file=outfile)
        
def confusion_counts(cts, outfile=sys.stdout):
    for v in cts:
        print(v[:25].ljust(26),end='', file=outfile)
        for w in cts:
            print(" %4d" % cts[v][w], end='', file=outfile)
        print(file=outfile)

# find the class and distance of the k nearest elements to tvec in refset
def find_nearest(refset, tvec, k=1):
    mindist = []
    for c in refset:
        for p in refset[c]:
            d = dist(p,tvec)
            if len(mindist)<k or d < max([x for x,v in mindist]):
                mindist.append((d,c))
                mindist.sort()
                mindist = mindist[:k]
    return mindist


# classify data in tdir using kNN with rdir as the reference
def knn_test(model, rdir, tdir, k=5):
    rvecs = get_vectors(model, rdir)
    tvecs = get_vectors(model, tdir)
    cmx = {}
    for x in tvecs:
        cmx[x] = {}
        for y in rvecs:
            cmx[x][y] = 0
    for c in tvecs:
        for v in tvecs[c]:
            xs = find_nearest(rvecs, v, k)
            rs = [v for c,v in xs]
            r = max(set(rs), key=rs.count)
            # if r != c: print(xs)
            cmx[c][r] = cmx[c][r]+1
    return cmx

# todo: PCA plots

# Default test to use
def run_test(model, tdir=C.test_dir, outfile=sys.stdout):
    vecs = get_vectors(model,tdir)

#    print('Centroid dists:')
#    centroid_distances(vecs)

#    print('Summarize:')
#    summarize(vecs)

    c = count_nearest_centroid(vecs)
    accuracy_counts(c, outfile=outfile)

#    print('Confusion matrix:')
#    confusion_counts(c)

def do_pca(X,y,path,i,features=3):
    pca = PCA(n_components=features)
    new = pca.fit_transform(X)

    label_color_dict = {label:idx for idx,label in enumerate(np.unique(y))}
    # Color vector creation
    cvec = [label_color_dict[label] for label in y]
    if(features==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(new[:,0],new[:,1],new[:,2],c=cvec,cmap='nipy_spectral')
        ax.set_xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
        ax.set_ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
        ax.set_zlabel('PC 3 (%.2f%%)' % (pca.explained_variance_ratio_[2]*100))
        ax.set_title(path+"PCA"+str(i))
        plt.savefig("pca"+str(i)+".png")
        plt.close()

        return pca.explained_variance_ratio_

def get_vectors_with_image(model, tdir=C.test_dir):
    classes = os.listdir(tdir)
    X = []
    y = []
    img = []

    for c in classes:
           images = os.listdir(os.path.join(tdir,c))
           for f in images:
            X.append(class_file(model, os.path.join(tdir,c,f)))
            y.append(c)
            img.append(os.path.join(tdir,c,f))
    X = np.array(X).reshape((len(X),64))
    return X,y,img

def vectors_to_points(vectors,dim):
    X = []
    y = []

    for cl in vectors.keys():
        for x in vectors[cl]:
            X.append(x)
            y.append(cl)

    X = np.array(X).reshape((len(X),dim))
    return X,y

def create_difference_vectors(X,lab,img):
    diff = []
    magnitude = []
    xs =[]
    ys = []
    X_img = []
    Y_img = []
    X_withlabel=list(zip(X,lab,img))

    for i,(x,x_label,ximg) in enumerate(X_withlabel):
        for y,y_label,yimg in X_withlabel[i+1:]:
            if x[0]>y[0]:
                sub = np.subtract(x,y)
                xs.append(x_label)
                ys.append(y_label)
                X_img.append(ximg)
                Y_img.append(yimg)
            else:
                sub = np.subtract(y,x)
                xs.append(y_label)
                ys.append(x_label)
                X_img.append(yimg)
                Y_img.append(ximg)

            diff.append(sub)
            m = np.linalg.norm(sub)
            magnitude.append(m)
            #print(m)

    dtype = [('diff', np.float64,(64,)),('magnitude',np.float64),('x_label',np.unicode_,16),('y_label',np.unicode_,16),('x_img',np.unicode_,128),('y_img',np.unicode_,128)]
    array = np.asarray(list(zip(diff,magnitude,xs,ys,X_img,Y_img)), dtype=dtype)
    return  np.sort(array,order='magnitude')

def get_equal_vectors(vecs,epsilon):
    elems=[]
    for i,x in enumerate(vecs[:-2]):
        eq = dist(x['diff'],vecs[i+1]['diff'])
        if eq<epsilon:
            elems.append((x,vecs[i+1]))
            plot_similars(x,vecs[i+1],eq)
            print("({}, {}) -> {} -> ({}, {})".format(x['x_label'],x['y_label'],eq,vecs[i+1]['x_label'],vecs[i+1]['y_label']))
    return len(elems)

def plot_similars(x,y,diff):
    fig = plt.figure(figsize=(8,8))
    fig.suptitle("Plot of pictures with similar relation",fontsize=16)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    #plt.subplots_adjust(bottom=0.1, right=0.8, top=0.8)
    ax1.title.set_text('X1: '+ x['x_label']+x['x_img'].split('/')[-1])
    ax2.title.set_text('Y1: '+ x['y_label']+x['y_img'].split('/')[-1])
    ax3.title.set_text('X2: '+ y['x_label']+y['x_img'].split('/')[-1])
    ax4.title.set_text('Y2: '+ y['y_label']+y['y_img'].split('/')[-1])
    ax1.imshow(np.array(Image.open(x['x_img']))/256)
    ax2.imshow(np.array(Image.open(x['y_img']))/256)
    ax3.imshow(np.array(Image.open(y['x_img']))/256)
    ax4.imshow(np.array(Image.open(y['y_img']))/256)
    plt.savefig("diff/"+str(diff)+".png")
    plt.close()

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



def PCA_over_models(data,path,start=0,end=27,features=3):
	model = load_model("models/epoch_"+end+".model")
	pca = PCA(n_components=features)
	X,y,img = get_vectors_with_image(model, data)
    pca.fit(X)

    label_color_dict = {label:idx for idx,label in enumerate(np.unique(y))}
    # Color vector creation
    cvec = [label_color_dict[label] for label in y]


    for i in range(start,end+1):
    	model = load_model("models/epoch_"+i+".model")
    	X,y,img = get_vectors_with_image(model, data)
    	new = pca.transform(X)

    	if(features==3):
	        fig = plt.figure()
	        ax = fig.add_subplot(111, projection='3d')
	        ax.scatter(new[:,0],new[:,1],new[:,2],c=cvec,cmap='nipy_spectral')
	        ax.set_xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
	        ax.set_ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
	        ax.set_zlabel('PC 3 (%.2f%%)' % (pca.explained_variance_ratio_[2]*100))
	        ax.set_title("PCA comparable "+str(i))
	        plt.savefig(path+"pca"+str(i)+".png")
	        plt.close()

def train_mlp_classifier(model,X,y,valid_set):

    result = model.fit(X,y,
                        batch_size=64,
                        epochs=20,
                        verbose=2,
                        validation_data=valid_set)

    save_loss(result, "mlp")

    return model


def save_model(model, name):
    with open(name+'.json','w') as f:
        f.write(model.to_json())
    model.save_weights(name+'.h5')

def load_model(name):
    with open(name+'.json','r') as f:
        model = model_from_json(f.read())
    model.load_weights(name+'.h5')
    return model

def confusion_matrix(model,X,y):
        conf_matrix = np.zeros((10,10),dtype= int)
        predicts = model.predict(X)
        
        errorEx= []
        n = 0
        for xi,yi,img in zip(predicts,y,X):
                pred = int(np.argmax(xi))
                corr = int(np.argmax(yi))
                if pred != corr and n < 6:
                    errorEx.append(img)
                    n+=1
                conf_matrix[pred][corr]+=1
        save_error_examples(errorEx)
        return conf_matrix

    
def save_error_examples(errors):
    fig, axs = plt.subplots(nrows=2,ncols=3)
    for img, ax in zip(errors,axs.flat):
        img = img.reshape((28,28))
        ax.imshow(img,cmap='gray')
    plt.savefig(FILENAME+'_'+'error_example'+'.png')
    plt.close()

def array_to_csv(array, model_name, array_name):
    pd.DataFrame(array).to_csv(model_name+'_'+array_name+'.csv',header=None, index=None)

if __name__ == '__main__':
    base_model = load_model("models/epoch_27.model")
    model = create_trivial()
    model.compile(optimizer=SGD(momentum=0.9))
    #          loss=std_triplet_loss())

    #vectors = get_vectors(base_model, C.val_dir)
    PCA_over_models(C.test_dir,"pca/comparable/")
    #X,y,img = vectors_to_points(vectors,64)

    X_train,y_train,img = get_vectors_with_image(base_model, C.train_dir)
    X_val,y_val,img = get_vectors_with_image(base_model, C.train_dir)
    model = train_mlp_classifier(model,X_train,y_train,(X_val,y_val))


    X_test,y_test,img = get_vectors_with_image(base_model, C.train_dir)
    conf_matrix = confusion_matrix(model,X_test,y_test)

    array_to_csv(conf_matrix, FILENAME, 'conf_matrix')
    #diff_vec = create_difference_vectors(X,y,img)
    #print(len(diff_vec))
    #print(get_equal_vectors(diff_vec,0.30))
    #X = diff_vec['diff']
    #y = diff_vec['magnitude']
    #print(do_pca(X,y,"pca/test/",0,features=3))