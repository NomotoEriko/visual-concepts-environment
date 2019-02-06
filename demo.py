# Auto converted script
# $ ipython nbconvert --to python demo.ipynb
# Source script is https://github.com/s-gupta/visual-concepts/blob/master/demo.ipynb

# coding: utf-8

# In[1]:


import _init_paths
import caffe, test_model, cap_eval_utils, sg_utils as utils
import cv2, numpy as np
# import matplotlib
# import matplotlib.pyplot as plt


# In[2]:


# Load the vocabulary
vocab_file = 'vocabs/vocab_train.pkl'
vocab = utils.load_variables(vocab_file)

# Set up Caffe
caffe.set_mode_gpu()
caffe.set_device(0)

# Load the model
mean = np.array([[[ 103.939, 116.779, 123.68]]]);
base_image_size = 565;    
prototxt_deploy = 'output/vgg/mil_finetune.prototxt.deploy'
model_file = 'output/vgg/snapshot_iter_240000.caffemodel'
model = test_model.load_model(prototxt_deploy, model_file, base_image_size, mean, vocab)


# In[3]:


# define functional words
functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
is_functional = np.array([x not in functional_words for x in vocab['words']])

# load the score precision mapping file
eval_file = 'output/vgg/snapshot_iter_240000.caffemodel_output/coco_valid1_eval.pkl'
pt = utils.load_variables(eval_file);

# Set threshold_metric_name and output_metric_name
threshold_metric_name = 'prec'; output_metric_name = 'prec';


# In[4]:


# Load the image
im = cv2.imread('./demo.jpg')

# Run the model
dt = {}
dt['sc'], dt['mil_prob'] = test_model.test_img(im, model['net'], model['base_image_size'], model['means'])

# Compute precision mapping - slow in per image mode, much faster in batch mode
prec = np.zeros(dt['mil_prob'].shape)
for jj in xrange(prec.shape[1]):
    prec[:,jj] = cap_eval_utils.compute_precision_score_mapping(        pt['details']['score'][:,jj]*1,         pt['details']['precision'][:,jj]*1,         dt['mil_prob'][:,jj]*1);
dt['prec'] = prec

# Output words
out = test_model.output_words_image(dt[threshold_metric_name][0,:], dt[output_metric_name][0,:],     min_words=3, threshold=0.5, vocab=vocab, is_functional=is_functional)


# In[5]:


# plt.rcParams['figure.figsize'] = (10, 10)
# plt.imshow(im[:,:,[2,1,0]])
# plt.gca().set_axis_off()
for (a,b,c) in out:
    print '{:s} [{:.2f}, {:.2f}]   '.format(a, np.round(b,2), np.round(c,2))


# In[5]:




