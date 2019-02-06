# Build
```
nvidia-docker build -t nomotoeriko/visual-concepts:latest .
```

# Download pre-trained models
```
$ nvidia-docker run -it --name viscon --rm -v `pwd`/output/vgg:/workspace/output/vgg  nomotoeriko/visual-concepts:latest
# Get the caffe imagenet models 
(viscon) $ wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/caffe-data.tgz && tar -xf caffe-data.tgz
# Get the pretrained models. The precomputed results in the previous tarball
# were not complete. v2 tar ball fixes this.
(viscon) $ wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/trained-coco.v2.tgz && tar -xf trained-coco.v2.tgz 
```

# Run
```
(viscon) $ [Cntl+pq]
$ docker cp demo.py viscon:/workspace/code/
(viscon) $ cd code
(viscon) $ python demo.py
```

