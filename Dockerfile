FROM bvlc/caffe:gpu

RUN pip install numpy==1.10 opencv-python
RUN apt-get update && apt-get install -y python-tk
RUN git clone https://github.com/s-gupta/visual-concepts.git code
RUN git clone https://github.com/pdollar/coco.git code/coco && cd code/coco && git checkout 3736a0068b6e4634563b0c5847d7783dae9bd461 && cd ../../
RUN git clone https://github.com/s-gupta/caffe.git code/caffe && cd code/caffe && git checkout mil
RUN unlink /usr/include/numpy && ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy /usr/include/numpy 
COPY Makefile.config /workspace/code/caffe
RUN cd /workspace/code/caffe && make -j 16 && make pycaffe

# cafe-data: caffe pretrained model をダウンロードして workspace/caffedata に置く．実際にはこれはマウントで行った方がいいと思う．
# RUN wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/caffe-data.tgz && tar -xf caffe-data.tgz
# vgg を workspace/code/output/vgg に置く
# RUN wget ftp://ftp.cs.berkeley.edu/pub/projects/vision/im2cap-cvpr15b/trained-coco.v2.tgz && tar -xf trained-coco.v2.tgz 
# workspace/data に coco をリンクする

CMD ["/bin/bash"]
 
