FROM tensorflow/tensorflow:1.14.0-gpu-jupyter

COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

RUN apt-get update
RUN apt-get install -y git screen vim wget graphviz

RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --sys-prefix
#setting the notebbok password to "root"
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py

#exposing the ports used by jupyter and tensorboard
EXPOSE 7777
EXPOSE 6006