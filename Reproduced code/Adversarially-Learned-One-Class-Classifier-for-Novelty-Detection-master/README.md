# Adversarially Learned One-Class Classifier for Novelty Detection (ALOCC-CVPR2018)

- Install pillow,scipy,skimage,imageio,numpy,matplotlib
```
pip install numpy scipy scikit-image imageio matplotlib pillow
```

### ALOCC train
- Download a UCSD dataset:
```
mkdir dataset
cd dataset
wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
tar -xzf UCSD_Anomaly_Dataset.tar.gz
```

- Config path of dataset and dataset name :
```
# for train on UCSD and patch_size 30*30
python train.py

<hr>


### ALOCC test
- You can run the following commands:
```
For test on UCSD and patch_size 30*30 and some specific dir like ['Test004'], etc. We prefer to open test.py file and edit every thing that you want

```
python test.py
```
<hr>

For changing the patch size, change the input_height from 30 

### Apply a pre-trained model (ALOCC)
- The pretrained model is saved in checkpoints directory

Saved model can be found in https://drive.google.com/open?id=1oprESeLKbbt2Fwse0K9vx1FxiftjmllA
```
At the time of testing, change the checkpoint manually in f_check_checkpoint function in models.py
<hr>

Anomalies of the directories are dumped in anomalies folder.
