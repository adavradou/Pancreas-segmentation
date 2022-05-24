# Pancreas segmentation
A U-Net network for pancreas segmentation in CT-Scans.


The repository was based on repo: https://github.com/mirzaevinom/promise12_segmentation 


To train the model run:
python train_cv.py train --epochs 1000 --regenerate True | tee train.txt 

To test the model run: 
python test.py test --weights model_3.h5 --volume_name | tee test.txt 

