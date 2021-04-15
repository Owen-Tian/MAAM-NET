# MAAM-NET
 
This is the source code of MAAM-Net.

This repo includes:

    CKPT - checkpoint among three datasets;
    *.py - codes.
    
The dataset should be like:

	~/** dataset/training(testing)/01_111/001.jpg
    ~/** dataset/training_flow(testing_flow)/01_111/001.flo

To train the model, run with:

    python train.py
    
To test the model, run with:

    python test.py
    
Look the detail hypermeters in the train(test).py...  
