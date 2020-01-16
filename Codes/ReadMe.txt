For the producing results, first run "Pre_Processing.py", then it will processed images and save some data in your path. 

These two data:
	PreProcessing_Method1.npy
	PreProcessing_Method2.npy


The first file is results of preprocessing without scaling, and the second one is the results of preprocessing with scaling. 

After you find these files generated on your path, you can run any classification code.

In classification codes, you will find these:
	#processingMethod = 1
	processingMethod = 2
	#normalizingMethod = 'Quantizing'
	normalizingMethod = 'Binarizing'

simply, you can comment or comment out to use the corresponding preprocessing and normalization method.