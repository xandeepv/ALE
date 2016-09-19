#include "main.h"

int main(int argc, char *argv[])
{
 
	ilInit();

	int from = 0, to = -1;
	if(argc == 3) from = atoi(argv[1]), to = atoi(argv[2]);
	if(argc == 4) from = atoi(argv[2]), to = atoi(argv[3]);

	LDataset *dataset = new LMsrcDataset();  // change the .bmp in dataset.cpp
//	LDataset *dataset = new LVOCDataset();
//	LDataset *dataset = new LCamVidDataset();
//	LDataset *dataset = new LCorelDataset();
//	LDataset *dataset = new LSowerbyDataset();
//	LDataset *dataset = new LLeuvenDataset();

	LCrf *crf = new LCrf(dataset);
	printf("Train File Numbers = %d \n", dataset->trainImageFiles.GetCount());
	for (int i = 0; i < dataset->trainImageFiles.GetCount(); i++){
		printf("Image %s \n", dataset->trainImageFiles[i]);
	}
	printf("Test File Numbers = %d \n", dataset->testImageFiles.GetCount());
	for (int i = 0; i < dataset->testImageFiles.GetCount(); i++){
		printf("Image %s \n", dataset->testImageFiles[i]);
	}
	printf("All File Numbers = %d \n", dataset->allImageFiles.GetCount());
	dataset->SetCRFStructure(crf);
	printf("Set CRF Structure Completed \n");
	crf->Segment(dataset->allImageFiles, from, to);
	printf("Segmentation of All Images done. \n");
	crf->TrainFeatures(dataset->trainImageFiles);
	printf("Training Features Completed. \n");
	crf->EvaluateFeatures(dataset->allImageFiles, from, to);
	printf("Evaluate Features Completed. \n");
	crf->TrainPotentials(dataset->trainImageFiles);
	printf("Training Potentials Completed. \n");
	crf->EvaluatePotentials(dataset->testImageFiles, from, to);
	printf("Evaluate Potentials Completed. \n");
	crf->Solve(dataset->testImageFiles, from, to);
	printf("Solve CRF Completed. \n");
	crf->Confusion(dataset->testImageFiles, "results.txt");
	printf("Confusion Matrix Completed. \n");

	delete crf;
	delete dataset;

	return(0);

}


