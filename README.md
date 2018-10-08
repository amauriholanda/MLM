
# Minimal Learning Machine - MLM

The MLM is a distance-based supervised learning method. The learning phase consists of a linear regression model between distances taken from fixed points (reference points) in the input and output spaces. Despite its simplicity, MLM has achieved competitive performance against reference machine learning methods on both classification and regression tasks.

This code consists of a simple Matlab implementation of the MLM and some of its variants.

 ## Using this API

 ```matlab
 load('./DATA/CLASSIFICATION/iris.mat');
 nRuns = 10;
 nFoldsCV = 10;

 accuracy = zeros(nRuns,1);
 for r = 1:nRuns
     %% Split data
     [Xl, Yl, Xt, Yt] = shuffle(X,y,0.8);
     
     data.x      = Xl;
     data.y      = MLMUtil.outputEncoding(Yl);
     
     testData.x  = Xt;
     testData.y  = MLMUtil.outputEncoding(Yt);    
     
     %% Training the model
     K = modelSelection(data, 0.1:0.1:1, nFoldsCV);
     model = train(data, K);
     
     %% Testing
     yhat = predict(model, testData, 'nn');
     accuracy(r,1) = MLMUtil.getAccuracy(testData.y, yhat);        
 end
 ```


 ## Citing MLM

 If you use this in a scientific publication, please cite:

 ```
 @article{de2015minimal,
   title={Minimal learning machine: a novel supervised distance-based approach for regression and classification},
   author={de Souza J{\'u}nior, Amauri Holanda and Corona, Francesco and Barreto, Guilherme A and Miche, Yoan and Lendasse, Amaury},
   journal={Neurocomputing},
   volume={164},
   pages={34--44},
   year={2015},
   publisher={Elsevier}
 }
 ```
