
import DL
import DLTrainer
import DLUtils
import Display
import numpy as np

batch_size        = 20
num_epochs        = 200
samples_per_class = 100
num_classes       = 3
hidden_units      = 100
data,target       = DLUtils.genSpiralData(samples_per_class,num_classes)
model             = DLTrainer.Model()
model.add(DL.Linear(2,hidden_units))
model.add(DL.ReLU())
model.add(DL.Linear(hidden_units,num_classes))
optim   = DL.SGD(model.params,learning_rate=1.0,weight_decay=0.001,momentum=.9)
loss_fn = DL.SoftmaxWithLoss()
model.fit(data,target,batch_size,num_epochs,optim,loss_fn)
predicted_labels = np.argmax(model.predict(data),axis=1)
accuracy         = np.sum(predicted_labels==target)/len(target)
print("Model Accuracy = {}".format(accuracy))
Display.plot2DDataWithDecisionBoundary(data,target,model)