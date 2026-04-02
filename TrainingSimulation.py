import matplotlib.pyplot as plt

from utilis import *
from sklearn.model_selection import train_test_split

##########STEP 1########
path = 'myData'
data = importDataInfo(path)

######STEP 2:VISUALIZATION & DISTRIBUTION OF DATA #######
data = balanceData(data, display=False)

#####STEP 3 LOADING THE DATA#######
imagesPath, steerings = loadData(path, data)
#print(imagesPath[0], steering[0])

########STEP 4: SPLITTING THE DATA INTO TRAINING AND VALIDATION#######
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5) #80% will be for training and 20% for validation
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

#####STEP 5: AUGMENT DATA TO ADD MORE VARIETY AND VARIANTS #############


###STEP 6: PREPROCESSION####

####STEP 7

####STEP 8: Create Model
model = creatModel()
model.summary()

####STEP 9: TRAIN THE MODEL#####
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
                    validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200) #train 10 times 30,000 images and generate a batch of 100 images

#######STEP 10: SAVE TRAINED MODEL ######
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()