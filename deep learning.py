from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score

classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape = (51, 51, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))


classificador.add(Conv2D(32, (3,3), input_shape = (51, 51, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
#classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))


classificador.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale=None)

gerador_teste = ImageDataGenerator(rescale=None)

#passa o diretorio das imagens e define as classe automaticamente
#target size precisa ser o msm definido na rede neural
base_treinamento = gerador_treinamento.flow_from_directory('data/train', target_size = (51,51), batch_size = 32, class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('data/test', target_size = (51,51), batch_size = 32, class_mode = 'binary')

classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 /32, epochs = 100, validation_data = base_teste, validation_steps = 30)



gerador_valida = ImageDataGenerator(rescale=None)
base_valida = gerador_valida.flow_from_directory('data/validation', target_size = (51,51), batch_size = 32, class_mode = 'binary')


resultado = classificador.predict(base_valida)

resultado = (resultado > 0.5)



