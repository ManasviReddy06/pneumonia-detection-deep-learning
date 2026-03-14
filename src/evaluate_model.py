import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("models/pneumonia_model.h5")

datagen = ImageDataGenerator(rescale=1./255)

test = datagen.flow_from_directory(
    "dataset/test",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

loss, accuracy = model.evaluate(test)
print("Test Accuracy:", accuracy)

plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.savefig("results/accuracy_plot.png")

pred = model.predict(test)
predicted_classes = (pred > 0.5).astype("int32")

cm = confusion_matrix(test.classes, predicted_classes)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.savefig("results/confusion_matrix.png")