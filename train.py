import tensorflow as tf
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

def instantiate_classifier():
    """Generate & compile a simple CNN classifier using tf.keras sequential API"""
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    
    METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
    ]

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=METRICS
                 )

    return model

def _parse_args():
    """Parse user-defined command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--model_version", type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    
    # Parse command line args
    args, unknown = _parse_args()

    # Load processed data (numpy arrays)
    cls_weight = np.load(args.train + "/cls_weight.npy")
    x_train = np.load(args.train + "/x_train.npy")
    y_train = np.load(args.train + "/y_train.npy")
    x_test = np.load(args.test + "/x_test.npy")
    y_test = np.load(args.test + "/y_test.npy")

    # Instantiate classifier
    classifier = instantiate_classifier()
    
    # Fit classifier
    classifier.fit(x_train, y_train,
                   batch_size=args.batch_size,
                   epochs=args.epochs,
                   class_weight=cls_weight,
                   verbose=1
                  )
    
    # Print AUC on test-set
    print("AUC: " + str(classifier.evaluate(x_test, y_test)[-1]))
    
    classifier.save(os.path.join(args.model_dir, args.model_version), 'my_model.h5')
    
    
    