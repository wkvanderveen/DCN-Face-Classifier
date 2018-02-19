from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import argparse
import sys
import os
import time

# The next two lines suppress warnings and clear the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.system('cls' if os.name == 'nt' else 'clear')

import tensorflow as tf
import numpy as np

class Face_classifier:
  def __init__(self):
    # Define model parameters
    self.make_graph = False
    self.print_bar = True
    self.print_time = True
    self.color = True # keep True if using frontalization
    self.front = True # remember to set data sizes accordingly
    self.learning_rate = 0.0001
    self.hidden_size1 = 300
    self.hidden_size2 = 150
    self.hidden_size3 = 75
    #self.hidden_size4 = 50  
    self.number_of_features1 = 32
    self.number_of_features2 = 16
    self.train_percentage = 80.00 # in range [20.00 -- 99.00]
    self.epochs = 100000 
    self.batch_size = 10
    self.image_width = 150
    self.image_height = 150
    self.filter_size1 = 11
    self.filter_size2 = 9
    self.pool_size = 3

    self.wild_size = 4500 # in range [50 -- 124290]
    self.TA_size = 4500 # per TA, in range [50 -- 11120]

    self.start_time = time.time()

    self.image_depth = 1 # is default, will change if self.color == True
    if(self.color):
      self.image_depth = 3

    # Resize progress bar to terminal size
    _, self.columns = os.popen('stty size', 'r').read().split()
    self.columns = int(self.columns) - 30
    if self.columns < 5:
        self.columns = 5
    
    # Weights for each layer
    self.w_conv1 = self.weight_variable([self.filter_size1, self.filter_size1, self.image_depth, self.number_of_features1], "w_conv1") 
    self.w_conv2 = self.weight_variable([self.filter_size2, self.filter_size2, self.number_of_features1, self.number_of_features2], "w_conv2")

    self.w_hl1 = self.weight_variable([int(self.number_of_features2*self.image_width*self.image_height/(self.pool_size * self.pool_size)), self.hidden_size1], "w_hl1")
    self.w_hl2 = self.weight_variable([self.hidden_size1, self.hidden_size2], "w_hl2")
    self.w_hl3 = self.weight_variable([self.hidden_size2, self.hidden_size3], "w_hl3")
    #self.w_hl4 = self.weight_variable([self.hidden_size3, self.hidden_size4], "w_hl4")
    self.w_y = self.weight_variable([self.hidden_size3, 5], "w_y")
    
    # Biases for each layer
    self.b_conv1 = self.bias_variable([self.number_of_features1], "b_conv")
    self.b_conv2 = self.bias_variable([self.number_of_features2], "b_conv")
    self.b_hl1 = self.bias_variable([self.hidden_size1], "b_hl1")
    self.b_hl2 = self.bias_variable([self.hidden_size2], "b_hl2")
    self.b_hl3 = self.bias_variable([self.hidden_size3], "b_hl3")
    #self.b_hl4 = self.bias_variable([self.hidden_size4], "b_hl4")
    
    self.b_y = self.bias_variable([5], "b_y")

    # Input layer
    self.x = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_depth], name = "input")

    # Convolution layer 1
    self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)
    

    # Pooling layer
    self.h_pool = self.max_pool(self.h_conv1)

    # Convolution layer 2
    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool, self.w_conv2) + self.b_conv2)


    # Hidden layer 1
    self.hl1 = tf.nn.relu(tf.matmul(tf.reshape(self.h_conv2,[-1, int(self.number_of_features2*self.image_width*self.image_height/(self.pool_size * self.pool_size))]), self.w_hl1) + self.b_hl1)

    # Hidden layer 2
    self.hl2 = tf.nn.relu(tf.matmul(self.hl1, self.w_hl2) + self.b_hl2)

    # Hidden layer 3
    self.hl3 = tf.nn.relu(tf.matmul(self.hl2, self.w_hl3) + self.b_hl3)
    
    # Hidden layer 4
    #self.hl4 = tf.nn.relu(tf.matmul(self.hl3, self.w_hl4) + self.b_hl4)

    # Output layer
    self.y = tf.matmul(self.hl3, self.w_y) + self.b_y
    
    # Labels
    self.y_ = tf.placeholder(tf.float32, [None, 5], name = "output")

    # Define loss and optimizer
    self.cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = self.y_, logits = self.y))

    # Define the training step
    self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)

    self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    # Define session and initialize global variables (i.e., weights and biases)
    self.sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Create a saver object which will save all the variables
    self.saver = tf.train.Saver()

    # Write out a visualization of the session model, which can be viewed using TensorBoard
    if (self.make_graph):
      self.graph_destination = './my_graph'
      self.writer = tf.summary.FileWriter(graph_destination, sess.graph)


  def prepareImages(self):
    # Read in all images and labels of the ORL face database
    self.imgListTrain, self.imgListTest, self.labelListTrain, self.labelListTest =         self.read_images(self.train_percentage, self.front, self.color, self.wild_size, self.TA_size)

  def print_progress(self, iteration, total):
    # Print progress bar
    
    percents = "{0:.2f}".format(100 * ((iteration+1) / float(total)))
    filled_length = int(round(self.columns * (iteration+1) / float(total)))
    bar = u'\u2588' * filled_length + '-' * (self.columns - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % ('Progress', bar, percents, '%', 'Complete')),

    if iteration == total:
      sys.stdout.write('\n')
    sys.stdout.flush()
      
  def weight_variable(self, shape,n):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name = n)

  def bias_variable(self, shape,n):
    initial = tf.constant(0.1, shape=shape,)
    return tf.Variable(initial, name = n)

  def conv2d(self, x, W):
    return tf.nn.conv2d(
      x, 
      W, 
      strides = [1, 1, 1, 1], 
      padding = 'SAME'
      )

  def max_pool(self, x):
    return tf.nn.max_pool(
      x, 
      ksize   = [1, self.pool_size, self.pool_size, 1], 
      strides = [1, 3, 3, 1], 
      padding = 'SAME'
      )

  def read_images(self, train_percentage, front, color = False, wild_size = 13020, TA_size = 500):
    # Read images from the face database, and store them in training and testing image lists.
    # Also store image labels, indicating the identities, in training and testing label lists.
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # The next line defines the "home" folder, from which the database is accessed.
    # This way, the code can be run from a Windows OS as well.
    home = os.path.expanduser("~")

    # Boolean switch to access correct data folder (color or gray faces, respectively).
    if front:
      col = 'front_preprocessed/'    
    elif color:
      col = 'col/'
    else:
      col = 'gray/'

    print("\nReading face database ('TAs' | {0} ".format(TA_size) + u'\u00D7' +" 4 = {0} faces)...".format(TA_size*4))

    if self.print_bar:
        # The next line initializes the progress bar for loading the TA faces.
        progress_bar_counter = 0
    
    # There are five labels of images -- one for each of four TAs, and one for a large collection of other faces.
    # The next loop iterates through the TAs and adds them to the train and test list.
    for TA in os.listdir(home + '/dataset/' + col + 'TAs'):
      for filename in os.listdir(home + '/dataset/' + col + 'TAs/' + TA):

        # Update the progress bar for loading the TA faces (length: 2000 TA faces total)

        # Read a single image from the subdirectory, and store it as a numpy array.
        image_number = int(filename.split('.')[0])

        if (image_number <= TA_size):

          if self.print_bar:
              self.print_progress(progress_bar_counter, TA_size*4)
              progress_bar_counter += 1
          
          # Assign a random chance that the face be appended to the test set.
          # If "train_prop*100" is larger than a random integer in the range [1,10000],
          # then the face is added to the training set.
          rand = random.randrange(1, 10000)
          image = cv2.imread(home + '/dataset/' + col + 'TAs/' + TA + '/' + filename)
          #cv2.imshow("",image)
          #cv2.waitKey(0)
          image = cv2.resize(
            image,
            dsize = (150, 150),
            interpolation = cv2.INTER_CUBIC
            )
         
          image = image.reshape(150, 150, self.image_depth)
          np_image_data = np.asarray(image)
          np_image_data = np_image_data.astype(float) / 255

          # Split the database up into train and test sets, and label accordingly.
          if(train_percentage*100 > rand):
            train_images.append(np_image_data)
            if(TA == 'Amir'):
              train_labels.append( (1.,0.,0.,0.,0.) )
            elif(TA == 'Marc'):
              train_labels.append( (0.,1.,0.,0.,0.) )
            elif(TA == 'Rik'):
              train_labels.append( (0.,0.,1.,0.,0.) )
            elif(TA == 'Totti'):
              train_labels.append( (0.,0.,0.,1.,0.) )

          else:
            test_images.append(np_image_data)
            if(TA == 'Amir'):
              test_labels.append( (1.,0.,0.,0.,0.) )
            elif(TA == 'Marc'):
              test_labels.append( (0.,1.,0.,0.,0.) )
            elif(TA == 'Rik'):
              test_labels.append( (0.,0.,1.,0.,0.) )
            elif(TA == 'Totti'):
              test_labels.append( (0.,0.,0.,1.,0.) )

    
    print("\n\nReading face database ('Wild' | {0} faces)...".format(wild_size))

    progress_bar_counter = 0

    for filename in os.listdir(home + '/dataset/' + col + 'wild'):

      progress_bar_counter += 1
      if (progress_bar_counter >= wild_size):
        break

      if self.print_bar:
          self.print_progress(progress_bar_counter, wild_size)
          

      # Read a single image from the subdirectory, and store it as a numpy array.
      image_number = int(filename.split('.')[0])

      # Assign a random chance that the face be appended to the test set.
      # If "train_prop*100" is larger than a random integer in the range [1,10000],
      # then the face is added to the training set.
      rand = random.randrange(1, 10000)

      image = cv2.imread(home + '/dataset/' + col + 'wild/' + filename)
      image = cv2.resize(
        image,
        dsize = (150, 150),
        interpolation = cv2.INTER_CUBIC
        )
      image = image.reshape(150, 150, self.image_depth)
      np_image_data = np.asarray(image)
      np_image_data = np_image_data.astype(float) / 255

      # Split the database up into train and test sets, and label accordingly.
      if(train_percentage*100 > rand):
        train_images.append(np_image_data)
        train_labels.append( (0.,0.,0.,0.,1.) )

      else:
        test_images.append(np_image_data)
        test_labels.append( (0.,0.,0.,0.,1.) )

    return train_images, test_images, train_labels, test_labels

  def make_batches(self, batch_size, train_images, train_labels):
    # Return a randomly selected batch of specified size from image and label lists.
    random_index = random.randrange(0, len(train_images)-batch_size)
    return train_images[random_index: random_index+batch_size], train_labels[random_index: random_index+batch_size]

  def classify(self, feed, sess):
    y = self.y.eval(session = sess, feed_dict = {self.x:feed})
    output = y[0]
    return np.argmax(output)


  def train(self):
    print("\n\nTraining model...")
    
    for epoch in range(self.epochs):
      if self.print_bar:
        self.print_progress(epoch, self.epochs)

      # Make a small batch of images and corresponding labels, and feed it to the training step
      batch_x, batch_y = self.make_batches(self.batch_size, self.imgListTrain, self.labelListTrain)
      self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y_: batch_y})
        

      if epoch%20 == 0:
         batch_x_test, batch_y_test = self.make_batches(self.batch_size, self.imgListTest, self.labelListTest)
         plus = self.sess.run(self.accuracy, feed_dict={self.x: batch_x_test, self.y_: batch_y_test})
         print(plus)
      




    self.saver.save(self.sess, 'face_classifier_test')
    acc = 0
    
    for i in range(int(len(self.imgListTest)/self.batch_size)):
        batch_x_test, batch_y_test = self.make_batches(self.batch_size, self.imgListTest, self.labelListTest)
        plus = self.sess.run(self.accuracy, feed_dict={self.x: batch_x_test, self.y_: batch_y_test})
        acc += plus
        if not(i%10):
            print("\nBatch accuracy ({0}/{1}) = {2:.2f}\n".format(i, int(len(self.imgListTest)/self.batch_size), plus))
      
    
    acc = acc / (len(self.imgListTest)/self.batch_size)
        

  



    # Print the final accuracy and corresponding parameter settings
    if self.print_bar:
        print(
          "\n\nAccuracy = {0:.5f}\n\t\
          Hidden Layer Sizes \t= {1}\n\t\
          Training Percentage \t= {2:.2f}% \n\t\
          Number of Features 1 \t= {3}\n\t\
          Number of Features 2 \t= {12}\n\t\
          Color \t\t= {4}\n\t\
          Learning Rate \t= {5}\n\t\
          Filter Size 1 \t= {6}\n\t\
          Filter Size 2 \t= {11}\n\t\
          Batch Size \t\t= {7}\n\t\
          'TA' Size \t\t= {8}\n\t\
          'Wild' Size \t\t= {9}\n\t\
          Number of Epochs \t= {10}\n\t\
          Frontalized \t\t= {13}\n"
          .format(
            acc,
            [self.hidden_size1, self.hidden_size2, self.hidden_size3],
            self.train_percentage,
            self.number_of_features1,
            self.color,
            self.learning_rate,
            self.filter_size1,
            self.batch_size,
            self.TA_size,
            self.wild_size,
            self.epochs,
            self.filter_size2,
            self.number_of_features2,
            self.front
          )
        )
    else:
        print("\n\nAccuracy = {0:.5f}\n".format(acc))

    if (self.make_graph):
        print("\n\nA graph of the network has been placed in '{0}'.\n".format(graph_destination))


    if (self.print_time):
        end_time = time.time()
        print(time.strftime("Time elapsed: \n%H hours, %M minutes, %S seconds\n", time.gmtime(int(end_time - self.start_time))))

if __name__ == '__main__':
  network = Face_classifier()
  network.prepareImages()
  network.train()
