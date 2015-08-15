==============================
GADANN:  GPU Accelerated Deep Artificial Neural Network Library
==============================

GADANN is a Python library for creating high-performance neural network architectures on the GPU.  The primary intent is to provide a platform for implementing and experimenting with algorithms, while providing a transition path to creating a production system.

The overall goals of the library are:

* **Clarity and Simplicity**: Understanding and correctly implementing the latest algorithms is enough of a challenge, GADANN aims to minimize its own learning curve.   It does this by keeping algorithm implementations as explicit as possible, rather than depending on techniques such as automatic symbolic differentiation.

* **Fast**: By focusing on performance, GADANN increases research productivity, and reduces the optimization effort to transition to a production system.

* **Minimal, high-quality dependencies**:  Keeping the dependencies minimal, new users can get running faster, and a system can be transitioned to a production system easier.

------------------------------
Usage
------------------------------
.. code:: python

	model = gadann.NeuralNetworkModel(
	    input_shape = (1,28,28),
	    layers = [
	        {'layer':gadann.LinearLayer,     'n_features':10},
	        {'layer':gadann.ActivationLayer, 'activation':gadann.softmax}
	    ],
	    updater = gadann.SgdUpdater(learning_rate=0.1, weight_cost=0.00)
	)

	gadann.BatchGradientDescentTrainer(model, model.updater).train(self.train_features, self.train_labels_onehot, n_epochs=1)

	train_accuracy = model.evaluate(self.test_features, self.test_labels)
	test_accuracy = model.evaluate(self.test_features, self.test_labels)
	print("Training set accuracy = " + str(train_accuracy*100) + "%")
	print("Test set accuracy = " + str(test_accuracy*100) + "%")

------------------------------
About Me
------------------------------
GADANN was written by Daniel Pineo.  I specialize in developing high-performance algorithms based on state-of-the-art research in computer vision and machine learning.

I can be contacted at daniel@pineo.net

