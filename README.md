# carbon
My Machine Learning Tools

I am trying to build a machine learning tool box of my own interest. All the algorithms will be written in Python, and my goal for this project is that all of them must have very simple interface and easy to operate in Python interactive mode.

For now, I am working on some basic supervised learning tools, and more things will be added day by day.
For supervised learning, now I have kNN, ID3 and NaiveBayes

By simple interface, I mean 2 aspects.
First, all supervised learning tools have 6 core functions: 
1. train(), used to build a model base on training data
2. view(), used to view a model trained
3. classify(), used to classify a new instance
4. test(), used to test on test data
5. save(), used to save the model for later usage
6. grab(), read the model saved
With these 6 functions, all the classification task can be done easily.

Second, all the structure of the objects can be checked easily so it will be easier for later modification.

To use the tools in this folder, you only need to follow the example in example.doc, a sample data is provided too.
