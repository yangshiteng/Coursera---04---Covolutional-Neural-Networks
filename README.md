# Week 1

## 1.1 Edge Detection

- In the Neural Networks for the task of image recognition,
- the early layers of the neural network detect edges of image
- then, some later layers detect part of the object
- and, even later layers, detect the whole object
![image](https://user-images.githubusercontent.com/60442877/156870138-2ad3217c-ccac-46ee-ad39-007bfa6e3b57.png)
![image](https://user-images.githubusercontent.com/60442877/156870203-120de7ab-6032-4a8a-ad57-344210f477dc.png)

## 1.2 Edge Detection with Filter and Convolution Operation (卷积运算)

![image](https://user-images.githubusercontent.com/60442877/156870380-a0532378-b83f-4977-b534-da4bad98e7e6.png)
![image](https://user-images.githubusercontent.com/60442877/156870505-472c6f36-4b7b-4c1f-b900-0ae57de4d63a.png)
![image](https://user-images.githubusercontent.com/60442877/156870691-5eaf6268-9516-4f92-8739-d1a8ccde6357.png)
![image](https://user-images.githubusercontent.com/60442877/156870786-d664c0b6-561f-4370-97d0-a1ec043282cf.png)
![image](https://user-images.githubusercontent.com/60442877/156871083-c98879a7-76e9-45b6-af57-419bfa9abaed.png)

- Learn these w1,w2,...,w9 from back-propagation

## 1.3 Convolution Operation with Padding

- every time you apply the convolution operation, your output image shrinks
- the pixels in the edge or cornner of the image can only be touched only once, while the pixels in the middle of image can be touched more than once, this will results in the information loss in the edge of the image 
- With the padding, we can solve above 2 downside of convolution operation. One aspect is that, with padding, the size of output image can be the same with the size of input image, another aspect is that the pixels in the edge or corner of input image can be tounched more than once

![image](https://user-images.githubusercontent.com/60442877/156941685-da2df422-6523-4eac-b81b-8c002974c8f1.png)
![image](https://user-images.githubusercontent.com/60442877/156941723-d4bd370a-365a-4d68-a815-ece8937279fa.png)

## 1.4 Convolution Operation with Stride

![image](https://user-images.githubusercontent.com/60442877/156941763-bf148626-d344-46b0-a6cb-460ec0b69139.png)
![image](https://user-images.githubusercontent.com/60442877/156941770-a83b022b-dbbc-49e1-904b-68240486ff83.png)

## 1.5 Convolution Over Volume (RGB image with 3 channels)

![image](https://user-images.githubusercontent.com/60442877/156942215-23c85268-e0c5-4498-9122-288b99a43696.png)

## 1.6 One Layer of a Convolutional Network

![image](https://user-images.githubusercontent.com/60442877/156942283-7a57de3b-cfe1-43ab-a040-b40171005062.png)
![image](https://user-images.githubusercontent.com/60442877/156942314-8821246a-c786-4125-ba81-c72947aee211.png)
![image](https://user-images.githubusercontent.com/60442877/156942399-33c524a4-24e5-475e-b136-24d47638756e.png)

## 1.7 Simple CNN Example

![image](https://user-images.githubusercontent.com/60442877/156942893-29109ed5-b40e-44a0-9f8a-a6c602a13764.png)

## 1.8 Pooling Layers

![image](https://user-images.githubusercontent.com/60442877/156942904-e4b509cd-0c0e-41c3-9e54-589993f2bbf4.png)
![image](https://user-images.githubusercontent.com/60442877/156942914-7a7c9354-24ab-43e1-b224-84afdd09aced.png)
![image](https://user-images.githubusercontent.com/60442877/156942951-dbd1a6f0-f055-4010-9559-0921f74bae72.png)
![image](https://user-images.githubusercontent.com/60442877/157139633-e50d0c54-b56b-4d1b-b4ed-77f4ebf71fcc.png)


## 1.9 CNN example (Complete)

![image](https://user-images.githubusercontent.com/60442877/156944308-0d4fe3ce-5236-487e-85de-3dfa9b2e6e11.png)
 
Convolution layer -> Pooling Layer -> onvolution layer -> Pooling Layer -> Fully Connected Layer -> Fully Connected Layer -> Output Layer(SoftMax)

## 1.10 Why Convolutions?

- Reduce the parameters to be learned in back-propagation
- Parameter Sharing: A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image
- Sparsity of Connection: In each layer, each output value depends only on a small number of inputs
- make CNN be very good at capturing the "Translation Invariance" (a picture of cat shifted a couple of pixels to the right is still pretty clearly cat, and CNN can find the fact that an image shifted a few pixels should result in pretty similar features and should probably be assigned the same output label)



# Week 2

## Outline

Classic Networks:
1. LeNet-5
2. AlexNet
3. VGG
4. ResNet

### LeNet-5

- The goal of LeNet-5 is to recognize handwritten digits
- Trained on grayscale images (only one channel)(32x32x1)
- quite old neural network model
![image](https://user-images.githubusercontent.com/60442877/157272246-3205e030-0a66-476e-b489-41d0be30ae89.png)

### AlexNet

- input image is 227x227x3
![image](https://user-images.githubusercontent.com/60442877/157274693-1d9b69d7-00bb-4d2e-806c-e49e7b463d51.png)

### VGG-16

- filers are all 3x3 with stride = 1 and same padding
- all max-pooling are 2x2 with stride = 2
- The 16 refers to the fact that this has 16 layers that have weights
![image](https://user-images.githubusercontent.com/60442877/157283612-6d2fa97e-42e7-4a17-8d8d-c66fe4d33c0b.png)

### ResNets

- Residual Networks
- Very, very deep neural networks are difficult to train, because of vanishing and exploding gradient types of problems
- ResNets are built out of something called a residual block
- Skip Connection and Residual Block
![image](https://user-images.githubusercontent.com/60442877/157289100-0e17e14e-7147-4e2c-af8e-57be723db844.png)
![image](https://user-images.githubusercontent.com/60442877/157289839-e4c8f57f-4994-4cd4-a37c-ab505615c4c5.png)
- If you train the plain Neural network (without residual blocks), empirically, you find that as you increase the number of layers, the training error will tend to decrease after a while but then they will tend to go back up. However, in theory, as you make a neural network deeper, it should only do better and better on the training set.
- Residual Blocks really help with the vanishing and exploding gradient problems, and allow you to train much deeper neural networks without loss in performance

### Why ResNets work?

- ResNets can train very very deep Neural Networks without the loss in performance
![image](https://user-images.githubusercontent.com/60442877/157297719-2ae4daae-f777-4d4b-b02c-2cbd2c456bdc.png)
![image](https://user-images.githubusercontent.com/60442877/158300358-a2003d9b-4fba-460a-b2ae-1df742f07a9e.png)

## Convolution Layer with 1x1 filter

- can be used to shrink the number of channels of input volumn, and therefore, save on computation in some networks
- can even increase the number of channels of input volumn
![image](https://user-images.githubusercontent.com/60442877/157305937-4f2ff1e1-5364-4bb7-be5d-fd1a9f52677b.png)

## Inception Network (Inception Moduel)

- When designing a layer for a ConvNet, you might have to pick, do you want a 1 by 3 filter, or 3 by 3, or 5 by 5, or do you want a pooling layer? What the inception network does is it says, why should you do them all? And this makes the network architecture more complicated, but it also works remarkably well. 
![image](https://user-images.githubusercontent.com/60442877/158007137-3b5043c5-40ff-4fed-bff6-2885801fb6fb.png)
- It has the problem of computational cost
![image](https://user-images.githubusercontent.com/60442877/158007446-55ce5c7e-0d30-4acd-bea1-05d85c9c5d0e.png)
- Reduece the computation by introducing the 1x1 convolution
![image](https://user-images.githubusercontent.com/60442877/158007616-1ac6e19e-f7de-45ed-ba52-55b8f0e5801f.png)
- if you are building a layer of a neural network and you don't want to have to decide, do you want a 1 by 1, or 3 by 3, or 5 by 5, or pooling layer, the inception module lets you say let's do them all, and let's concatenate the results. And then we run to the problem of computational cost. And what you saw here was how using a 1 by 1 convolution, you can create this bottleneck layer thereby reducing the computational cost significantly. Now you might be wondering, does shrinking down the representation size so dramatically, does it hurt the performance of your neural network? It turns out that so long as you implement this bottleneck layer so that within reason, you can shrink down the representation size significantly, and it doesn't seem to hurt the performance, but saves you a lot of computation. So these are the key ideas of the inception module.
![image](https://user-images.githubusercontent.com/60442877/158044872-a9ed1675-9373-446c-a8bd-da2ce19cf048.png)
![image](https://user-images.githubusercontent.com/60442877/158044976-c2761d28-a31e-4314-bce2-2145e576d5f0.png)

## MobileNet

- MobileNets will allow you to build and deploy neural networks that work even in low compute environment, such as a mobile phone. 
- Why do you need another neural network architecture? It turns out in other neural networks you've learned about so far are quite computationally expensive. If you want your neural network to run on a device with less powerful CPU or a GPU at deployment, then there's another neural network architecture called the MobileNet that could perform much better. 
![image](https://user-images.githubusercontent.com/60442877/158045217-aebdc15e-533e-4302-94cd-aed2d268e2c5.png)
![image](https://user-images.githubusercontent.com/60442877/158045577-ab93a4d4-1b3a-432e-aa17-4a88cc51fce5.png)

### MobileNet - Depthwise Separable Convolution

![image](https://user-images.githubusercontent.com/60442877/158045610-8c921762-ceab-48c8-8ebf-593c4d416aba.png)

#### Depthwise Convolution (pooling operation style)

The operation of Depthwise Convolution is very similar to the pooling layer operation, except that we need to do the sum of dot product instead of the max pooling or average pooling.
![image](https://user-images.githubusercontent.com/60442877/158046083-d9c449bc-87fa-4322-9822-b1f26ace1ee6.png)

#### Pointwise Convolution (just usual convolution operation)

![image](https://user-images.githubusercontent.com/60442877/158046268-f1552bc4-e61b-4efc-b045-6882f96f275e.png)
 
#### Cost Comparison

![image](https://user-images.githubusercontent.com/60442877/158046359-90763a5e-ca80-4b3d-a0c6-decc7257b496.png)
![image](https://user-images.githubusercontent.com/60442877/158046448-bbd06b5e-8e72-447a-a6aa-74d4e8d5613e.png)


### MobileNet Architecture

![image](https://user-images.githubusercontent.com/60442877/158047179-ea9af8f4-0367-47b2-99c8-112f3d422682.png)
![image](https://user-images.githubusercontent.com/60442877/158047390-8f06825b-3f49-494d-9f51-e1b5b18a0c09.png)

## EfficientNet

MobileNet V1 and V2 gave you a way to implement a neural network, that is more computationally efficient. But is there a way to tune MobileNet, or some other architecture, to your specific device? Maybe you're implementing a computer vision algorithm for different brands of mobile phones with different amounts of compute resources, or for different edge devices. If you have a little bit more computation, maybe you have a slightly bigger neural network and hopefully you get a bit more accuracy, or if you are more computationally constraint, maybe you want a slightly smaller neural network that runs a bit faster, at the cost of a little bit of accuracy. How can you automatically scale up or down neural networks for a particular device? EfficientNet, gives you a way to do so.

## Practical Advice for using ConvNets

### Open Source Implementation

So if you're developing a computer vision application, a very common workflow would be to pick an architecture that you like, maybe one of the ones you learned about in this course. Or maybe one that you heard about from a friend or from some literature. And look for an open source implementation and download it from GitHub to start building from there. One of the advantages of doing so also is that sometimes these networks take a long time to train, and someone else might have used multiple GPUs and a very large dataset to pretrain some of these networks. And that allows you to do transfer learning using these networks which we'll discuss in the next video as well. Of course, if you're computer vision researcher implementing these things from scratch, then your workflow will be different. And if you do that, then do contribute your work back to the open-source community. But because so many vision researchers have done so much work implementing these architectures, I found that often starting with open-source implementations is a better way, or certainly a faster way to get started on a new project.

### Transfer Learning

- If you're building a computer vision application rather than training the ways from scratch, from random initialization, you often make much faster progress if you download ways that someone else has already trained on the network architecture and use that as pre-training and transfer that to a new task that you might be interested in. The computer vision research community has been pretty good at posting lots of data sets on the Internet so if you hear of things like Image Net, or MS COCO, or Pascal types of data sets, these are the names of different data sets that people have post online and a lot of computer researchers have trained their algorithms on. Sometimes these training takes several weeks and might take many GP use and the fact that someone else has done this and gone through the painful high-performance search process, means that you can often download open-source ways that took someone else many weeks or months to figure out and use that as a very good initialization for your own neural network. And use transfer learning to sort of transfer knowledge from some of these very large public data sets to your own problem.
![image](https://user-images.githubusercontent.com/60442877/158050500-ada51221-a19d-4c2e-9230-8de7f4fcfee8.png)
- In all the different disciplines, in all the different applications of deep learning, I think that computer vision is one where transfer learning is something that you should almost always do unless, you have an exceptionally large data set to train everything else from scratch yourself. But transfer learning is just very worth seriously considering unless you have an exceptionally large data set and a very large computation budget to train everything from scratch by yourself.


### Data Augmentation

Most computer vision task could use more data. And so data augmentation is one of the techniques that is often used to improve the performance of computer vision systems.

Data Augmentation Methods:
1. Mirroring
![image](https://user-images.githubusercontent.com/60442877/158095235-012a70ae-0243-47b5-9d13-0b64c3c68bc9.png)
2. Random Cropping
![image](https://user-images.githubusercontent.com/60442877/158095365-359516bd-35b3-4908-9cc5-ffb411b1644b.png)
3. Rotation
4. Shearing
5. Local warping..
6. Color shifting
![image](https://user-images.githubusercontent.com/60442877/158096834-8af63b15-9bda-4b82-af7c-c44e34c4c659.png)


### State of Computer Vision

Deep learning has been successfully applied to computer vision, natural language processing, speech recognition, online advertising, logistics, many, many, many problems. There are a few things that are unique about the application of deep learning to computer vision, about the status of computer vision.
![image](https://user-images.githubusercontent.com/60442877/158098553-c7530491-9c7a-4a22-91a7-0b31018fc4ac.png)

### Tips for doing well on benchmarks or winning competitions

1. Ensembling 
2. Multi-crop at test time
![image](https://user-images.githubusercontent.com/60442877/158099863-bf9267ba-86da-400b-a8ff-63ba20d8131c.png)
![image](https://user-images.githubusercontent.com/60442877/158100047-37658c44-03f4-49a9-a327-fc412f242f4e.png)

# Week 3

This week you learn about object detection. This is one of the areas of computer vision that's just exploding and is working so much better than just a couple of years ago. In order to build up to object detection, you first learn about object localization, which means not only do you have to label this as say a car but the algorithm also is responsible for putting a bounding box, or drawing a red rectangle around the position of the car in the image. So that's called the classification with localization problem, where the term localization refers to figuring out where in the picture is the car you've detective. 

![image](https://user-images.githubusercontent.com/60442877/159142563-d84e5bb2-193a-4ccd-a31e-d89e7eea0318.png)

Later this week, you then learn about the detection problem where now there might be multiple objects in the picture and ,you have to detect them all and and localized them all. And if you're doing this for an autonomous driving application, then you might need to detect not just other cars, but maybe other pedestrians and motorcycles and maybe even other objects. So you'll see that later this week.

![image](https://user-images.githubusercontent.com/60442877/159142636-06faa530-7a1c-4d9a-8a49-ce143f4c6af9.png)

## Object Localization

![image](https://user-images.githubusercontent.com/60442877/159142757-75bdaac4-5682-4d62-bd40-cfbd64bb425c.png)

So now if your training set contains not just the object class label, which a neural network is trying to predict up here, but it also contains four additional numbers. Giving the bounding box then you can use supervised learning to make your algorithm outputs not just a class label but also the four parameters to tell you where is the bounding box of the object you detected. So in this example the ideal bx might be about 0.5 because this is about halfway to the right to the image. by might be about 0.7 since it's about maybe 70% to the way down to the image. bh might be about 0.3 because the height of this red square is about 30% of the overall height of the image. And bw might be about 0.4 let's say because the width of the red box is about 0.4 of the overall width of the entire image.

### Defining the target label y and loss function

![image](https://user-images.githubusercontent.com/60442877/159142976-47ea8de8-c03e-4a5c-b539-c07bce6b873a.png)


## Landmark Detection

In the previous video, you saw how you can get a neural network to output four numbers of bx, by, bh, and bw to specify the bounding box of an object you want a neural network to localize. In more general cases, you can have a neural network just output X and Y coordinates of important points and image, sometimes called landmarks, that you want the neural networks to recognize.

![image](https://user-images.githubusercontent.com/60442877/159150485-ebe2d1d4-6452-4ca6-b2da-96a1ae3d48ca.png)


## Sliding Windows Detection 

Let's say you want to build a car detection algorithm. Here's what you can do. You can first create a label training set, so x and y with closely cropped examples of cars. And for our purposes in this training set, you can start off with the one with the car closely cropped images. Meaning that x is pretty much only the car. So, you can take a picture and crop out and just cut out anything else that's not part of a car. So you end up with the car centered in pretty much the entire image. Given this label training set, you can then train a ConvNet that inputs an image, like one of these closely cropped images. And then the job is to output y, zero or one, is there a car or not. Once you've trained up this ConvNet you can then use it in Sliding Windows Detection.

![image](https://user-images.githubusercontent.com/60442877/159151531-ac7d984c-98b4-4ab8-99b1-baea13344e08.png)

If you have a test image like this what you do is you start by picking a certain window size, shown down there. And then you would input into this ConvNet a small rectangular region. So, take just this below red square, input that into the ConvNet, and have a ConvNet make a prediction. And presumably for that little region in the red square, it'll say, no that little red square does not contain a car. In the Sliding Windows Detection Algorithm, what you do is you then pass as input a second image now bounded by this red square shifted a little bit over and feed that to the ConvNet. So, you're feeding just the region of the image in the red squares of the ConvNet and run the ConvNet again. And then you do that with a third image and so on. And you keep going until you've slid the window across every position in the image. 

![image](https://user-images.githubusercontent.com/60442877/159151955-88d3f4a9-ca62-41a9-8b07-1d4e01969e24.png)

Now there's a huge disadvantage of Sliding Windows Detection, which is the computational cost. Because you're cropping out so many different square regions in the image and running each of them independently through a ConvNet. And if you use a very coarse stride, a very big stride, a very big step size, then that will reduce the number of windows you need to pass through the ConvNet, but that courser granularity may hurt performance. Whereas if you use a very fine granularity or a very small stride, then the huge number of all these little regions you're passing through the ConvNet means that means there is a very high computational cost.

Fortunately however, this problem of computational cost has a pretty good solution. In particular, the Sliding Windows Object Detector can be implemented convolutionally or much more efficiently

