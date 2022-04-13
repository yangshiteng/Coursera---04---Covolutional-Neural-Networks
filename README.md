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

### Convolutional Implementation of Sliding Windows

#### Turning FC (Fully Connected) layer into convolutional layers

![image](https://user-images.githubusercontent.com/60442877/159154099-c22d949a-46cb-4bcb-919b-c4c7eacd8529.png)

#### Implement Sliding Windows convolutionally

![image](https://user-images.githubusercontent.com/60442877/159154521-cc0746d1-35ed-4e9e-bdca-ac8549a59786.png)

In the above figure, the 14x14x3 images' convolutional neural network is used to perform the classification task, and without change the setting of filters or max pooling, we can apply this Convenet to 16x16x3 image, and the final output is 2x2x4, which shows 4 classification results for 4 different sliding.

![image](https://user-images.githubusercontent.com/60442877/159154686-910e80a1-2194-4b2e-921f-ce46da7e3cf0.png)

To implement sliding windows, previously, what you do is you crop out a region. Let's say this is 14 by 14 and run that through your convnet and do that for the next region over, then do that for the next 14 by 14 region, then the next one, then the next one, then the next one, then the next one and so on, until hopefully that one recognizes the car. But now, instead of doing it sequentially, with this convolutional implementation that you saw in the previous slide, you can implement the entire image, all maybe 28 by 28 and convolutionally make all the predictions at the same time by one forward pass through this big convnet and hopefully have it recognize the position of the car. So that's how you implement sliding windows convolutionally and it makes the whole thing much more efficient. Now, this algorithm still has one weakness, which is the position of the bounding boxes is not going to be too accurate. In the next video, let's see how you can fix that problem.

### Bounding Box Predictions

In the last video, you learned how to use a convolutional implementation of sliding windows. That's more computationally efficient, but it still has a problem of not quite outputting the most accurate bounding boxes. In this video, let's see how you can get your bounding box predictions to be more accurate. 

#### YOLO Algorithm

YOLO stands for, you only look once 

![image](https://user-images.githubusercontent.com/60442877/159190182-95fb4379-691b-42f2-bb1d-a4551d4c2265.png)

And so what you do is you have an input X which is the input image like that, and you have these target labels Y which are 3 by 3 by 8, and you use back-propagation to train the neural network to map from any input X to this type of output volume Y. So the advantage of this algorithm is that the neural network outputs precise bounding boxes as follows. So at test time, what you do is you feed an input image X and run forward prop until you get this output Y. And then for each of the nine outputs of each of the 3 by 3 positions in which of the output, you can then just read off 1 or 0. Is there an object associated with that one of the nine positions? And that there is an object, what object it is, and where is the bounding box for the object in that grid cell? And so long as you don't have more than one object in each grid cell, this algorithm should work okay. And the problem of having multiple objects within the grid cell is something we'll address later.

![image](https://user-images.githubusercontent.com/60442877/159190561-8d576e77-43d9-493f-873c-f4b8bb4c38b4.png)


## Intersection Over Union Function (Evaluate the bounding box)

So how do you tell if your object detection algorithm is working well? In this video, you'll learn about a function called, "Intersection Over Union". And as we use it both for evaluating your object detection algorithm, as well as in the next video, using it to add another component to your object detection algorithm, to make it work even better.

![image](https://user-images.githubusercontent.com/60442877/159191109-a753955a-4ba8-4b72-8886-fc141b30790b.png)

In the object detection task, you expected to localize the object as well. So if that's the ground-truth bounding box, and if your algorithm outputs this bounding box in purple, is this a good outcome or a bad one? So what the intersection over union function does, or IoU does, is it computes the intersection over union of these two bounding boxes. So, the union of these two bounding boxes is this area, is really the area that is contained in either bounding boxes, whereas the intersection is this smaller region here. So what the intersection of a union does is it computes the size of the intersection. So that orange shaded area, and divided by the size of the union, which is that green shaded area. And by convention, the low compute division task will judge that your answer is correct if the IoU is greater than 0.5. And if the predicted and the ground-truth bounding boxes overlapped perfectly, the IoU would be one, because the intersection would equal to the union. But in general, so long as the IoU is greater than or equal to 0.5, then the answer will look okay, look pretty decent. And by convention, very often 0.5 is used as a threshold to judge as whether the predicted bounding box is correct or not.


## Non-Max Suppression (detect each object only once)

One of the problems of Object Detection as you've learned about this so far, is that your algorithm may find multiple detections of the same objects. Rather than detecting an object just once, it might detect it multiple times. Non-max suppression is a way for you to make sure that your algorithm detects each object only once.

![image](https://user-images.githubusercontent.com/60442877/159192743-357c7719-78db-4514-87a0-d3a4246a8664.png)

## Anchor Boxes (make a grid cell detect multiple object)

One of the problems with object detection as you have seen it so far is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects? Here is what you can do. You can use the idea of anchor boxes.

![image](https://user-images.githubusercontent.com/60442877/159330752-66659101-5629-4927-a70c-76d110477297.png)

![image](https://user-images.githubusercontent.com/60442877/159331832-1fd1ce6e-190b-4534-b755-ba791ed961b2.png)

![image](https://user-images.githubusercontent.com/60442877/159332765-d2ded804-9792-4cf6-b9b7-546485ae58aa.png)

## YOLO object detection algorithm

You've already seen most of the components of object detection. In this video, let's put all the components together to form the YOLO object detection algorithm. 

![image](https://user-images.githubusercontent.com/60442877/159335280-6385f160-b8b5-40db-a4cd-7327af055fc9.png)

![image](https://user-images.githubusercontent.com/60442877/159335851-80b2c1b4-507a-4ce2-87ed-04a44ff8120a.png)

![image](https://user-images.githubusercontent.com/60442877/159336501-2bdda275-71db-4be9-a3b6-3968679d19c9.png)


## Semantic Segmentation with U-Net

Semantic segmentation, where the goal is to draw a careful outline around the object that is detected so that you know exactly which pixels belong to the object and which pixels don't.

What is semantic segmentation? Let's say you're building a self-driving car and you see an input image like this and you'd like to detect the position of the other cars. If you use an object detection algorithm, the goal may be to draw bounding boxes like these around the other vehicles. This might be good enough for self-driving car, but if you want your learning algorithm to figure out what is every single pixel in this image, then you may use a semantic segmentation algorithm whose goal is to output this. Where, for example, rather than detecting the road and trying to draw a bounding box around the roads, which isn't going to be that useful, with semantic segmentation the algorithm attempts to label every single pixel as is this drivable roads or not, indicated by the dark green there. One of the uses of semantic segmentation is that it is used by some self-driving car teams to figure out exactly which pixels are safe to drive over because they represent a drivable surface.

![image](https://user-images.githubusercontent.com/60442877/159352536-095b445a-7849-45ef-8553-4457c53aebd5.png)

![image](https://user-images.githubusercontent.com/60442877/159354906-50a96c9c-c55f-42b8-9135-07ab338269b6.png)

Here's a familiar convolutional neural network architecture, where you input an image which is fed forward through multiple layers in order to generate a class label y hat. In order to change this architecture into a semantic segmentation architecture, let's get rid of the last few layers and one key step of semantic segmentation is that, whereas the dimensions of the image have been generally getting smaller as we go from left to right, it now needs to get bigger so they can gradually blow it back up to a full-size image, which is a size you want for the output. 

![image](https://user-images.githubusercontent.com/60442877/159357617-17b4e24f-de3a-4d1e-a3e9-0d29b64e1634.png)

Specifically, this is what a unit architecture looks like. As we go deeper into the unit, the height and width will go back up while the number of channels will decrease so the unit architecture looks like this until eventually, you get your segmentation map of the cat. One operation we have not yet covered is what does this look like? To make the image bigger. To explain how that works, you have to know how to implement a transpose convolution. That's semantic segmentation, a very useful algorithm for many computer vision applications where the key idea is you have to take every single pixel and label every single pixel individually with the appropriate class label. As you've seen in this video, a key step to do that is to take a small set of activations and to blow it up to a bigger set of activations. In order to do that, you have to implement something called the transpose convolution, which is important operation that is used multiple times in the unit architecture.

![image](https://user-images.githubusercontent.com/60442877/159357670-1d1544dd-d5ff-4350-8548-db8101b2f66f.png)

### Transpose Convolutions

The transpose convolution is a key part of the unit architecture. How do you take a two-by-two inputs and blow it up into a four- by-four-dimensional output? The transpose convolution lets you do that.

You're familiar with the normal convolution in which a typical layer of a new network may input a six by six by three image, convolve that with a set of, say, three by three by three filters and if you have five of these, then you end up with an output that is four by four by five. A transpose convolution looks a bit difference. You might inputs a two-by-two, said that activation, convolve that with a three by three filter, and end up with an output that is four by four, that's bigger than the original inputs.

![image](https://user-images.githubusercontent.com/60442877/159361094-bef9ca40-dcb3-4eef-aa87-83cce3ef23a4.png)

![image](https://user-images.githubusercontent.com/60442877/159362212-6c871f66-4c90-4ce6-aee1-71c92c77eaaf.png)
 
![image](https://user-images.githubusercontent.com/60442877/159362520-8d44511e-bb73-4c75-931b-1d1e8ea90389.png)

![image](https://user-images.githubusercontent.com/60442877/159362842-7c2df038-1a23-4dd5-8e82-58ff720e81c7.png)

![image](https://user-images.githubusercontent.com/60442877/159363039-80289440-6d1e-4ea9-89da-9ed94ec392d9.png)


### U-Net Architecture

![image](https://user-images.githubusercontent.com/60442877/159366051-1f990637-bd98-48da-a0e5-d2bed99a4bbe.png)

![image](https://user-images.githubusercontent.com/60442877/159499335-44a79b0d-d29b-4f92-9178-08932bd50e3e.png)

The dimensions of this output layer is going to be h by w, so the same dimensions as our original input by number of classes. So if you have three classes to try and recognize, this will be three. If you have ten different classes to try to recognize in your segmentation at then that last number will be ten. And so what this does is for every one of your pixels you have h by w pixels, you have, an array or a vector, essentially of n classes numbers that tells you for our pixel how likely is that pixel to come from each of these different classes. And if you take a arg max over these n classes, then that's how you classify each of the pixels into one of the classes, and you can visualize it like the segmentation map showing on the right


# Week 4

## Face Verification and Recognition

The video you just saw demoed both face recognition as well as liveness detection. The latter meaning making sure that you are a live human. It turns out liveness detection can be implemented using supervised learning as well to predict live human versus not live human but I want to spend less time on that. Instead, I want to focus our time on talking about how to build the face recognition portion of the system.

![image](https://user-images.githubusercontent.com/60442877/161120531-2f4b59ad-285c-4a74-9e2a-da6db450f6aa.png)

In the face recognition literature, people often talk about face verification and face recognition. This is the face verification problem which is if you're given an input image as well as a name or ID of a person and the job of the system is to verify whether or not the input image is that of the claimed person. So, sometimes this is also called a one to one problem where you just want to know if the person is the person they claim to be. So, the recognition problem is much harder than the verification problem. To see why, let's say, you have a verification system that's 99 percent accurate. So, 99 percent might not be too bad, but now suppose that K is equal to 100 in a recognition system. If you apply this system to a recognition task with a 100 people in your database, you now have a hundred times of chance of making a mistake and if the chance of making mistakes on each person is just one percent. So, if you have a database of a 100 persons, and if you want an acceptable recognition error, you might actually need a verification system with maybe 99.9 or even higher accuracy before you can run it on a database of 100 persons

## One Shot Learning

One of the challenges of face recognition is that you need to solve the one-shot learning problem. What that means is that for most face recognition applications you need to be able to recognize a person given just one single image, or given just one example of that person's face. And, historically, deep learning algorithms don't work well if you have only one training example.

![image](https://user-images.githubusercontent.com/60442877/161122740-34acee3a-003e-4ad4-9e7f-ba229f1283cf.png)

Let's see an example of what this means, and talk about how to address this problem. Let's say you have a database of four pictures of employees in you're organization. These are actually some of my colleagues at Deeplearning AI; Khan, Danielle, Younes, and Thian. Now let's say someone shows up at the office and they want to be let through the turnstile. What the system has to do is, despite ever having seen only one image of Danielle, to recognize that this is actually the same person. And, in contrast, if it sees someone that's not in this database, then it should recognize that this is not any of the four persons in the database. So in the one shot learning problem, you have to learn from just one example to recognize the person again. And you need this for most face recognition systems use, because you might have only one picture of each of your employees or of your team members in your employee database. So one approach you could try is to input the image of the person, feed it too a ConvNet. And have it output a label, y, using a softmax unit with four outputs or maybe five outputs corresponding to each of these four persons or none of the above. So that would be 5 outputs in the softmax. But this really doesn't work well. Because if you have such a small training set it is really not enough to train a robust neural network for this task. And also what if a new person joins your team? So now you have 5 persons you need to recognize, so there should now be six outputs. Do you have to retrain the ConvNet every time? That just doesn't seem like a good approach. So to carry out face recognition, to carry out one-shot learning.

![image](https://user-images.githubusercontent.com/60442877/161123902-30a6623d-ad54-42e0-bd81-daae2232145c.png)

So instead, to make this work, what you're going to do instead is learn a similarity function. In particular, you want a neural network to learn a function which going to denote d, which inputs two images and outputs the degree of difference between the two images. So if the two images are of the same person, you want this to output a small number. And if the two images are of two very different people you want it to output a large number. So during recognition time, if the degree of difference between them is less than some threshold called tau, which is a hyperparameter. Then you would predict that these two pictures are the same person. And if it is greater than tau, you would predict that these are different persons.

And so this is how you address the face verification problem. To use this for a recognition task, what you do is, given this new picture, you will use this function d to compare these two images. And maybe I'll output a very large number, let's say 10, for this example. And then you compare this with the second image in your database. And because these two are the same person, hopefully you output a very small number. You do this for the other images in your database and so on.

And based on this, you would figure out that this is actually that person, which is Danielle. And in contrast, if someone not in your database shows up, as you use the function d to make all of these pairwise comparisons, hopefully d will output have a very large number for all four pairwise comparisons. And then you say that this is not any one of the four persons in the database. Notice how this allows you to solve the one-shot learning problem. So long as you can learn this function d, which inputs a pair of images and tells you, basically, if they're the same person or different persons. Then if you have someone new join your team, you can add a fifth person to your database, and it just works fine.


## Siamese Network

The job of the function d, which you learned about in the last video, is to input two faces and tell you how similar or how different they are. A good way to do this is to use a Siamese network.

![image](https://user-images.githubusercontent.com/60442877/161446389-f4ddfdae-c947-40a7-b76d-90f9d3b25c39.png)

You're used to seeing pictures of confidence like these where you input an image, let's say x1. And through a sequence of convolutional and pulling and fully connected layers, end up with a feature vector like that. And sometimes this is fed to a softmax unit to make a classification. We're not going to use that in this video. Instead, we're going to focus on this vector of let's say 128 numbers computed by some fully connected layer that is deeper in the network. And I'm going to give this list of 128 numbers a name. I'm going to call this f of x1, and you should think of f of x1 as an encoding of the input image x1. So it's taken the input image, here this picture of Kian, and is re-representing it as a vector of 128 numbers. The way you can build a face recognition system is then that if you want to compare two pictures, let's say this first picture with this second picture here. What you can do is feed this second picture to the same neural network with the same parameters and get a different vector of 128 numbers, which encodes this second picture. So I'm going to call this second picture. So I'm going to call this encoding of this second picture f of x2, and here I'm using x1 and x2 just to denote two input images. They don't necessarily have to be the first and second examples in your training sets. It can be any two pictures. Finally, if you believe that these encodings are a good representation of these two images, what you can do is then define the image d of distance between x1 and x2 as the norm of the difference between the encodings of these two images.

![image](https://user-images.githubusercontent.com/60442877/161446561-24c6effb-5793-485a-bd1f-28491a28885c.png)

So more formally, the parameters of the neural network define an encoding f of xi. So given any input image xi, the neural network outputs this 128 dimensional encoding f of xi. So more formally, what you want to do is learn parameters so that if two pictures, xi and xj, are of the same person, then you want that distance between their encodings to be small. And in the previous slide, l was using x1 and x2, but it's really any pair xi and xj from your training set. And in contrast, if xi and xj are of different persons, then you want that distance between their encodings to be large. So as you vary the parameters in all of these layers of the neural network, you end up with different encodings. And what you can do is use back propagation to vary all those parameters in order to make sure these conditions are satisfied.


## Triplet Loss

One way to learn the parameters of the neural network, so that it gives you a good encoding for your pictures of faces, is to define and apply gradient descent on the triplet loss function. 

![image](https://user-images.githubusercontent.com/60442877/161448013-a31656cd-434a-4619-90a9-378528d22fd8.png)

This is what gives rise to the term triplet loss, which is that you always be looking at three images at a time. You'll be looking at an anchor image, a positive image, as well as a negative image. 

If f always output zero, then this is 0 minus 0, which is 0, this is 0 minus 0, which is 0, and so, well, by saying f of any image equals a vector of all zero's, you can see almost trivially satisfy this equation. To make sure that the neural network doesn't just output zero, for all the encodings, or to make sure that it doesn't set all the encodings equal to each other. Another way for the neural network to give a trivial outputs is if the encoding for every image was identical to the encoding to every other image, in which case you again get 0 minus 0. To prevent your neural network from doing that, what we're going to do is modify this objective to say that this doesn't need to be just less than equal to zero, it needs to be quite a bit smaller than zero. In particular, if we say this needs to be less than negative Alpha, where Alpha is another hyperparameter then this prevents a neural network from outputting the trivial solutions. 

![image](https://user-images.githubusercontent.com/60442877/161450263-62bcf061-ee88-4929-b75b-7e2ca515e7b6.png)

If you have a training set of say, 10,000 pictures with 1,000 different persons, what you'd have to do is take your 10,000 pictures and use it to generate, to select triplets like this, and then train your learning algorithm using gradient descent on this type of cost function, which is really defined on triplets of images drawn from your training set. Notice that in order to define this dataset of triplets, you do need some pairs of A and P, pairs of pictures of the same person. For the purpose of training your system, you do need a dataset where you have multiple pictures of the same person. That's why in this example I said if you have 10,000 pictures of 1,000 different persons, so maybe you have ten pictures, on average of each of your 1,000 persons to make up your entire dataset. If you had just one picture of each person, then you can't actually train this system. But of course, after having trained a system, you can then apply it to your one-shot learning problem where for your face recognition system, maybe you have only a single picture of someone you might be trying to recognize. But for your training set, you do need to make sure you have multiple images of the same person, at least for some people in your training set, so that you can have pairs of anchor and positive images. 

![image](https://user-images.githubusercontent.com/60442877/161452624-ba041665-8821-4576-9712-6e5390947682.png)

Now, how do you actually choose these triplets to form your training set? One of the problems is if you choose A, P, and N randomly from your training set, subject to A and P being the same person and A and N being different persons, one of the problems is that if you choose them so that they're random, then this constraint is very easy to satisfy. Because given two randomly chosen pictures of people, chances are A and N are much different than A and P. If you choose the triplets randomly, then too many triplets would be really easy and gradient descent won't do anything because you're Neural Network would get them right pretty much all the time. It's only by choosing ''hard'' to triplets that the gradient descent procedure has to do some work to try to push these quantities further away from those quantities

![image](https://user-images.githubusercontent.com/60442877/161452857-8dcfd75a-667c-4fcf-bfec-12f75f911ae8.png)


## Face Verification and Binary Classification

The Triplet Loss is one good way to learn the parameters of a ConvNet for face recognition. There's another way to learn these parameters. Let me show you how face recognition can also be posed as a straight binary classification problem. 

![image](https://user-images.githubusercontent.com/60442877/161453213-cb87a839-6097-4a56-b345-53af6306c366.png)

![image](https://user-images.githubusercontent.com/60442877/161453419-50a44fe6-2b1e-4409-8a58-b41e3cffee06.png)

Another way to train a neural network, is to take this pair of neural networks to take this Siamese Network and have them both compute these embeddings, maybe 128 dimensional embeddings, maybe even higher dimensional, and then have these be input to a logistic regression unit to then just make a prediction. Where the target output will be one if both of these are the same persons, and zero if both of these are of different persons. So, this is a way to treat face recognition just as a binary classification problem. And this is an alternative to the triplet loss for training a system like this. Now, what does this final logistic regression unit actually do? The output y hat will be a sigmoid function, applied to some set of features but rather than just feeding in, these encodings, what you can do is take the differences between the encodings. 

Lastly, just to mention, one computational trick that can help neural deployment significantly, which is that, if this is the new image, so this is an employee walking in hoping that the turnstile the doorway will open for them and that this is from your database image. Then instead of having to compute, this embedding every single time, where you can do is actually pre-compute that, so, when the new employee walks in, what you can do is use this upper components to compute that encoding and use it, then compare it to your pre-computed encoding and then use that to make a prediction y hat. Because you don't need to store the raw images and also because if you have a very large database of employees, you don't need to compute these encodings every single time for every employee database. This idea of free computing, some of these encodings can save a significant computation. And this type of pre-computation works both for this type of Siamese Central architecture where you treat face recognition as a binary classification problem, as well as, when you were learning encodings maybe using the Triplet Loss function as described in the last couple of videos. 

![image](https://user-images.githubusercontent.com/60442877/161453388-7f44f042-eaeb-4dfc-955d-0c8c286e16c1.png)

## Neural Style Transfer

One of the most fun and exciting applications of ConvNet recently has been Neural Style Transfer. You get to implement this yourself and generate your own artwork in the problem exercise. But what is Neural Style Transfer? Let me show you a few examples. Let's say you take this image, this is actually taken from the Stanford University not far from my Stanford office and you want this picture recreated in the style of this image on the right. This is actually Van Gogh's, Starry Night painting. What Neural Style Transfer allows you to do is generated new image like the one below which is a picture of the Stanford University Campus that painted but drawn in the style of the image on the right

![image](https://user-images.githubusercontent.com/60442877/163197627-6f1224e7-87a1-4ce7-9f17-8d5b355f6080.png)

### What are deep ConvNets Learning?

What are deep ConvNets really learning? In this video, I want to share with you some visualizations that will help you hone your intuition about what the deeper layers of a ConvNet really are doing. And this will help us think through how you can implement neural style transfer as well

![image](https://user-images.githubusercontent.com/60442877/163202090-8609635c-3ae8-46ac-9faa-645fee02fe13.png)

![image](https://user-images.githubusercontent.com/60442877/163203107-8c523cc9-0b45-4c31-bd4c-6dd82cd63d02.png)

And what does the neural network then learning at a deeper layers. So in the deeper layers, a hidden unit will see a larger region of the image. Where at the extreme end each pixel could hypothetically affect the output of these later layers of the neural network. So later units are actually seen larger image patches

### Cost Function for Neural Style Transfer

To build a Neural Style Transfer system, let's define a cost function for the generated image. What you see later is that by minimizing this cost function, you can generate the image that you want. Remember what the problem formulation is. You're given a content image C, given a style image S and you goal is to generate a new image G. In order to implement neural style transfer, what you're going to do is define a cost function J of G that measures how good is a particular generated image and we'll use gradient descent to minimize J of G in order to generate this image.

![image](https://user-images.githubusercontent.com/60442877/163209572-e689b581-e02c-4e72-9b71-ede6b1530f1a.png)

How good is a particular image? Well, we're going to define two parts to this cost function. The first part is called the content cost. This is a function of the content image and of the generated image and what it does is it measures how similar is the contents of the generated image to the content of the content image C. And then going to add that to a style cost function which is now a function of (S,G) and what this does is it measures how similar is the style of the image G to the style of the image S. Finally, we'll weight these with two hyper parameters alpha and beta to specify the relative weighting between the content costs and the style cost.

![image](https://user-images.githubusercontent.com/60442877/163209990-91e88db2-fc85-48b4-acf0-1ab0aba254be.png)

The way the algorithm would run is as follows, having to find the cost function J of G in order to actually generate a new image what you do is the following. You would initialize the generated image G randomly so it might be 100 by 100 by 3 or 500 by 500 by 3 or whatever dimension you want it to be. Then we'll define the cost function J of G on the previous slide. What you can do is use gradient descent to minimize this so you can update G as G minus the derivative respect to the cost function of J of G. In this process, you're actually updating the pixel values of this image G which is a 100 by 100 by 3 maybe rgb channel image.

### Content Cost Function

![image](https://user-images.githubusercontent.com/60442877/163264630-52bc6320-430e-46df-9c5c-c64a09672654.png)


### Style Cost Function

![image](https://user-images.githubusercontent.com/60442877/163264983-6e896b2c-9a3e-4584-b222-ca5ca6a39c3a.png)

![image](https://user-images.githubusercontent.com/60442877/163266568-121ff6a3-f91a-4723-b02d-c60d1161acf9.png)


