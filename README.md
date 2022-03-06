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

## 1.9 CNN example (Complete)

![image](https://user-images.githubusercontent.com/60442877/156944308-0d4fe3ce-5236-487e-85de-3dfa9b2e6e11.png)
 
Convolution layer -> Pooling Layer -> onvolution layer -> Pooling Layer -> Fully Connected Layer -> Fully Connected Layer -> Output Layer(SoftMax)
