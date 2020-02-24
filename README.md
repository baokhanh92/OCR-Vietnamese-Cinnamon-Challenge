# [OCR VIETNAMESE]()
---
## [ 1. OVERVIEW ]()

[ **1.2 About the data**: ]() 

**1. Cinnamon challenge datatset:**  
- Given an image of a handwritten line, participants are required to create an OCR model to transcribe the image into text.
- There are three folder with 2 samples image folder and 1 private test folder.

---
## [ 2. THE RESULT ]()

### With only 10 epochs train, the model val-loss is around ~0.40*
![](https://i.imgur.com/Wfo8vOd.png)

[ **2.1 Data preprocessing**: ]() 
- For this problem, I used CV2 to make the binary image, then blur the binary image and used the edged version for training later.
- I copied the edged images into another folder.
![](https://i.imgur.com/0SabmiB.png)
![](https://i.imgur.com/6hJyD7n.png)

[ **2.2 Model**: ]() 
- For the OCR problem, the basic knowledge is to use CRNN + CTC.
- With CNN layers: the model will extract features of images and learn them.
- CTC to caculate the loss function.
- RNN: The text is fed into the neural network character by character and the network is triggered to generate a sequence of characters.
![](https://i.imgur.com/KhzrHmo.png)

---




