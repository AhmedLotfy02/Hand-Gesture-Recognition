<h1 align="center">Hand Gesture Recognition 🤙</h1>

</div>
<div align="center">
   <img align="center"  width="525px" src="https://img.freepik.com/free-vector/set-hand-gesture_1308-24432.jpg?w=740&t=st=1686630042~exp=1686630642~hmac=ef6860e73210bad0dded60a618cd8b015f94ffd6f5c1a2aad56f5d11fbb7655b" alt="logo">


   
</div>
<br>

## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#Overview"> 📝 Overview</a>
- <a href ="#Achievement"> 🎉 Our Achievement</a>
- <a href ="#started">  🏁 Get Started</a>
- <a href ="#modules"> 📜 Project pipeline</a>
  - <a href ="#preprocessing"> 🧹 Preprocessing</a>
  - <a href ="#feature"> ✂️ Feature Extraction</a>
  - <a href ="#classification"> ❓ Classification</a>
- <a href ="#report"> 📰 Project report</a>
- <a href ="#contributors"> ✍️ Contributors</a>
<br>

## 📑 Overview <a id = "Overview"></a>
- Given an image containing a single hand, the system classifies the hand gesture into one of six digits (from 0 to 5).
- The project was done for "Pattern Recognition and Neural Networks" college course to put our understanding of machine learning algorithms into practice with a real world problem.


## 🎉 Our Achievement <a id = "Achievement"></a>
We  have been ranked as the **5th team** on the leader-board out of 20 teams with an accuracy of **66%** on the hidden test set 


## 🏁 Get Started <a id = "started"></a>
<blockquote>
  <p>This is a list of needed steps to set up your project locally, to get a local copy up and running follow these instructions.
 </p>
</blockquote>
<ol>
  <li><em>Clone the repository</em>
    <div>
        <code>$ git clone https://github.com/AhmedLotfy02/Hand-Gesture-Recognition.git</code>
    </div>
  </li>
     <li><em>Libraries Needed</em>
    <div>
        <code>HoG Skimage</code>
       <code>OpenCV</code>
       <code>NumPy</code>
       <code>sklearn</code>
        <code>OS</code>
        <code>Pickle</code>
        <code>time</code>
    </div>
  </li>
  <li><em>You need to run **_final.py_**. Both **_Preprocess.py_** and **_features_extraction.py_** need to be in the same directory as **_final.py_**</em>
  </li>
    <li><em>Also needed is to have both _**SVC.sav**_ and _**pca_model.pkl**_ in the same directory as the aforementioned files.</em>
  </li>
  <li><em>The pics need to be in one of the following formats [**_jpg_**, **_jpeg_**, **_png_**, **_bmp_**, **_gif_**]</em>
  </li>
   <li><em>Create a folder _**data**_ that includes the test images with numerical names **_i.e. 1.png_**</em>
  </li>

</ol>


## 📜 Project pipeline <a id = "modules"></a>
<p>A complete machine learning pipeline, composed of the following modules:</p>


<div align="center">
   <img align="center" width="525px" src="https://i.postimg.cc/Z5BXjdyt/Preprocessing-1.png" alt="logo">
</div>
<br>

## 🧹 Preprocessing <a id = "preprocessing"></a>

<ol>
  <li><em>Convert to gray</em>
    <div>
        <code> to simplify the image by removing color information and focusing on intensity values.</code>
    </div>
  </li>
     <li><em>Apply gaussian blur</em>
    <div>
        <code> to reduce noise and smooth out details.</code>
    </div>
  </li>
    </li>
     <li><em>Apply threshold</em>
    <div>
        <code>to convert the grayscale image into a binary image. It uses the Otsu's thresholding method to automatically determine the threshold value.
</code>
    </div>
  </li>
  </li>
     <li><em>Find contours</em>
    <div>
        <code>Contours: the boundaries of objects in an image.</code>
    </div>
   </li>
    <li><em>Find largest contour</em>
    <div>
        <code> based on the contour area.</code>
    </div>
  </li>
  <li><em>Create mask</em>
    <div>
        <code> using the largest contour. </code>
    </div>
  </li>
  <li><em>Apply mask</em>
    <div>
        <code>  to keep only the region of interest (ROI) within the contour.</code>
    </div>
  </li>
  
   </li>
  <li><em>Resize image</em>
    <div>
        <code> to a predefined target size. This ensures that all images have the same dimensions for further processing.</code>
    </div>
  </li>
  
   <li><em>Normalize image</em>
    <div>
        <code> Normalizes the image by scaling its pixel values to the range [0, 1] to improve the performance of machine learning algorithms.</code>
    </div>
  </li>
  
   <li><em>Image enhancement</em>
    <div>
        <code> gamma correction is applied to adjust the brightness of the image.  </code><br>
        <code> histogram equalization is performed to improve the contrast of the image.  </code>
    </div>
  </li>
  
  <li><em>Background subtraction</em>
  </li>
  
  <li><em>Hand segmentation</em>
    <div>
        <code> Shadow Removal is applied via thresholding.  </code><br>
         <code> Hand Masking using the color range of the human skin.</code><br>
          <code> Find Contours of the hand. </code><br>
           <code> Morphological operations are applied to the drawn contours.</code><br>
    </div>
  </li>

</ol>

## ✂️ Feature Extraction <a id = "feature"></a>


<ul>
  <li>Histogram of Oriented Gradients(HOG)</li>
  <li>Principal Component Analysis (PCA)</li>
</ul>

<br>


## ❓ Classification <a id = "classification"></a>

<ul>
  <li> SVM (Support Vector Machine) model </li>
    </ul>
  </li>
</ul>


<br>


## 📰 project Report <a id = "report"></a>
https://docs.google.com/document/d/1uywz1iWL-lTDIG7x8srUjqE_ajOLCpR4LI4gCEy6a9Y/edit
<br>
<!-- Contributors -->
## ✍️ Contributors <a id = "contributors"></a>

<!-- Contributors list -->
<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/AhmedLotfy02"><img src="https://avatars.githubusercontent.com/u/76037906?v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Lotfy</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/MohamedWw"><img src="https://avatars.githubusercontent.com/u/64079821?v=4" width="150px;" alt=""/><br /><sub><b>Mohamed Walid</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/waleedhesham446" ><img src="https://avatars.githubusercontent.com/u/72695729?v=4" width="150px;" alt=""/><br /><sub><b>Walid Hesham</b></sub></a><br />
    </td>
     <td align="center"><a href="https://github.com/hebaashraf21"><img src="https://github.com/hebaashraf21.png" width="150px;" alt=""/><br /><sub><b>Heba Ashraf</b></sub></a><br /></td>
  </tr>
</table>

<br>








