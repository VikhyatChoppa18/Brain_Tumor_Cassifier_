## Brain MRI tumor detection system


<html>
<head>Brain Mri classification system</head>
<body>
<p>dataset source: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset</p>
<p>
<li>1.PyTorch</li>
<li>2.FastAPI</li>
<li>3.Docker</li>
<p>The </p>
<code>docker build -t mri_classifier .</code>

<p>The api can be accessed using the following command in terminal</p>

<code>curl -X POST "http://0.0.0.0:8000/classify_upload/" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.png"
</code>
<img src="./assets/Screenshot from 2024-10-17 12-48-25.png"></img>

<p><strog>Project Demo</strog></p>

[Demo]("https://github.com/VikhyatChoppa18/Brain_Tumor_Classifier_/blob/00e3eee6db2cefdfb6ce92d32d4002048468dcd3/assets/Screencast%20from%2010-17-2024%2012_59_57%20PM.webm")
</body>


</html>
