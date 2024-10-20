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

[Demo]("./assets/Screencast from 10-17-2024 12:59:57 PM.webm")
</body>


</html>