# 线腔识别
####项目数据目录:
<pre>
├─ gland
│ ├─ dataset
│ │ ├─ 1_json
│ ├─ img
│ ├─ json
│ └─ mask
</pre>
dataset包含各个json转出单个文件夹, img是原图文件夹, json是脚本文件夹, mask是对应原图序号的mask图文件夹

####· 7-5
Mask_RCNN可使用灰度图进行训练，但验证集需要使用RGB图像，目前效果最好为5-11训练模型(基于RGB的腺体整体识别模型)
模型验证图中出现重叠，初步预计为anchor设置问题，导致不同尺寸仍被识别

####· 7-8
5-11的训练模型已上传至<a href=http://dreamdarker.top:8000/d/d72652e7f38f4126b0a3/>云</a>, 下载后可在validation.py中修改加载权重的模型路径来验证测试
调整识别区域, 使用腺体周围边界进行识别训练
![gland](\gland.jpg)