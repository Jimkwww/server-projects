把数据集文件放到本目录（code/Data/），命名为：

    RML2016.10a_dict.pkl     （约 611 MB）

校验（务必，保证和服务器 train/test 划分完全一致）：
    md5sum RML2016.10a_dict.pkl
    期望值：61bf35ac7f0b7d8843613453447ea290

获取方式：
  1) 首选：从服务器 scp 拷贝（服务器上现成，无需下载）
     /data1/datas/RML201610a/RML2016.10a_dict.pkl
  2) 备选：DeepSig 官网（现在需邮箱验证，不能匿名直下）：
     浏览器打开
       https://opendata.deepsig.io/datasets/2016.10/RML2016.10a_dict.pkl.tar.bz2
     （或 ..._dict.pkl.zip），填写姓名/单位/邮箱提交，
     DeepSig 会把真实下载链接发到邮箱；务必下 _dict 版本。
     下载后仍须用上面的 md5 校验。

详见包根目录《复现与算力对比指南.md》第 4 节。
