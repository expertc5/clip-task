# CLIP 模型 Zero-Shot 性能评估项目

本项目记录了在本地环境中部署 CLIP 模型，并针对 CIFAR-10 数据集中的特定类别进行零样本分类评估的全过程。

## 1. 结果汇总表 (每类 accuracy、总体 accuracy)

根据测试输出，模型在 3000 张目标图片上的表现如下：

| 类别 (Class) | 样本量 (Samples) | 正确识别数 (Correct) | 准确率 (Accuracy) |
| --- | --- | --- | --- |
| **Cat (猫)** | 1,000 | 878 | 87.8% 

 |
| **Dog (狗)** | 1,000 | 907 | 90.7% 

 |
| **Bird (鸟)** | 1,000 | 932 | 93.2% 

 |
| **总体 (Overall)** | **3,000** | **2,717** | <br>**90.57%** 

 |

---

## 2. 环境配置与仓库准备

在 anaconda prompt 中创建 clip_task 仓库并配置环境：

```bash
# 在anaconda prompt中创建clip_task仓库：
conda create -n clip_task python=3.9 -y

# 激活clip_task仓库 准备下载pytorch：
conda activate clip_task

# 利用镜像网站加速pytorch下载速度：
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

# 下载pytorch(用于训练)和matplotlib(用于最后可视化)库：
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
conda install matplotlib

# 下载transformer pillow （镜像加速）：
pip install transformers pillow notebook -i https://pypi.tuna.tsinghua.edu.cn/simple

# 在huggingface上下载clip的核心权重以及配置文件 将其存放在my_clip文件夹
# 用anaconda prompt启动一个Jupyter用于编译以及训练模型：
jupyter notebook

```

## 3. 模型加载与安全检查绕过

因为 clip 是 2021 年发布的，当时 huggingface 还没有普及 safetensors 格式，而现在的 pytorch 又因为安全原因不能够加载 .bin 文件，所以我编写了一串强制通过的代码：

```python
from transformers import CLIPModel, CLIPProcessor
import torch

local_path = r"D:\my_clip"

try:
    # 核心目标：添加 weights_only=False 绕过安全检查
    model = CLIPModel.from_pretrained(
        local_path,
        torch_dtype=torch.float32,
        weights_only=False #强制通过
    )
    processor = CLIPProcessor.from_pretrained(local_path)
    print("已完成")

```

## 4. 单图推理测试

自己拿了一张图片先试一试这个模型好不好用，测试图片命名为 test.jpg：

```python
from PIL import Image
import torch

try:
    image = Image.open("test.jpg")
    print("读取中")
    text_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    #此处为考核要求，我仅用要求的三种提示词
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1) #因为是自己拿来玩玩我就在softmax归一化这里结束了没有用argmax输出 反正后面看准确率的时候用得到
    print("-" * 30)#浅浅优化一下界面
    for i, text in enumerate(text_prompts):
        percentage = probs[0][i].item() * 100
        print(f"标签: {text:20} | 可能性: {percentage:.2f}%")
    print("-" * 30)#浅浅优化一下界面

except FileNotFoundError:
    print("没找到文件")#防止把测试文件放错地方

```

## 5. 数据集评估 (CIFAR-10)

### 10.开始加载数据集

```python
import torch
from torchvision import datasets, transforms
test_set = datasets.CIFAR10(root='./data', train=False, download=True)
cifar10_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]#想自己玩玩就把10000张都下载下来了
target_mapping = {3: 0, 5: 1, 2: 2} # 按照任务要求只选择bird(2)，cat(3),dog(5)
filtered_images = []
filtered_labels = []
for img, label in test_set:
    if label in target_mapping:
        filtered_images.append(img)
        filtered_labels.append(target_mapping[label] )# 遍历刚才下载的整个数据集
print(f"已获得 {len(filtered_labels)} 张目标图片。")#看看成功没有

```

### 开始训练了

```python
import torch
from tqdm import tqdm
text_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
#严格按照要求只给这三个提示词
correct_counts = {"cat": 0, "dog": 0, "bird": 0}
for i in tqdm(range(len(filtered_images))):
    image = filtered_images[i]
    true_label_idx = filtered_labels[i]#遍历刚才选择的3000张


    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)


    prediction = outputs.logits_per_image.argmax(dim=1).item()#直接归一化出结果了
    if prediction == true_label_idx:
        label_name = ["cat", "dog", "bird"][true_label_idx]
        correct_counts[label_name] += 1#这里是计数
print(correct_counts)#输出最后的结果

```

## 6. 结果可视化

画张图看看最后的训练成果(画图不是很会借助了一下 ai)：

```python
import matplotlib.pyplot as plt
# 1. 整理数据
labels = ['Cat', 'Dog', 'Bird', 'Overall']
# 对应正确数
counts = [878, 907, 932]
total_correct = sum(counts) # 2717
# 计算百分比
accuracies = [
    87.8,         # Cat (878/1000)
    90.7,         # Dog (907/1000)
    93.2,         # Bird (932/1000)
    (total_correct / 3000) * 100  # Overall (90.57%)
]
# 2. 设置颜色：前三个用浅色，总体用金黄色醒目显示
colors = ['#ff9999', '#66b3ff', '#99ff99', '#f39c12']
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', width=0.6)
# 3. 设置横放的左侧标题 (关键改动)
plt.ylabel('Accuracy\n(%)', fontsize=12, fontweight='bold',
           rotation=0, labelpad=45, va='center')
# 4. 在每个柱子上方标注精确到两位的百分比
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height:.2f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')
# 5. 画出任务书要求的 90% 及格红线
plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Task Target (90%)')
# 6. 图表装饰
plt.ylim(0, 115) # 留出顶部空间放数字
plt.title('CLIP Zero-Shot Evaluation: Classes vs. Overall', fontsize=14, pad=20)
plt.legend(loc='upper left')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

```
