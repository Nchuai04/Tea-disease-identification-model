#!/usr/bin/env python
# coding: utf-8

# In[1]:


#hide
#!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
from fastai.data.external import *
from fastai.vision.all import *
from fastai.metrics import accuracy_multi, Precision, Recall, F1Score
import os
from pathlib import Path
import pandas as pd
import cv2  # 导入OpenCV模块
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# In[14]:


# 打印当前工作目录
print("Current Working Directory:", os.getcwd())

# 设置數據路徑
path = Path(r'C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024')

# 加载CSV文件
df = pd.read_csv(path/'train.csv')
print(df.head())

# 定义获取图像和标签的函数
def get_x(r): return path/'Images'/r['fname']
def get_y(r): return r['labels'].split(' ')

# 创建裁剪图像目录
cropped_img_path = path/'CroppedImages'
cropped_img_path.mkdir(parents=True, exist_ok=True)

# 裁剪图像并保存
for idx, row in df.iterrows():
    img_path = get_x(row)
    img = cv2.imread(str(img_path))
    if img is not None:
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        # 打印调试信息
        print(f"Processing {img_path}, bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
        
        if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
            print(f"Invalid bounding box for image {img_path}: ({xmin}, {ymin}, {xmax}, {ymax})")
            continue

        cropped_img = img[ymin:ymax, xmin:xmax]
        
        if cropped_img.size == 0:
            print(f"Empty cropped image for {img_path}, bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
            continue
        
        cropped_img_name = f"{row['fname'].split('.')[0]}_{idx}.jpg"
        cv2.imwrite(str(cropped_img_path/cropped_img_name), cropped_img)
        df.loc[idx, 'cropped_fname'] = cropped_img_name
    else:
        print(f"Image not found: {img_path}")

# 确保 'cropped_fname' 列为字符串类型
df['cropped_fname'] = df['cropped_fname'].astype(str)

# 保存更新后的DataFrame
df.to_csv(path/'train_cropped.csv', index=False)


# In[3]:


import pandas as pd
from pathlib import Path

# 设置數據路徑
path = Path(r'C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024')

# 加载 CSV 文件
df_cropped = pd.read_csv(path/'train_cropped.csv')
df_labels_only = pd.read_csv(path/'trainLablesOnly.csv')

# 添加來源標註
df_cropped['source'] = True
df_labels_only['source'] = False

# 重命名列，避免衝突
df_cropped.rename(columns={'fname': 'original_fname', 'cropped_fname': 'fname'}, inplace=True)

# 選擇並確保列名一致
df_cropped = df_cropped[['fname', 'labels', 'source']]
df_labels_only = df_labels_only[['fname', 'labels', 'source']]

# 合併數據集
df_combined = pd.concat([df_cropped, df_labels_only], ignore_index=True)

# 打印合併後的數據集查看格式
print(df_combined.head())
print(df_combined.tail())
# 保存合併後的數據集
df_combined.to_csv(path/'trainpluscropped.csv', index=False)



# In[4]:


import pandas as pd
from collections import Counter

# 定義所有合法的標籤
all_classes = {"tDC01": "tDC01", "tDE02": "tDE02", "tDE03": "tDE03", "tDM04": "tDM04", "tDP05": "tDP05", "tDP06": "tDP06", "tDS07": "tDS07", "tDC08": "tDC08", "tDL09": "tDL09", "tDM10": "tDM10", "tDD11": "tDD11", "tDA12": "tDA12"}

# 加載標籤數據
df = pd.read_csv('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/trainpluscropped.csv')

# 過濾出合法標籤的行
def is_valid_label(label):
    labels = label.split(' ')
    return all(l in all_classes.keys() for l in labels)

filtered_data = df[df['labels'].apply(is_valid_label)]

# 保存過濾後的數據集
filtered_data.to_csv('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/filtered_trainpluscropped.csv', index=False)

# 提取標籤列
labels = filtered_data['labels']  # 假設標籤列的名稱是 'labels'

# 計算每個類別的樣本數量
label_counts = Counter(labels)

# 打印每個類別的樣本數量
for label, count in label_counts.items():
    print(f"{label}: {count}")

# 假設標籤是以空格分隔的多標籤
all_labels = filtered_data['labels'].str.split(' ')
flattened_labels = [item for sublist in all_labels for item in sublist]
multi_label_counts = Counter(flattened_labels)

# 打印多標籤情況下每個類別的樣本數量
for label, count in multi_label_counts.items():
    print(f"{label}: {count}")



# In[16]:


import matplotlib.pyplot as plt

# 假設你已經有每個類別的樣本數量
class_counts = {
    'tDA12': 83, 'tDC01': 356, 'tDC08': 839, 'tDD11': 71, 'tDE03': 709,'tDE02': 207,
    'tDL09': 394, 'tDM04': 522, 'tDM10': 23, 'tDP06': 328, 'tDS07': 209
}

# 1. 計算各個類別的樣本數量
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# 2. 繪製柱狀圖
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# 3. 計算變異係數（CV）
import numpy as np

counts = np.array(list(class_counts.values()))
mean = np.mean(counts)
std_dev = np.std(counts)
cv = std_dev / mean

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Coefficient of Variation (CV): {cv:.2f}")

# 判斷CV值
if cv < 0.1:
    print("資料集是平衡的")
elif cv < 0.5:
    print("資料集存在輕微的不平衡")
else:
    print("資料集不平衡")


# In[15]:


from fastai.vision.all import *
import pandas as pd
from pathlib import Path

# 設置數據路徑
path = Path(r'C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024')
img_path1 = path / 'Images'
img_path2 = path / 'CroppedImages'

# 加載更新后的CSV文件
df = pd.read_csv(path/'filtered_trainpluscropped.csv')

# 確保 'source' 列是布爾值
df['source'] = df['source'].astype(bool)

# 確保 'fname' 列是字符串類型並過濾 'nan' 值
df['fname'] = df['fname'].astype(str)
df = df[df['fname'].notna() & (df['fname'] != 'nan')]

# 過濾掉無效的文件路徑
def file_exists(r):
    if r['source']:
        return (img_path2 / r['fname']).exists()
    else:
        return (img_path1 / r['fname']).exists()

df = df[df.apply(file_exists, axis=1)]

print(df.head())

# 定义获取图像和标签的函数
def get_x(r):
    if r['source']:
        return img_path2 / r['fname']
    else:
        return img_path1 / r['fname']

def get_y(r):
    return r['labels'].split(' ')

# 創建DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_x,
    get_y=get_y,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(640),  # 調整輸入圖像的大小
    batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]  # 數據增強和標準化
)

# 創建DataLoaders
dls = dblock.dataloaders(df, bs=32)  # bs 是批次大小

# 查看一個批次的數據
dls.show_batch(max_n=9, figsize=(7, 8))




# In[16]:


from fastai.vision.all import *
from sklearn.metrics import precision_score, recall_score, f1_score
from functools import partial
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.callback.schedule import fit_one_cycle

# 定义自定义指标
def precision_multi(inp, targ, thresh=0.5, average='macro'):
    pred = (inp > thresh).float()
    return precision_score(targ.cpu(), pred.cpu(), average=average, zero_division=1)

def recall_multi(inp, targ, thresh=0.5, average='macro'):
    pred = (inp > thresh).float()
    return recall_score(targ.cpu(), pred.cpu(), average=average, zero_division=1)

def f1_score_multi(inp, targ, thresh=0.5, average='macro'):
    pred = (inp > thresh).float()
    return f1_score(targ.cpu(), pred.cpu(), average=average, zero_division=1)


# 创建学习器并添加自定义指标
learn = vision_learner(
    dls, 
    resnet34, 
    metrics=[
        partial(accuracy_multi, thresh=0.8),
        AccumMetric(partial(precision_multi, thresh=0.8), flatten=False),
        AccumMetric(partial(recall_multi, thresh=0.8), flatten=False),
        AccumMetric(partial(f1_score_multi, thresh=0.8), flatten=False)
    ],
    pretrained=True  # 使用预训练的ResNet34模型权重
).to_fp16()

# 使用学习率查找器
learn.lr_find()

# 绘制学习率查找器曲线
learn.recorder.plot_lr_find()

# 获取建议的学习率
suggested_lrs = learn.lr_find()
suggested_lr = suggested_lrs.valley

print(f"Suggested learning rate: {suggested_lr}")

# 增加 epochs，使用早停回調和學習率調度器
learn.fine_tune(6, freeze_epochs=4, base_lr=suggested_lr)


# In[17]:


from fastai.callback.tracker import EarlyStoppingCallback
from fastai.callback.schedule import fit_one_cycle
# 增加 epochs，使用早停回調和學習率調度器
learn.fine_tune(12, freeze_epochs=4, base_lr=1e-3, cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=3)])


# In[18]:


learn.export('exportsu0606v1.pkl')


# In[19]:


from fastai.vision.all import *
import matplotlib.pyplot as plt


# In[20]:


learner = load_learner('exportsu0606v1.pkl')


# In[30]:


from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
import os

# 設置支持中文的字體
font_path = 'C:/Windows/Fonts/msjh.ttc'  # 替換為你的系統中支持中文的字體路徑
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def plot_model_prediction(image_path, learn, diseases_dict,output_folder):
    img = PILImage.create(image_path)
    preds = learn.predict(img)
    predicted_labels = [diseases_dict.get(label, "未知標籤") for label in preds[0].items]
    disease_text = ', '.join(predicted_labels)
    plt.imshow(img)
    plt.title(f"Disease: {disease_text}", fontsize=12, color='blue')
    plt.axis('off')

    # 保存圖像
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
# 疾病對應字典
diseases = {
    'tDC01': '赤葉枯病',
    'tDE02': '網餅病',
    'tDE03': '餅病',
    'tDM04': '髮狀病',
    'tDP05': '褐根病',
    'tDP06': '輪斑病',
    'tDS07': '煤煙病',
    'tDC08': '藻斑病',
    'tDL09': '地衣',
    'tDM10': '苔癬',
    'tDD11': '枝枯病',
    'tDA12': '藻類'
}

# 測試圖片資料夾路徑
test_folder = 'C:/Users/user/Desktop/gradcam/sample photo'
# 指定輸出資料夾
output_folder = 'C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/result'
os.makedirs(output_folder, exist_ok=True)


# 生成預測結果圖像
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    plot_model_prediction(image_path, learn, diseases,output_folder)



# In[31]:


from fastai.vision.all import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from fastai.callback.hook import hook_outputs, Hook, hook_output
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import font_manager
import os

# 設置支持中文的字體
font_path = 'C:/Windows/Fonts/msjh.ttc'  # 替換為你的系統中支持中文的字體路徑
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def preprocess_image(img):
    img_arr = np.array(img)
    img_resize = cv2.resize(img_arr, (224, 224))  # 根據模型的輸入尺寸調整大小
    img_float = np.float32(img_resize) / 255.
    return img_float

def get_combined_gradcam_heatmap(img, learn, target_layers):
    img_float = preprocess_image(img)
    transform = transforms.ToTensor()
    tensor = transform(img_float).unsqueeze(0)

    cams = []
    for layer in target_layers:
        gradcam = GradCAM(model=learn.model, target_layers=[layer])
        grayscale_cam = gradcam(input_tensor=tensor)[0, :, :]
        cams.append(grayscale_cam)

    # 將多個熱力圖平均
    combined_cam = np.mean(cams, axis=0)
    cam_image = show_cam_on_image(img_float, combined_cam, use_rgb=True)
    return cam_image

def plot_gradcam_image(image_path, learn, diseases_dict, target_layers, output_folder):
    img = PILImage.create(image_path)
    heatmap = get_combined_gradcam_heatmap(img, learn, target_layers)
    plt.imshow(heatmap)
    preds = learn.predict(img)
    predicted_labels = [diseases_dict.get(label, "未知標籤") for label in preds[0].items]
    disease_text = ', '.join(predicted_labels)
    plt.title(f"Disease: {disease_text}", fontsize=12, color='blue')
    plt.axis('off')

    # 保存圖像
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# 設定目標層為模型的最後一層卷積層
target_layers = [learn.model[0][7][2].conv2]  # Layer 7 的最後一個 BasicBlock 中的卷積層

# 指定輸出資料夾
output_folder = 'C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/gradcam'
os.makedirs(output_folder, exist_ok=True)

# 使用目標層生成 GradCAM 圖像並保存
test_folder = 'C:/Users/user/Desktop/gradcam/sample photo'
for image_name in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_name)
    plot_gradcam_image(image_path, learn, diseases, target_layers, output_folder)



# In[23]:


import pandas as pd
from collections import Counter

# 定義所有合法的標籤
all_classes = {"tDC01": "tDC01", "tDE02": "tDE02", "tDE03": "tDE03", "tDM04": "tDM04", "tDP05": "tDP05", "tDP06": "tDP06", "tDS07": "tDS07", "tDC08": "tDC08", "tDL09": "tDL09", "tDM10": "tDM10", "tDD11": "tDD11", "tDA12": "tDA12"}

# 加載標籤數據
df = pd.read_csv('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/testLablesOnly.csv')

# 過濾出合法標籤的行
def is_valid_label(label):
    labels = label.split(' ')
    return all(l in all_classes.keys() for l in labels)

filtered_data = df[df['labels'].apply(is_valid_label)]

# 保存過濾後的數據集
filtered_data.to_csv('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/filtered_test.csv', index=False)

# 提取標籤列
labels = filtered_data['labels']  # 假設標籤列的名稱是 'labels'

# 計算每個類別的樣本數量
label_counts = Counter(labels)

# 打印每個類別的樣本數量
for label, count in label_counts.items():
    print(f"{label}: {count}")

# 假設標籤是以空格分隔的多標籤
all_labels = filtered_data['labels'].str.split(' ')
flattened_labels = [item for sublist in all_labels for item in sublist]
multi_label_counts = Counter(flattened_labels)

# 打印多標籤情況下每個類別的樣本數量
for label, count in multi_label_counts.items():
    print(f"{label}: {count}")


# In[24]:


from fastai.vision.all import *
import pandas as pd
from sklearn.metrics import f1_score

# 加載測試數據
test_df = pd.read_csv('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/filtered_test.csv')
path = Path('C:/Users/user/Desktop/Tea disease identification model/pre/InputDataSu/test06032024/Test/')

# 定義測試數據集
test_files = [path/fname for fname in test_df['fname']]
test_dl = learn.dls.test_dl(test_files)

# 運行模型預測
preds, _ = learn.tta(dl=test_dl, n=5)  # 使用 TTA (Test Time Augmentation) 獲取預測

# 將標籤轉換為索引，處理多標籤情況
def labels_to_indices(labels):
    indices = [learn.dls.vocab.o2i[label] for label in labels.split()]
    return indices

true_labels = test_df['labels'].apply(labels_to_indices).values

# 將多標籤轉換為二進制矩陣
num_classes = len(learn.dls.vocab)
true_labels_matrix = np.zeros((len(true_labels), num_classes))

for i, label_list in enumerate(true_labels):
    true_labels_matrix[i, label_list] = 1

pred_labels_matrix = (preds > 0.5).int().numpy()  # 將預測轉換為二進制矩陣

# 計算 F1 分數
f1 = f1_score(true_labels_matrix, pred_labels_matrix, average='macro')
print(f'F1 Score: {f1:.4f}')


# In[ ]:




