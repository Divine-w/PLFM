# 基于多模态大语言模型的关系抽取研究
## 安装依赖项
```bash
pip install -r requirements.txt
pip install -e .
```
## 下载数据集和预训练模型
1. MNRE Dataset [link](https://github.com/thecharm/MNRE)
2. METER-CLIP16-RoBERTa (resolution: 224^2) pre-trained on GCC+SBU+COCO+VG [link](https://github.com/zdou0830/METER/releases/download/checkpoint2/meter_clip16_224_roberta_pretrain.ckpt)
## 关系抽取流程
1. 参考METER将MNRE数据集和人工标记样本分别处理成.arrow格式放在data文件夹下
2. 运行run_mmfeat_extract.py提取样本多模态特征
```bash
python run_mmfeat_extract.py meter_clip16_roberta_pretrain
```
3. 运行compute_similarities.py为MNRE数据集中每个样本匹配前三个最相似的人工标记样本
```bash
python compute_similarities.py
```
4. 利用多模态提示模板和多模态大语言模型生成高质量辅助知识
5. 运行run.py利用原始文本和辅助知识训练文本模型并进行评估
```bash
python run.py with task_finetune_mnre_bert full_train
```
## 鸣谢
本代码基于[METER](https://github.com/zdou0830/METER)授权于[Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE)。