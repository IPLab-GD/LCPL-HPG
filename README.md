For the paper : LCPL-HPG: Local Context-Aware Prompt Learning on Heterogeneous Patent Graphs for Technology Demand Prediction
论文题目：LCPL-HPG：基于异构专利图的局部上下文感知提示学习技术需求预测


在项目根目录依次执行：
python patent_pretrain.py --dataset B64
python patent_pretrain.py --dataset G06
python patent_pretrain.py --dataset H04


预训练输出位于项目根目录：
pretrained_patent_model_b64.pth
pretrained_patent_model_g06.pth
pretrained_patent_model_h04.pth


微调前必须保证对应源域的预训练模型已经存在。例如运行 `--pretrain-from G06` 时，需要先生成 `pretrained_patent_model_g06.pth`。

### 5.2 运行域内微调

域内实验表示预训练来源和目标领域相同。

python patent_finetune.py --dataset B64
python patent_finetune.py --dataset G06
python patent_finetune.py --dataset H04


### 5.3 运行跨域微调

跨域实验通过 `--pretrain-from` 指定源域，通过 `--dataset` 指定目标域。

完整六组跨域实验如下：

python patent_finetune.py --dataset G06 --pretrain-from B64
python patent_finetune.py --dataset H04 --pretrain-from B64

python patent_finetune.py --dataset B64 --pretrain-from G06
python patent_finetune.py --dataset H04 --pretrain-from G06

python patent_finetune.py --dataset B64 --pretrain-from H04
python patent_finetune.py --dataset G06 --pretrain-from H04
