import os
import gc
import argparse
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import pandas as pd
import dgl

from patent_pretrain import hg_propagate_feat_dgl, HeteroGNNPretrain


torch.manual_seed(42)
np.random.seed(42)


FINETUNE_CONFIG = {
    "gpu_id": 0,
    "epochs": 2000,
    "lr": 0.0005,
    "weight_decay": 1e-5,
    "freeze_pretrain": True,
    "hidden_dim": 256,
    "in_dim": 64,
    "threshold_grid_size": 81,
    "focal_alpha": 0.5,
    "focal_gamma": 1.5,
    "task_weight_power": 0.5,
}


class LocalContextPromptLayer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, in_dim))
        self.context_gate = nn.Linear(in_dim * 4, in_dim)
        self.fc = nn.Linear(in_dim * 2, 1)
        self.act = nn.PReLU()

    def forward(self, u_repr, v_repr, u_ctx, v_ctx):
        context_feat = torch.cat([u_repr, v_repr, u_ctx, v_ctx], dim=1)
        gate = torch.sigmoid(self.context_gate(context_feat))
        u_prompted = u_repr + self.prompt + gate * u_ctx
        v_prompted = v_repr + self.prompt + gate * v_ctx
        return self.fc(self.act(torch.cat([u_prompted, v_prompted], dim=1))).squeeze()


class PromptFinetuneModel(nn.Module):
    def __init__(self, pretrain_model, in_dim):
        super().__init__()
        self.pretrain_model = pretrain_model
        self.prompt_pk = LocalContextPromptLayer(in_dim)
        self.prompt_pa = LocalContextPromptLayer(in_dim)
        self.prompt_ak = LocalContextPromptLayer(in_dim)

    def forward(self, patent_feats, assignee_feats, keyword_feats):
        return self.pretrain_model(patent_feats, assignee_feats, keyword_feats)

    def predict(self, patent_repr, assignee_repr, keyword_repr, task_type, src, dst, local_contexts):
        if task_type == "p-k":
            p_ctx, k_ctx = local_contexts[task_type]
            return self.prompt_pk(patent_repr[src], keyword_repr[dst], p_ctx[src], k_ctx[dst])
        if task_type == "p-a":
            p_ctx, a_ctx = local_contexts[task_type]
            return self.prompt_pa(patent_repr[src], assignee_repr[dst], p_ctx[src], a_ctx[dst])
        if task_type == "a-k":
            a_ctx, k_ctx = local_contexts[task_type]
            return self.prompt_ak(assignee_repr[src], keyword_repr[dst], a_ctx[src], k_ctx[dst])
        raise ValueError(f"未知任务类型: {task_type}")


def aggregate_neighbor_mean(num_nodes, src_index, messages):
    out = torch.zeros(num_nodes, messages.size(1), device=messages.device)
    deg = torch.zeros(num_nodes, 1, device=messages.device)
    out.index_add_(0, src_index, messages)
    deg.index_add_(0, src_index, torch.ones(messages.size(0), 1, device=messages.device))
    return out / deg.clamp_min(1.0)


def build_local_contexts(g, patent_repr, assignee_repr, keyword_repr):
    p_from_a_src, p_from_a_dst = g.edges(etype=("patent", "p-a", "assignee"))
    p_from_k_src, p_from_k_dst = g.edges(etype=("patent", "p-k", "keyword"))

    a_from_p_src, a_from_p_dst = g.edges(etype=("assignee", "a-p", "patent"))
    a_from_k_src, a_from_k_dst = g.edges(etype=("assignee", "a-k", "keyword"))

    k_from_p_src, k_from_p_dst = g.edges(etype=("keyword", "k-p", "patent"))
    k_from_a_src, k_from_a_dst = g.edges(etype=("keyword", "k-a", "assignee"))

    patent_ctx_pk = aggregate_neighbor_mean(len(patent_repr), p_from_a_src, assignee_repr[p_from_a_dst])
    keyword_ctx_pk = aggregate_neighbor_mean(len(keyword_repr), k_from_a_src, assignee_repr[k_from_a_dst])

    patent_ctx_pa = aggregate_neighbor_mean(len(patent_repr), p_from_k_src, keyword_repr[p_from_k_dst])
    assignee_ctx_pa = aggregate_neighbor_mean(len(assignee_repr), a_from_k_src, keyword_repr[a_from_k_dst])

    assignee_ctx_ak = aggregate_neighbor_mean(len(assignee_repr), a_from_p_src, patent_repr[a_from_p_dst])
    keyword_ctx_ak = aggregate_neighbor_mean(len(keyword_repr), k_from_p_src, patent_repr[k_from_p_dst])

    return {
        "p-k": (patent_ctx_pk, keyword_ctx_pk),
        "p-a": (patent_ctx_pa, assignee_ctx_pa),
        "a-k": (assignee_ctx_ak, keyword_ctx_ak),
    }


class PatentHeteroGraphDownstreamDataset:
    def __init__(self, node_data_dir, split_edge_dir):
        self.node_data_dir = node_data_dir
        self.split_edge_dir = split_edge_dir
        self.load_node_data()
        self.load_downstream_edges()
        self.build_graph()

    @staticmethod
    def _require_columns(df, cols, file_name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_name} 缺少必要列: {missing}")

    def load_node_data(self):
        print("加载节点数据（与预训练一致）...")
        self.patents = pd.read_csv(os.path.join(self.node_data_dir, "patents_list.tsv"), sep="\t", dtype=str)
        self.assignees = pd.read_csv(os.path.join(self.node_data_dir, "assignees_list.tsv"), sep="\t", dtype=str)
        self.keywords = pd.read_csv(os.path.join(self.node_data_dir, "keywords_list.tsv"), sep="\t", dtype=str)

        self._require_columns(self.patents, ["PatentID"], "patents_list.tsv")
        self._require_columns(self.assignees, ["AssigneeID"], "assignees_list.tsv")
        self._require_columns(self.keywords, ["KeywordID"], "keywords_list.tsv")

        self.patent_to_idx = {pid: i for i, pid in enumerate(self.patents["PatentID"])}
        self.assignee_to_idx = {aid: i for i, aid in enumerate(self.assignees["AssigneeID"])}
        self.keyword_to_idx = {kid: i for i, kid in enumerate(self.keywords["KeywordID"])}

        print(f"节点统计: 专利{len(self.patents)}, 专利权人{len(self.assignees)}, 关键词{len(self.keywords)}")

    def load_downstream_edges(self):
        print("加载划分后的下游边数据...")
        self.patent_keyword_train = pd.read_csv(os.path.join(self.split_edge_dir, "patent_keyword_train.tsv"), sep="\t", dtype=str)
        self.patent_assignee_train = pd.read_csv(os.path.join(self.split_edge_dir, "patent_assignee_train.tsv"), sep="\t", dtype=str)
        self.assignee_keyword_train = pd.read_csv(os.path.join(self.split_edge_dir, "assignee_keyword_train.tsv"), sep="\t", dtype=str)

        self.patent_keyword_val = pd.read_csv(os.path.join(self.split_edge_dir, "patent_keyword_val.tsv"), sep="\t", dtype=str)
        self.patent_assignee_val = pd.read_csv(os.path.join(self.split_edge_dir, "patent_assignee_val.tsv"), sep="\t", dtype=str)
        self.assignee_keyword_val = pd.read_csv(os.path.join(self.split_edge_dir, "assignee_keyword_val.tsv"), sep="\t", dtype=str)

        self.patent_keyword_test = pd.read_csv(os.path.join(self.split_edge_dir, "patent_keyword_test.tsv"), sep="\t", dtype=str)
        self.patent_assignee_test = pd.read_csv(os.path.join(self.split_edge_dir, "patent_assignee_test.tsv"), sep="\t", dtype=str)
        self.assignee_keyword_test = pd.read_csv(os.path.join(self.split_edge_dir, "assignee_keyword_test.tsv"), sep="\t", dtype=str)

        self._require_columns(self.patent_keyword_train, ["PatentID", "KeywordID"], "patent_keyword_train.tsv")
        self._require_columns(self.patent_assignee_train, ["PatentID", "AssigneeID"], "patent_assignee_train.tsv")
        self._require_columns(self.assignee_keyword_train, ["AssigneeID", "KeywordID"], "assignee_keyword_train.tsv")

    def build_graph(self):
        print("构建下游图:")

        valid_p = set(self.patent_to_idx.keys())
        valid_a = set(self.assignee_to_idx.keys())
        valid_k = set(self.keyword_to_idx.keys())

        def _filter(df, src_col, dst_col, src_valid, dst_valid):
            mask = df[src_col].isin(src_valid) & df[dst_col].isin(dst_valid)
            out = df[mask]
            if len(out) < len(df):
                print(f"  警告: {src_col}-{dst_col} 过滤了 {len(df) - len(out)} 条边")
            return out.reset_index(drop=True)

        self.patent_keyword_train = _filter(self.patent_keyword_train, "PatentID", "KeywordID", valid_p, valid_k)
        self.patent_keyword_val = _filter(self.patent_keyword_val, "PatentID", "KeywordID", valid_p, valid_k)
        self.patent_keyword_test = _filter(self.patent_keyword_test, "PatentID", "KeywordID", valid_p, valid_k)

        self.patent_assignee_train = _filter(self.patent_assignee_train, "PatentID", "AssigneeID", valid_p, valid_a)
        self.patent_assignee_val = _filter(self.patent_assignee_val, "PatentID", "AssigneeID", valid_p, valid_a)
        self.patent_assignee_test = _filter(self.patent_assignee_test, "PatentID", "AssigneeID", valid_p, valid_a)

        self.assignee_keyword_train = _filter(self.assignee_keyword_train, "AssigneeID", "KeywordID", valid_a, valid_k)
        self.assignee_keyword_val = _filter(self.assignee_keyword_val, "AssigneeID", "KeywordID", valid_a, valid_k)
        self.assignee_keyword_test = _filter(self.assignee_keyword_test, "AssigneeID", "KeywordID", valid_a, valid_k)

        graph_data = {
            ("patent", "p-k", "keyword"): (
                torch.tensor([self.patent_to_idx[p] for p in self.patent_keyword_train["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_keyword_val["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_keyword_test["PatentID"]]),
                torch.tensor([self.keyword_to_idx[k] for k in self.patent_keyword_train["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.patent_keyword_val["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.patent_keyword_test["KeywordID"]]),
            ),
            ("keyword", "k-p", "patent"): (
                torch.tensor([self.keyword_to_idx[k] for k in self.patent_keyword_train["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.patent_keyword_val["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.patent_keyword_test["KeywordID"]]),
                torch.tensor([self.patent_to_idx[p] for p in self.patent_keyword_train["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_keyword_val["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_keyword_test["PatentID"]]),
            ),
            ("patent", "p-a", "assignee"): (
                torch.tensor([self.patent_to_idx[p] for p in self.patent_assignee_train["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_assignee_val["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_assignee_test["PatentID"]]),
                torch.tensor([self.assignee_to_idx[a] for a in self.patent_assignee_train["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.patent_assignee_val["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.patent_assignee_test["AssigneeID"]]),
            ),
            ("assignee", "a-p", "patent"): (
                torch.tensor([self.assignee_to_idx[a] for a in self.patent_assignee_train["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.patent_assignee_val["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.patent_assignee_test["AssigneeID"]]),
                torch.tensor([self.patent_to_idx[p] for p in self.patent_assignee_train["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_assignee_val["PatentID"]] +
                             [self.patent_to_idx[p] for p in self.patent_assignee_test["PatentID"]]),
            ),
            ("assignee", "a-k", "keyword"): (
                torch.tensor([self.assignee_to_idx[a] for a in self.assignee_keyword_train["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.assignee_keyword_val["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.assignee_keyword_test["AssigneeID"]]),
                torch.tensor([self.keyword_to_idx[k] for k in self.assignee_keyword_train["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.assignee_keyword_val["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.assignee_keyword_test["KeywordID"]]),
            ),
            ("keyword", "k-a", "assignee"): (
                torch.tensor([self.keyword_to_idx[k] for k in self.assignee_keyword_train["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.assignee_keyword_val["KeywordID"]] +
                             [self.keyword_to_idx[k] for k in self.assignee_keyword_test["KeywordID"]]),
                torch.tensor([self.assignee_to_idx[a] for a in self.assignee_keyword_train["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.assignee_keyword_val["AssigneeID"]] +
                             [self.assignee_to_idx[a] for a in self.assignee_keyword_test["AssigneeID"]]),
            ),
        }

        self.graph = dgl.heterograph(
            graph_data,
            num_nodes_dict={
                "patent": len(self.patents),
                "assignee": len(self.assignees),
                "keyword": len(self.keywords),
            },
        )

        self.graph.nodes["patent"].data["feat"] = torch.randn(len(self.patents), 64)
        self.graph.nodes["assignee"].data["feat"] = torch.randn(len(self.assignees), 64)
        self.graph.nodes["keyword"].data["feat"] = torch.randn(len(self.keywords), 64)

        self.graph.nodes["patent"].data["p"] = self.graph.nodes["patent"].data["feat"].clone()
        self.graph.nodes["assignee"].data["a"] = self.graph.nodes["assignee"].data["feat"].clone()
        self.graph.nodes["keyword"].data["k"] = self.graph.nodes["keyword"].data["feat"].clone()

    def get_task_edges(self, task_type, split_type):
        if task_type == "p-k":
            edges = self.patent_keyword_train if split_type == "train" else self.patent_keyword_val if split_type == "val" else self.patent_keyword_test
            src_col, dst_col = "PatentID", "KeywordID"
            src_map, dst_map = self.patent_to_idx, self.keyword_to_idx
        elif task_type == "p-a":
            edges = self.patent_assignee_train if split_type == "train" else self.patent_assignee_val if split_type == "val" else self.patent_assignee_test
            src_col, dst_col = "PatentID", "AssigneeID"
            src_map, dst_map = self.patent_to_idx, self.assignee_to_idx
        elif task_type == "a-k":
            edges = self.assignee_keyword_train if split_type == "train" else self.assignee_keyword_val if split_type == "val" else self.assignee_keyword_test
            src_col, dst_col = "AssigneeID", "KeywordID"
            src_map, dst_map = self.assignee_to_idx, self.keyword_to_idx
        else:
            raise ValueError(f"未知任务类型: {task_type}")

        src = [src_map[edges.iloc[i][src_col]] for i in range(len(edges))]
        dst = [dst_map[edges.iloc[i][dst_col]] for i in range(len(edges))]
        return torch.tensor(src), torch.tensor(dst)


def find_best_threshold(probs, labels, grid_size=81):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, grid_size):
        y_pred = (probs >= threshold).astype(int)
        cur_f1 = f1_score(labels, y_pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(threshold)
    return best_threshold


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * bce).mean()


def build_task_loss_weights(task_data, power=0.5):
    raw = {}
    for task_type, splits in task_data.items():
        train_size = max(len(splits["train"]["src"]), 1)
        raw[task_type] = 1.0 / (train_size ** power)
    mean_w = sum(raw.values()) / max(len(raw), 1)
    return {k: v / mean_w for k, v in raw.items()}


def _safe_load_state_dict(ckpt_path, map_location):
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj
    return obj, None


def finetune(dataset_name="B64", pretrain_from=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    domain = dataset_name.upper()
    source = pretrain_from.upper() if pretrain_from else domain

    config = dict(FINETUNE_CONFIG)
    config["node_data_dir"] = os.path.join(base_dir, "patent_data", domain)
    config["split_edge_dir"] = os.path.join(base_dir, "patent_data", domain, "split_edges")
    config["pretrain_model_path"] = os.path.join(base_dir, f"pretrained_patent_model_{source.lower()}.pth")
    config["save_dir"] = (
        os.path.join(base_dir, f"finetune_results_{domain.lower()}")
        if source == domain
        else os.path.join(base_dir, f"finetune_results_{domain.lower()}_from_{source.lower()}")
    )

    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["save_dir"], exist_ok=True)

    logger = logging.getLogger(f"finetune_{domain}_{source}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = logging.FileHandler(os.path.join(config["save_dir"], f"finetune_{ts}.log"), encoding="utf-8")
    logfile.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logger.addHandler(logfile)
    logger.addHandler(logging.StreamHandler())

    logger.info("=" * 60)
    logger.info(f"微调启动: target={domain}, pretrain={source}")
    logger.info(f"node_data_dir={config['node_data_dir']}")
    logger.info(f"split_edge_dir={config['split_edge_dir']}")
    logger.info(f"pretrain_model_path={config['pretrain_model_path']}")
    logger.info(f"save_dir={config['save_dir']}")
    logger.info("=" * 60)

    dataset = PatentHeteroGraphDownstreamDataset(config["node_data_dir"], config["split_edge_dir"])
    g = dataset.graph.to(device)

    metapaths_per_type = [
        "p", "pk", "pa", "pkp", "pap", "pak",
        "a", "ap", "ak", "apa", "aka", "akp",
        "k", "kp", "ka", "kpk", "kak", "kap",
    ]
    g = hg_propagate_feat_dgl(g, num_hops=2, max_length=3, metapaths_per_type=metapaths_per_type, echo=False)

    patent_feats = {k: v.to(device) for k, v in g.nodes["patent"].data.items() if k.startswith("p")}
    assignee_feats = {k: v.to(device) for k, v in g.nodes["assignee"].data.items() if k.startswith("a")}
    keyword_feats = {k: v.to(device) for k, v in g.nodes["keyword"].data.items() if k.startswith("k")}

    num_metapaths = min(len(patent_feats), len(assignee_feats), len(keyword_feats))
    patent_feats = dict(list(patent_feats.items())[:num_metapaths])
    assignee_feats = dict(list(assignee_feats.items())[:num_metapaths])
    keyword_feats = dict(list(keyword_feats.items())[:num_metapaths])
    logger.info(f"元路径数量: {num_metapaths}")

    pretrain_model = HeteroGNNPretrain(
        in_dim=config["in_dim"],
        hidden_dim=config["hidden_dim"],
        num_metapaths=num_metapaths,
    ).to(device)

    pretrain_state, _ = _safe_load_state_dict(config["pretrain_model_path"], map_location=device)
    pretrain_model.load_state_dict(pretrain_state)
    logger.info("预训练模型加载完成")

    if config["freeze_pretrain"]:
        for p in pretrain_model.parameters():
            p.requires_grad = False
        logger.info("已冻结预训练主干，仅训练 Prompt 层")

    model = PromptFinetuneModel(pretrain_model, config["hidden_dim"]).to(device)
    logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    tasks = ["p-k", "p-a", "a-k"]
    task_data = {}
    for task_type in tasks:
        train_src, train_dst = dataset.get_task_edges(task_type, "train")
        val_src, val_dst = dataset.get_task_edges(task_type, "val")
        test_src, test_dst = dataset.get_task_edges(task_type, "test")
        task_data[task_type] = {
            "train": {"src": train_src.to(device), "dst": train_dst.to(device)},
            "val": {"src": val_src.to(device), "dst": val_dst.to(device)},
            "test": {"src": test_src.to(device), "dst": test_dst.to(device)},
        }
        logger.info(f"任务 {task_type}: train={len(train_src)} val={len(val_src)} test={len(test_src)}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = BinaryFocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
    task_loss_weights = build_task_loss_weights(task_data, power=config["task_weight_power"])
    logger.info(
        "Task loss weights: " + ", ".join(f"{k}={task_loss_weights[k]:.3f}" for k in tasks)
    )

    best_val_auc = {t: 0.0 for t in tasks}
    best_thresholds = {t: 0.5 for t in tasks}

    logger.info("开始微调...")
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        patent_repr, assignee_repr, keyword_repr = model(patent_feats, assignee_feats, keyword_feats)
        local_contexts = build_local_contexts(g, patent_repr, assignee_repr, keyword_repr)

        for task_type in tasks:
            data = task_data[task_type]["train"]
            src, dst = data["src"], data["dst"]
            if len(src) == 0:
                continue

            pos_score = model.predict(patent_repr, assignee_repr, keyword_repr, task_type, src, dst, local_contexts)
            pos_label = torch.ones_like(pos_score, device=device)

            neg_samples = len(src)
            if task_type == "p-k":
                neg_src = torch.randint(0, len(patent_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(keyword_repr), (neg_samples,), device=device)
            elif task_type == "p-a":
                neg_src = torch.randint(0, len(patent_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(assignee_repr), (neg_samples,), device=device)
            else:
                neg_src = torch.randint(0, len(assignee_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(keyword_repr), (neg_samples,), device=device)

            neg_score = model.predict(
                patent_repr, assignee_repr, keyword_repr, task_type, neg_src, neg_dst, local_contexts
            )
            neg_label = torch.zeros_like(neg_score, device=device)

            all_scores = torch.cat([pos_score, neg_score])
            all_labels = torch.cat([pos_label, neg_label])
            task_loss = criterion(all_scores, all_labels) * task_loss_weights[task_type]
            total_loss += task_loss

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                p_val, a_val, k_val = model(patent_feats, assignee_feats, keyword_feats)
                val_contexts = build_local_contexts(g, p_val, a_val, k_val)
                val_metrics = {}

                for task_type in tasks:
                    data = task_data[task_type]["val"]
                    src, dst = data["src"], data["dst"]
                    if len(src) == 0:
                        continue

                    pos_score = model.predict(p_val, a_val, k_val, task_type, src, dst, val_contexts)
                    pos_label = torch.ones_like(pos_score)

                    neg_samples = len(src)
                    if task_type == "p-k":
                        neg_src = torch.randint(0, len(p_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(k_val), (neg_samples,), device=device)
                    elif task_type == "p-a":
                        neg_src = torch.randint(0, len(p_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(a_val), (neg_samples,), device=device)
                    else:
                        neg_src = torch.randint(0, len(a_val), (neg_samples,), device=device)
                        neg_dst = torch.randint(0, len(k_val), (neg_samples,), device=device)

                    neg_score = model.predict(p_val, a_val, k_val, task_type, neg_src, neg_dst, val_contexts)
                    neg_label = torch.zeros_like(neg_score)

                    all_scores = torch.cat([pos_score, neg_score]).cpu().numpy()
                    all_labels = torch.cat([pos_label, neg_label]).cpu().numpy()
                    unique_labels = np.unique(all_labels)

                    if len(unique_labels) <= 1:
                        auc = 0.5
                        aupr = 0.5
                        precision = 0.5
                        recall = 0.5
                        f1 = 0.5
                        best_threshold = 0.5
                    else:
                        probs = 1.0 / (1.0 + np.exp(-all_scores))
                        auc = roc_auc_score(all_labels, probs)
                        aupr = average_precision_score(all_labels, probs)
                        best_threshold = find_best_threshold(
                            probs, all_labels, grid_size=config["threshold_grid_size"]
                        )
                        y_pred = (probs >= best_threshold).astype(int)
                        precision = precision_score(all_labels, y_pred, zero_division=0)
                        recall = recall_score(all_labels, y_pred, zero_division=0)
                        f1 = f1_score(all_labels, y_pred, zero_division=0)

                    val_metrics[task_type] = (auc, aupr, precision, recall, f1)

                    if auc > best_val_auc[task_type]:
                        best_val_auc[task_type] = auc
                        best_thresholds[task_type] = best_threshold
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "threshold": float(best_threshold),
                                "best_val_auc": float(auc),
                                "task_type": task_type,
                            },
                            os.path.join(config["save_dir"], f"best_{task_type}.pt"),
                        )

            logger.info(f"Epoch {epoch + 1}/{config['epochs']} | 总损失: {total_loss.item():.4f}")
            for task_type, (auc, aupr, precision, recall, f1) in val_metrics.items():
                logger.info(
                    f"  任务 {task_type} - AUC: {auc:.4f}, AUPR: {aupr:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
                    f"BestAUC: {best_val_auc[task_type]:.4f}, Threshold: {best_thresholds[task_type]:.3f}"
                )

        del patent_repr, assignee_repr, keyword_repr
        gc.collect()

    logger.info("\n开始测试最佳模型...")
    for task_type in tasks:
        best_path = os.path.join(config["save_dir"], f"best_{task_type}.pt")
        state, info = _safe_load_state_dict(best_path, map_location=device)
        model.load_state_dict(state)
        threshold = 0.5 if info is None else float(info.get("threshold", 0.5))

        model.eval()
        with torch.no_grad():
            p_test, a_test, k_test = model(patent_feats, assignee_feats, keyword_feats)
            test_contexts = build_local_contexts(g, p_test, a_test, k_test)
            data = task_data[task_type]["test"]
            src, dst = data["src"], data["dst"]
            if len(src) == 0:
                continue

            pos_score = model.predict(p_test, a_test, k_test, task_type, src, dst, test_contexts)
            pos_label = torch.ones_like(pos_score)

            neg_samples = len(src)
            if task_type == "p-k":
                neg_src = torch.randint(0, len(p_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(k_test), (neg_samples,), device=device)
            elif task_type == "p-a":
                neg_src = torch.randint(0, len(p_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(a_test), (neg_samples,), device=device)
            else:
                neg_src = torch.randint(0, len(a_test), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(k_test), (neg_samples,), device=device)

            neg_score = model.predict(p_test, a_test, k_test, task_type, neg_src, neg_dst, test_contexts)
            neg_label = torch.zeros_like(neg_score)

            all_scores = torch.cat([pos_score, neg_score]).cpu().numpy()
            all_labels = torch.cat([pos_label, neg_label]).cpu().numpy()
            unique_labels = np.unique(all_labels)

            if len(unique_labels) <= 1:
                test_auc = 0.5
                test_aupr = 0.5
                test_precision = 0.5
                test_recall = 0.5
                test_f1 = 0.5
            else:
                probs = 1.0 / (1.0 + np.exp(-all_scores))
                test_auc = roc_auc_score(all_labels, probs)
                test_aupr = average_precision_score(all_labels, probs)
                y_pred = (probs >= threshold).astype(int)
                test_precision = precision_score(all_labels, y_pred, zero_division=0)
                test_recall = recall_score(all_labels, y_pred, zero_division=0)
                test_f1 = f1_score(all_labels, y_pred, zero_division=0)

            logger.info(
                f"任务 {task_type} 测试结果 - AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}, "
                f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, "
                f"Threshold: {threshold:.3f}"
            )

    logger.info("微调完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="patent finetune")
    parser.add_argument(
        "--dataset",
        type=str,
        default="B64",
        choices=["B64", "G06", "H04", "b64", "g06", "h04"],
        help="Target dataset domain",
    )
    parser.add_argument(
        "--pretrain-from",
        type=str,
        default=None,
        choices=["B64", "G06", "H04", "b64", "g06", "h04"],
        help="Source pretrain domain. Omit for in-domain finetuning.",
    )
    args = parser.parse_args()
    source = None if args.pretrain_from is None else args.pretrain_from.upper()
    finetune(dataset_name=args.dataset.upper(), pretrain_from=source)
