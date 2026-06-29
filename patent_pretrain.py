import os
import gc
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd


torch.manual_seed(42)
np.random.seed(42)


class PatentHeteroGraphDataset:
    """预训练数据集：固定使用 split_edges 下的 *_pretrain.tsv。"""

    def __init__(self, node_data_dir, split_edge_dir):
        self.node_data_dir = node_data_dir
        self.split_edge_dir = split_edge_dir
        self.load_data()
        self.build_graph()

    @staticmethod
    def _require_columns(df, cols, file_name):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_name} 缺少必要列: {missing}")

    def load_data(self):
        print("加载节点数据...")
        self.patents = pd.read_csv(os.path.join(self.node_data_dir, "patents_list.tsv"), sep="\t", dtype=str)
        self.assignees = pd.read_csv(os.path.join(self.node_data_dir, "assignees_list.tsv"), sep="\t", dtype=str)
        self.keywords = pd.read_csv(os.path.join(self.node_data_dir, "keywords_list.tsv"), sep="\t", dtype=str)

        self._require_columns(self.patents, ["PatentID"], "patents_list.tsv")
        self._require_columns(self.assignees, ["AssigneeID"], "assignees_list.tsv")
        self._require_columns(self.keywords, ["KeywordID"], "keywords_list.tsv")

        print("加载预训练边数据...")
        self.patent_keyword_edges = pd.read_csv(
            os.path.join(self.split_edge_dir, "patent_keyword_pretrain.tsv"), sep="\t", dtype=str
        )
        self.patent_assignee_edges = pd.read_csv(
            os.path.join(self.split_edge_dir, "patent_assignee_pretrain.tsv"), sep="\t", dtype=str
        )
        self.assignee_keyword_edges = pd.read_csv(
            os.path.join(self.split_edge_dir, "assignee_keyword_pretrain.tsv"), sep="\t", dtype=str
        )

        self._require_columns(self.patent_keyword_edges, ["PatentID", "KeywordID"], "patent_keyword_pretrain.tsv")
        self._require_columns(self.patent_assignee_edges, ["PatentID", "AssigneeID"], "patent_assignee_pretrain.tsv")
        self._require_columns(self.assignee_keyword_edges, ["AssigneeID", "KeywordID"], "assignee_keyword_pretrain.tsv")

        self.patent_to_idx = {pid: i for i, pid in enumerate(self.patents["PatentID"])}
        self.assignee_to_idx = {aid: i for i, aid in enumerate(self.assignees["AssigneeID"])}
        self.keyword_to_idx = {kid: i for i, kid in enumerate(self.keywords["KeywordID"])}

        print("数据统计:")
        print(f"  专利: {len(self.patents)}")
        print(f"  专利权人: {len(self.assignees)}")
        print(f"  关键词: {len(self.keywords)}")
        print(f"  专利-关键词预训练边: {len(self.patent_keyword_edges)}")
        print(f"  专利-专利权人预训练边: {len(self.patent_assignee_edges)}")
        print(f"  专利权人-关键词预训练边: {len(self.assignee_keyword_edges)}")

    def build_graph(self):
        print("构建异质图...")

        valid_patents = set(self.patent_to_idx.keys())
        valid_assignees = set(self.assignee_to_idx.keys())
        valid_keywords = set(self.keyword_to_idx.keys())

        def _filter(df, src_col, dst_col, src_valid, dst_valid):
            mask = df[src_col].isin(src_valid) & df[dst_col].isin(dst_valid)
            out = df[mask].reset_index(drop=True)
            if len(out) < len(df):
                print(f"  警告: {src_col}-{dst_col} 过滤掉 {len(df) - len(out)} 条非法边")
            return out

        self.patent_keyword_edges = _filter(
            self.patent_keyword_edges, "PatentID", "KeywordID", valid_patents, valid_keywords
        )
        self.patent_assignee_edges = _filter(
            self.patent_assignee_edges, "PatentID", "AssigneeID", valid_patents, valid_assignees
        )
        self.assignee_keyword_edges = _filter(
            self.assignee_keyword_edges, "AssigneeID", "KeywordID", valid_assignees, valid_keywords
        )

        graph_data = {
            ("patent", "p-k", "keyword"): (
                torch.tensor([self.patent_to_idx[p] for p in self.patent_keyword_edges["PatentID"]]),
                torch.tensor([self.keyword_to_idx[k] for k in self.patent_keyword_edges["KeywordID"]]),
            ),
            ("keyword", "k-p", "patent"): (
                torch.tensor([self.keyword_to_idx[k] for k in self.patent_keyword_edges["KeywordID"]]),
                torch.tensor([self.patent_to_idx[p] for p in self.patent_keyword_edges["PatentID"]]),
            ),
            ("patent", "p-a", "assignee"): (
                torch.tensor([self.patent_to_idx[p] for p in self.patent_assignee_edges["PatentID"]]),
                torch.tensor([self.assignee_to_idx[a] for a in self.patent_assignee_edges["AssigneeID"]]),
            ),
            ("assignee", "a-p", "patent"): (
                torch.tensor([self.assignee_to_idx[a] for a in self.patent_assignee_edges["AssigneeID"]]),
                torch.tensor([self.patent_to_idx[p] for p in self.patent_assignee_edges["PatentID"]]),
            ),
            ("assignee", "a-k", "keyword"): (
                torch.tensor([self.assignee_to_idx[a] for a in self.assignee_keyword_edges["AssigneeID"]]),
                torch.tensor([self.keyword_to_idx[k] for k in self.assignee_keyword_edges["KeywordID"]]),
            ),
            ("keyword", "k-a", "assignee"): (
                torch.tensor([self.keyword_to_idx[k] for k in self.assignee_keyword_edges["KeywordID"]]),
                torch.tensor([self.assignee_to_idx[a] for a in self.assignee_keyword_edges["AssigneeID"]]),
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

        print(f"图结构: {self.graph}")


def hg_propagate_feat_dgl(g, num_hops, max_length, metapaths_per_type, echo=False):
    all_ntypes = g.ntypes

    for hop in range(1, max_length):
        reserve_heads = {}
        for ntype in all_ntypes:
            ntype_metapaths = [mp for mp in metapaths_per_type if mp.startswith(ntype)]
            reserve_heads[ntype] = [mp[:hop] for mp in ntype_metapaths if len(mp) > hop]

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            for k in list(g.nodes[stype].data.keys()):
                if len(k) != hop:
                    continue

                current_dst_name = f"{dtype}{k}"
                if (hop == num_hops and current_dst_name not in reserve_heads.get(dtype, [])) or (
                    hop > num_hops and current_dst_name not in reserve_heads.get(dtype, [])
                ):
                    continue

                if echo:
                    print(f"传播 {k} 沿 {etype} 到 {current_dst_name}")

                g.update_all(fn.copy_u(k, "m"), fn.mean("m", current_dst_name), etype=etype)

        for ntype in all_ntypes:
            keep_features = [k for k in reserve_heads.get(ntype, [])]
            keep_features.extend([k for k in g.nodes[ntype].data.keys() if len(k) <= 1 or k.startswith(ntype)])
            remove_features = [k for k in g.nodes[ntype].data.keys() if k not in keep_features]
            for k in remove_features:
                del g.nodes[ntype].data[k]
            if echo and remove_features:
                print(f"移除 {ntype} 的特征: {remove_features}")

        gc.collect()
        if echo:
            print(f"--- 完成 hop={hop} 传播 ---")

    return g


class HeteroGNNPretrain(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_metapaths, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.shared_fc = nn.Linear(hidden_dim * num_metapaths, hidden_dim)

        self.patent_fc = nn.Linear(hidden_dim, hidden_dim)
        self.assignee_fc = nn.Linear(hidden_dim, hidden_dim)
        self.keyword_fc = nn.Linear(hidden_dim, hidden_dim)

        self.reconstruct_head = nn.Linear(hidden_dim, in_dim)
        self.edge_pred_head = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.PReLU()

    def _process_features(self, meta_feats):
        feats = torch.stack(list(meta_feats.values()), dim=1)
        bsz, num_mp, in_dim = feats.shape

        x = feats.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).reshape(bsz, -1)
        x = self.shared_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def forward(self, patent_feats, assignee_feats, keyword_feats):
        patent_feat = self._process_features(patent_feats)
        assignee_feat = self._process_features(assignee_feats)
        keyword_feat = self._process_features(keyword_feats)

        patent_repr = self.patent_fc(patent_feat)
        assignee_repr = self.assignee_fc(assignee_feat)
        keyword_repr = self.keyword_fc(keyword_feat)
        return patent_repr, assignee_repr, keyword_repr

    def reconstruct(self, node_repr):
        return self.reconstruct_head(node_repr)

    def predict_edge(self, src_repr, dst_repr):
        return self.edge_pred_head(torch.cat([src_repr, dst_repr], dim=1)).squeeze()


def pretrain(dataset_name="B64"):
    num_hops = 2
    hidden_dim = 256
    epochs = 2000
    lr = 0.001
    mask_ratio = 0.15
    recon_weight = 0.7
    edge_weight = 0.3

    base_dir = Path(__file__).resolve().parent
    domain = dataset_name.upper()
    node_data_dir = base_dir / "patent_data" / domain
    split_edge_dir = node_data_dir / "split_edges"
    save_path = base_dir / f"pretrained_patent_model_{domain.lower()}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"数据集: {domain}")
    print(f"节点目录: {node_data_dir}")
    print(f"预训练边目录: {split_edge_dir}")
    print(f"模型保存: {save_path}")

    dataset = PatentHeteroGraphDataset(node_data_dir=str(node_data_dir), split_edge_dir=str(split_edge_dir))
    g = dataset.graph.clone().to(device)

    metapaths_per_type = ["p", "pk", "pa", "a", "ap", "ak", "k", "kp", "ka"]
    print(f"元路径集合: {metapaths_per_type}")

    g = hg_propagate_feat_dgl(
        g,
        num_hops=num_hops,
        max_length=num_hops + 1,
        metapaths_per_type=metapaths_per_type,
        echo=False,
    )

    patent_feats = {k: v.to(device) for k, v in g.nodes["patent"].data.items() if k.startswith("p")}
    assignee_feats = {k: v.to(device) for k, v in g.nodes["assignee"].data.items() if k.startswith("a")}
    keyword_feats = {k: v.to(device) for k, v in g.nodes["keyword"].data.items() if k.startswith("k")}

    num_metapaths = min(len(patent_feats), len(assignee_feats), len(keyword_feats))
    patent_feats = dict(list(patent_feats.items())[:num_metapaths])
    assignee_feats = dict(list(assignee_feats.items())[:num_metapaths])
    keyword_feats = dict(list(keyword_feats.items())[:num_metapaths])
    print(f"每种节点类型保留元路径数量: {num_metapaths}")

    patent_mask = torch.rand(len(next(iter(patent_feats.values()))), device=device) < mask_ratio
    assignee_mask = torch.rand(len(next(iter(assignee_feats.values()))), device=device) < mask_ratio
    keyword_mask = torch.rand(len(next(iter(keyword_feats.values()))), device=device) < mask_ratio

    edge_types = [("patent", "p-k", "keyword"), ("patent", "p-a", "assignee"), ("assignee", "a-k", "keyword")]
    edge_data = {}
    for etype in edge_types:
        src, dst = g.edges(etype=etype)
        edge_data[etype] = {"src": src.to(device), "dst": dst.to(device)}

    model = HeteroGNNPretrain(in_dim=64, hidden_dim=hidden_dim, num_metapaths=num_metapaths).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    recon_criterion = nn.MSELoss()
    edge_criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    print("开始预训练...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        patent_repr, assignee_repr, keyword_repr = model(patent_feats, assignee_feats, keyword_feats)

        # 先分别计算三类节点平均重建损失，再做整体平均。
        patent_recon = model.reconstruct(patent_repr[patent_mask])
        patent_recon_loss = sum(recon_criterion(patent_recon, feat[patent_mask]) for feat in patent_feats.values())
        patent_recon_loss = patent_recon_loss / max(len(patent_feats), 1)

        assignee_recon = model.reconstruct(assignee_repr[assignee_mask])
        assignee_recon_loss = sum(
            recon_criterion(assignee_recon, feat[assignee_mask]) for feat in assignee_feats.values()
        )
        assignee_recon_loss = assignee_recon_loss / max(len(assignee_feats), 1)

        keyword_recon = model.reconstruct(keyword_repr[keyword_mask])
        keyword_recon_loss = sum(recon_criterion(keyword_recon, feat[keyword_mask]) for feat in keyword_feats.values())
        keyword_recon_loss = keyword_recon_loss / max(len(keyword_feats), 1)

        recon_loss = (patent_recon_loss + assignee_recon_loss + keyword_recon_loss) / 3.0

        edge_loss = 0.0
        for etype in edge_types:
            data = edge_data[etype]
            src, dst = data["src"], data["dst"]

            if etype == ("patent", "p-k", "keyword"):
                pos_scores = model.predict_edge(patent_repr[src], keyword_repr[dst])
            elif etype == ("patent", "p-a", "assignee"):
                pos_scores = model.predict_edge(patent_repr[src], assignee_repr[dst])
            else:
                pos_scores = model.predict_edge(assignee_repr[src], keyword_repr[dst])
            pos_labels = torch.ones_like(pos_scores, device=device)

            neg_samples = max(len(src) // 2, 1)
            if etype == ("patent", "p-k", "keyword"):
                neg_src = torch.randint(0, len(patent_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(keyword_repr), (neg_samples,), device=device)
            elif etype == ("patent", "p-a", "assignee"):
                neg_src = torch.randint(0, len(patent_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(assignee_repr), (neg_samples,), device=device)
            else:
                neg_src = torch.randint(0, len(assignee_repr), (neg_samples,), device=device)
                neg_dst = torch.randint(0, len(keyword_repr), (neg_samples,), device=device)

            if etype == ("patent", "p-k", "keyword"):
                neg_scores = model.predict_edge(patent_repr[neg_src], keyword_repr[neg_dst])
            elif etype == ("patent", "p-a", "assignee"):
                neg_scores = model.predict_edge(patent_repr[neg_src], assignee_repr[neg_dst])
            else:
                neg_scores = model.predict_edge(assignee_repr[neg_src], keyword_repr[neg_dst])
            neg_labels = torch.zeros_like(neg_scores, device=device)

            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])
            edge_loss += edge_criterion(all_scores, all_labels)

        edge_loss = edge_loss / len(edge_types)

        total_loss = recon_weight * recon_loss + edge_weight * edge_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_loss": float(best_loss.item()),
                "dataset": domain,
                "num_hops": num_hops,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "lr": lr,
                "mask_ratio": mask_ratio,
                "recon_weight": recon_weight,
                "edge_weight": edge_weight,
            }
            torch.save(checkpoint, save_path)
            improved = "*"
        else:
            improved = ""

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | 总损失: {total_loss.item():.4f} "
                f"(重建损失: {recon_loss.item():.4f}, 边预测损失: {edge_loss.item():.4f}) {improved}"
            )

        del patent_repr, assignee_repr, keyword_repr
        gc.collect()

    print(f"预训练完成，模型保存至: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="patent pretrain")
    parser.add_argument(
        "--dataset",
        type=str,
        default="B64",
        choices=["B64", "G06", "H04", "b64", "g06", "h04"],
        help="Target dataset domain",
    )
    args = parser.parse_args()
    pretrain(dataset_name=args.dataset.upper())
