import os
import argparse
import math
from email.policy import default
from os import replace
import torch
from typing import List, Tuple, Dict, Any
import logging
import numpy as np
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualize import  plot_query_distance_hist

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Movies,Games')
    p.add_argument('--input_path', type=str, default='dataset2/movies-games/')
    p.add_argument('--output_path', type=str, default='dataset2/movies-games/')
    p.add_argument('--suffix', type=str, default='feat1CLS')
    p.add_argument('--plm_size', type=int, default=768)
    p.add_argument('--residual_levels', type=int, default=2, help='number of residual PQ levels (L)')
    p.add_argument('--subvector_num', type=int, default=32, help='PQ M: number of subvectors per level')
    p.add_argument('--ksub', type=int, default=256, help='PQ centroids per subvector (power of 2)')
    p.add_argument('--coarse_clusters', type=int, default=1024, help='IVF coarse clusters (ncoarse)')
    p.add_argument('--nprobe', type=int, default=16, help='lists to probe at search')
    p.add_argument('--use_opq',type=int, default=0, help='use OPQ')
    p.add_argument('--strict', type=int, default=1)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--use_gpu', type=int, default=1, help='(unused) reserved for future accel')
    p.add_argument('--search_demo', action='store_true', help='run a small KNN search after building')
    p.add_argument('--k', type=int, default=10)

    return p.parse_args()


def _check_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _load_strict_filter(input_path: str, short_name: str, dataset_names: List[str]) -> List[np.ndarray]:
    test_path = os.path.join(input_path, short_name, f'{short_name}.test.inter')
    train_path = os.path.join(input_path, short_name, f'{short_name}.train.inter')

    if not (os.path.exists(test_path) and os.path.exists(train_path)):
        logging.info(f"[strict] Missing inter files: {test_path} or {train_path}. Strict mode disabled.")
        return []

    item_set = [set() for _ in range(len(dataset_names))]

    def _ingest(path: str, allow_new: bool):
        print(f"[strict] Loading from [{path}]")
        with open(path, 'r', encoding='utf-8') as f:
            _ = f.readline()  # header
            for line in tqdm(f):
                user_id, item_seq, item_id = line.strip().split('\t')
                did, pure_item_id = item_id.split('-')
                seq = [_.split('-')[-1] for _ in item_seq.split(' ') if _]
                for idx in seq + [pure_item_id]:
                    d = int(did)
                    try:
                        iid_int = int(idx)
                    except ValueError:
                        continue
                    if allow_new or iid_int in item_set[d]:
                        item_set[d].add(iid_int)

    _ingest(test_path, allow_new=True)
    _ingest(train_path, allow_new=False)

    filter_id_list = []
    out_file = os.path.join(input_path, short_name, f'{short_name}.filtered_id')
    with open(out_file, 'w', encoding='utf-8') as f:
        for did in range(len(dataset_names)):
            arr = np.array(sorted(item_set[did]), dtype=np.int64)
            filter_id_list.append(arr)
            for iid in arr.tolist():
                f.write(f'{did}-{iid}\n')
    logging.info(f"[strict] Wrote filtered ids -> {out_file}")
    return filter_id_list


def _load_features(input_path: str, short_name: str, dataset_names: List[str], suffix: str, d: int,
                    filter_ids: List[np.ndarray]):
    feat_list: List[np.ndarray] = []
    mapping: List[Tuple[int, int]] = []

    for did, ds in enumerate(dataset_names):
        feat_path = os.path.join(input_path, short_name, f'{ds}.{suffix}')
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found: {feat_path}")
        loaded = np.fromfile(feat_path, dtype=np.float32)
        if loaded.size % d != 0:
            raise ValueError(f"Feature size {loaded.size} not divisible by dim {d} for {feat_path}")
        loaded = loaded.reshape(-1, d)
        logging.info(f"Load {loaded.shape} from {feat_path}.")

        if filter_ids:
            max_idx = loaded.shape[0] - 1
            valid = filter_ids[did][filter_ids[did] <= max_idx]
            loaded = loaded[valid]
            feat_list.append(loaded)
            mapping.extend([(did, int(i)) for i in valid.tolist()])
        else:
            feat_list.append(loaded)
            mapping.extend([(did, i) for i in range(loaded.shape[0])])

    merged = np.concatenate(feat_list, axis=0)
    return merged.astype(np.float32, copy=False), mapping



class RPQ_Index:

    def __init__(self, d: int, L: int, M: int, ksub: int, ncoarse: int, outpath:str, use_gpu:int, dataset_names=None, mapping=None, feats=None, use_opq:bool=False):
        assert _check_power_of_two(ksub)
        assert d % M == 0
        self.d = d
        self.L = L
        self.M = M
        self.ksub = ksub
        self.nbits = int(math.log2(ksub))
        self.ncoarse = ncoarse

        self.coarse_centroids = None  # (ncoarse, d)
        self.pq_levels: List[faiss.ProductQuantizer] = []

        self.outpath=outpath
        self.use_gpu=use_gpu
        # inverted lists: per coarse list, store ids and per-level codes
        self.invlists: List[Dict[str, Any]] = []
        self._trained = False

        self.use_opq=bool(use_opq)
        self.opq=None

        self.dataset_names = dataset_names
        self.mapping = mapping
        self.feats = feats
        self.gpu_res=None
        if self.use_gpu:
            self.gpu_res = faiss.StandardGpuResources()
    
    def _opq_forward(self, x: np.ndarray) -> np.ndarray:
        if not self.use_opq or self.opq is None:
            return x
        # faiss.OPQMatrix provides apply_py
        if hasattr(self.opq, 'apply_py'):
            return self.opq.apply_py(x)
        # fallback: try matrix attribute
        for attr in ('R', 'A', 'mat', 'matrix'):
            if hasattr(self.opq, attr):
                R = getattr(self.opq, attr)
                return x.dot(R)
        raise RuntimeError('OPQ object does not support apply_py and no known matrix attribute found')

    def _opq_backward(self, x: np.ndarray) -> np.ndarray:
        if not self.use_opq or self.opq is None:
            return x
        # try to use reverse_transform_py if available
        if hasattr(self.opq, 'reverse_transform_py'):
            return self.opq.reverse_transform_py(x)
        # try to use apply_py on transpose (assuming opq.apply_py does x*R)
        if hasattr(self.opq, 'apply_py'):
            # attempt to get matrix and multiply by R.T
            for attr in ('R', 'A', 'mat', 'matrix'):
                if hasattr(self.opq, attr):
                    R = getattr(self.opq, attr)
                    return x.dot(R.T)
        # fallback: try known attrs
        for attr in ('R', 'A', 'mat', 'matrix'):
            if hasattr(self.opq, attr):
                R = getattr(self.opq, attr)
                return x.dot(R.T)
        raise RuntimeError('Cannot invert OPQ transform: no reverse method or matrix found')

    def gpu_matrix_subtract(self, X: np.ndarray, Y: np.ndarray, device: int = 0) -> np.ndarray:
        assert X.shape == Y.shape, "X 和 Y 必须形状相同"
        assert X.dtype == np.float32 and Y.dtype == np.float32, "X 和 Y 必须是 float32 类型"

        res = self.gpu_res

        X_gpu = torch.from_numpy(X).to(f"cuda:{device}")
        Y_gpu = torch.from_numpy(Y).to(f"cuda:{device}")

        Z_gpu = X_gpu - Y_gpu

        Z = Z_gpu.cpu().numpy()
        return Z

    # ------------ training ------------
    def train(self, xb: np.ndarray):
        n, d = xb.shape
        assert d == self.d


        if self.use_opq:
            logging.info(f"[train] Training OPQMatrix (d={self.d}, M={self.M}) ...")
            self.opq = faiss.OPQMatrix(self.d, self.M)
            self.opq.train(xb)
            xb = self._opq_forward(xb)  # rotate before clustering
            logging.info("[train] OPQ training done.")

        logging.info(f"[train] Coarse k-means with {self.ncoarse} clusters ...")
        if self.use_gpu:
            km = faiss.Kmeans(self.d, self.ncoarse, niter=25, verbose=True, spherical=False, gpu=True)
        else:
            km = faiss.Kmeans(self.d, self.ncoarse, niter=25, verbose=True, spherical=False)
        km.train(xb)
        self.coarse_centroids = km.centroids.copy()

        if self.use_gpu:
            indexc = faiss.index_cpu_to_gpu(self.gpu_res, 0, faiss.IndexFlatL2(self.d))
        else:
            indexc = faiss.IndexFlatL2(self.d)
        faiss.omp_set_num_threads(32)
        indexc.add(self.coarse_centroids)
        D, I = indexc.search(xb, 1)
        coarse_labels = I.reshape(-1)

        residuals = xb - self.coarse_centroids[coarse_labels]

        cur_residuals = residuals.astype(np.float32).copy()

        # build invlists structure (empty for now)
        self.invlists = [{
            'ids': np.empty((0,), dtype=np.int64),
            'codes': [np.empty((0, pq.code_size), dtype=np.uint8) for pq in self.pq_levels]
        } for _ in range(self.ncoarse)]

        self._trained = True
        logging.info("[train] done.")

    def add(self, xb: np.ndarray, ids: np.ndarray = None):
        assert self._trained
        n, d = xb.shape
        if ids is None:
            ids = np.arange(sum(len(lst['ids']) for lst in self.invlists),
                            sum(len(lst['ids']) for lst in self.invlists) + n, dtype=np.int64)
        assert ids.shape[0] == n

        # assign to coarse
        if self.use_gpu:
            indexc = faiss.index_cpu_to_gpu(self.gpu_res, 0, faiss.IndexFlatL2(self.d))
        else:
            indexc = faiss.IndexFlatL2(self.d)
        faiss.omp_set_num_threads(32)
        indexc.add(self.coarse_centroids)
        # indexc = faiss.index_gpu_to_cpu(indexc)
        _, I = indexc.search(xb, 1)
        coarse_labels = I.reshape(-1)

        # residuals
        residuals = xb - self.coarse_centroids[coarse_labels]

        cur = residuals.copy().astype(np.float32)
        if self.use_opq and self.opq is not None:
            cur = self._opq_forward(cur)

        level_codes: List[np.ndarray] = []
        for l, pq in enumerate(self.pq_levels):
            codes = pq.compute_codes(cur)  # (n, code_size) uint8
            recon = pq.decode(codes)
            cur = cur - recon
            level_codes.append(codes)

        for list_id in range(self.ncoarse):
            mask = (coarse_labels == list_id)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            lst = self.invlists[list_id]
            lst['ids'] = np.concatenate([lst['ids'], ids[mask]], axis=0)
            for l in range(self.L):
                lst['codes'][l] = np.vstack([lst['codes'][l], level_codes[l][mask]])

    # ------------ search (naive recon) ------------
    def search(self, xq: np.ndarray, k: int = 10, nprobe: int = 16):
        """Return (D, I) using brute-force over nprobe lists with naive reconstruction.
        xq: (nq, d)
        """
        assert self._trained
        nq, d = xq.shape
        # top-nprobe coarse lists per query
        if self.use_gpu:
            indexc = faiss.index_cpu_to_gpu(self.gpu_res, 0, faiss.IndexFlatL2(self.d))
        else:
            indexc = faiss.IndexFlatL2(self.d)
        indexc.add(self.coarse_centroids)
        # indexc = faiss.index_gpu_to_cpu(indexc)
        Dc, Ic = indexc.search(xq, min(nprobe, self.ncoarse))  # (nq, nprobe)

        all_D = np.full((nq, k), np.inf, dtype=np.float32)
        all_I = np.full((nq, k), -1, dtype=np.int64)

        for qi in range(nq):
            q = xq[qi]
            best = []  # list of (dist, id)
            _all_d_in_probe = []
            for li in Ic[qi]:
                lst = self.invlists[int(li)]
                if lst['ids'].size == 0:
                    continue
                # reconstruct all in this list
                recon = np.zeros((lst['ids'].shape[0], self.d), dtype=np.float32)
                # start from coarse centroid
                recon += self.coarse_centroids[int(li)]
                # add per-level decode
                # decode returns vectors in rotated space if OPQ used, so we must inverse-transform
                dec_sum = np.zeros_like(recon)
                for l, pq in enumerate(self.pq_levels):
                    dec = pq.decode(lst['codes'][l])
                    dec_sum += dec
                # if OPQ used, dec_sum is in rotated-space; map back
                if self.use_opq and self.opq is not None:
                    dec_sum = self._opq_backward(dec_sum)
                recon += dec_sum
                # for l, pq in enumerate(self.pq_levels):
                #     dec = pq.decode(lst['codes'][l])
                #     recon += dec

                # distances to q
                diff = recon - q
                dists = np.sum(diff * diff, axis=1)
                _all_d_in_probe.extend(dists.tolist())
                # keep top-k
                if len(best) == 0:
                    idx = np.argpartition(dists, min(k, dists.shape[0]) - 1)[:k]
                    for j in idx:
                        best.append((float(dists[j]), int(lst['ids'][j])))
                else:
                    for j in range(dists.shape[0]):
                        best.append((float(dists[j]), int(lst['ids'][j])))
                # prune
                if len(best) > 10 * k:
                    best = sorted(best)[:10 * k]
            # finalize
            best = sorted(best)[:k]
            if len(best) > 0:
                all_D[qi, :len(best)] = [b[0] for b in best]
                all_I[qi, :len(best)] = [b[1] for b in best]

            if len(_all_d_in_probe) > 0:
                plot_query_distance_hist(_all_d_in_probe, self.outpath+f"search_q{qi}_probe_distance_hist.png")

            valid_k = int(np.sum(np.isfinite(all_D[qi])))
            if valid_k > 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 4))
                plt.bar(range(valid_k), all_D[qi, :valid_k])
                plt.xlabel("Rank")
                plt.ylabel("L2 Distance")
                plt.title(f"Top-{valid_k} Distances for Query {qi}")
                plt.tight_layout()
                plt.savefig(self.outpath+f"search_q{qi}_topk_bar.png", dpi=150)
                plt.close()
        return all_D, all_I

    # ------------ save / load ------------
    def save(self, path: str, idmap: List[Tuple[int, int]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = {
            'd': self.d,
            'L': self.L,
            'M': self.M,
            'ksub': self.ksub,
            'ncoarse': self.ncoarse,
        }
        # pack invlists
        inv = []
        for lst in self.invlists:
            inv.append({
                'ids': lst['ids'],
                'codes': [c for c in lst['codes']],
            })
        # save opq if present
        opq_obj = None
        if self.use_opq and self.opq is not None:
            opq_obj = self.opq
        # np.savez does not handle list of objects well unless allow_pickle
        np.savez(path, meta=meta, coarse=self.coarse_centroids, invlists=inv, opq=opq_obj, allow_pickle=True)
        if idmap is not None:
            idmap_path = os.path.splitext(path)[0] + '.idmap.tsv'
            with open(idmap_path, 'w', encoding='utf-8') as f:
                f.write('faiss_id\tdataset_id\titem_id\n')
                for i, (did, iid) in enumerate(idmap):
                    f.write(f'{i}\t{did}\t{iid}\n')
        logging.info(f"[save] wrote {path}")

    @classmethod
    def load(cls, path: str, outpath: str = "./", use_gpu: int = 0,
             dataset_names=None, feats=None):
        """Load an RPQ_Index from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        data = np.load(path, allow_pickle=True)
        meta = data['meta'].item()
        coarse = data['coarse']
        invlists = data['invlists'].tolist()   # object array -> python list
        opq_obj = None
        if 'opq' in data and data['opq'] is not None:
            try:
                opq_obj = data['opq'].item()
            except Exception:
                opq_obj = data['opq']
        index = cls(
            d=meta['d'],
            L=meta['L'],
            M=meta['M'],
            ksub=meta['ksub'],
            ncoarse=meta['ncoarse'],
            outpath=outpath,
            use_gpu=use_gpu,
            dataset_names=dataset_names,
            mapping=None,
            feats=feats
        )

        index.coarse_centroids = coarse
        index.invlists = invlists
        index.opq = opq_obj
        index._trained = True

        idmap_path = os.path.splitext(path)[0] + ".idmap.tsv"
        if os.path.exists(idmap_path):
            mapping = []
            with open(idmap_path, "r", encoding="utf-8") as f:
                _ = f.readline()
                for line in f:
                    faiss_id, did, iid = line.strip().split("\t")
                    mapping.append((int(did), int(iid)))
            index.mapping = mapping
            logging.info(f"[load] loaded idmap with {len(mapping)} entries from {idmap_path}")

        logging.info(f"[load] loaded RPQ_Index from {path}")
        return index
    idx = RPQ_Index.load("checkpoints/myindex.npz", outpath="./results", use_gpu=1)
    mapped_codes = RPQ_Index.load(
        "checkpoints/myindex.npz",
        outpath="./results",
        use_gpu=0,
        return_codes=True, 
        config=config, 
        field2id_token=field2id_token,
        logger=logger
    )
# ------------------- main CLI -------------------

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,  # log.level
        format='%(asctime)s - %(levelname)s - %(message)s'  # log_format
    )

    dataset_names = args.dataset.split(',')
    short_name = ''.join([_[0] for _ in dataset_names])

    if not _check_power_of_two(args.ksub):
        raise ValueError(f"--ksub must be power of two, got {args.ksub}")
    if args.plm_size % args.subvector_num != 0:
        raise ValueError(f"plm_size ({args.plm_size}) % subvector_num ({args.subvector_num}) != 0")

    filter_ids = []
    if int(args.strict) == 1:
        filter_ids = _load_strict_filter(args.input_path, short_name, dataset_names)

    feats, idmap = _load_features(
        args.input_path, short_name, dataset_names, args.suffix, args.plm_size, filter_ids
    )
    logging.info(f"Merged feature shape: {feats.shape}")
    faiss.normalize_L2(feats)

    ncoarse = min(len(feats) // 100, args.coarse_clusters)
    ncoarse = max(int(ncoarse), 1)
    logging.info(f"Adjusted params: ncoarse={ncoarse}")

    rpq = RPQ_Index(d=args.plm_size,
                       L=int(args.residual_levels),
                       M=int(args.subvector_num),
                       ksub=int(args.ksub),
                       ncoarse=ncoarse,
                       dataset_names=dataset_names,
                       mapping=idmap,
                       feats=feats,
                       outpath=args.output_path,
                       use_gpu=args.use_gpu)
    rpq.train(feats)

    logging.info("[add] encoding and inserting vectors ...")
    ids = np.arange(feats.shape[0], dtype=np.int64)
    rpq.add(feats, ids=ids)

    # save
    out_dir = os.path.join(args.output_path, short_name)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{short_name}.OPQ{args.subvector_num}.RQ{args.residual_levels}-PQ{args.subvector_num}x{args.ksub}-C{ncoarse}-d{args.plm_size}{'.strict' if int(args.strict)==1 else ''}.npz")
    rpq.save(save_path, idmap=idmap)

    # if args.search_demo:
    #     logging.info("[search] running a quick demo ...")
    #     # pick 5 queries
    #     nq = min(5, feats.shape[0])
    #     nq_idx=np.random.choice(feats.shape[0], nq, replace=False)
    #     q=feats[nq_idx]
    #     D, I = rpq.search(q, k=args.k, nprobe=args.nprobe)

    #     logging.info("Top-{} for {} queries:".format(args.k, nq))
    #     for i in range(nq):
    #         logging.info(f'{i}, list(zip({I[i].tolist()}, {np.round(D[i].tolist(), 4)}))')


if __name__ == '__main__':
    main()
