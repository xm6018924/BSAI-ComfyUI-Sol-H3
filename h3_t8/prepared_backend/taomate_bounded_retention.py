"""Same native sink/recent selection, bounded host-cache retention allocation.

Selection/metadata comes from the unchanged upstream caller. On allocation or
validation failure the isolated cache is cleared and the original error raised;
partially transformed history must never be reused as a valid continuation.
"""
import torch


def bounded_cache_class(upstream):
    class BoundedRetentionCache(upstream.CleanAVKVCache):
        def __init__(self, contract):
            super().__init__(contract)
            self.retention_receipts = []

        def _retain_commit_rows(self, selection):
            offsets = [0]
            for count in self._commit_token_counts:
                offsets.append(offsets[-1] + count)
            selected_rows, selected_counts, selected_tags = [], [], []
            for block_index, video_only in selection:
                tags = self._commit_token_tags[block_index]
                rows = [i for i, tag in enumerate(tags) if not video_only or tag == upstream.VIDEO_TOKEN_TAG]
                if not rows:
                    continue
                selected_rows.extend(offsets[block_index] + row for row in rows)
                selected_counts.append(len(rows))
                selected_tags.append(tuple(tags[row] for row in rows))
            if not selected_rows:
                self._history.clear()
                self._commit_token_counts.clear()
                self._commit_token_tags.clear()
                return
            before = sum(t.nbytes for pair in self._history.values() for t in (pair.key, pair.value))
            largest_replacement = 0
            try:
                for name in tuple(self._history):
                    pair = self._history.pop(name)
                    indices = torch.tensor(selected_rows, dtype=torch.long, device=pair.key.device)
                    retained = upstream.AVKV(pair.key.index_select(0, indices), pair.value.index_select(0, indices))
                    upstream._validate_av_pair(retained, self.contract, layer_name=name)
                    largest_replacement = max(largest_replacement, retained.key.nbytes + retained.value.nbytes)
                    self._history[name] = retained
                    del pair, retained, indices
                self._commit_token_counts = selected_counts
                self._commit_token_tags = selected_tags
                self.retention_receipts.append({"input_bytes": before,
                    "output_bytes": sum(t.nbytes for p in self._history.values() for t in (p.key, p.value)),
                    "largest_layer_replacement_bytes": largest_replacement,
                    "layers": len(self._history), "rows": len(selected_rows),
                    "policy": "native row selection; replace and release one layer at a time"})
            except BaseException:
                self.clear()
                raise
    return BoundedRetentionCache
