"""Ray worker group with FSDP rank0 pinned off the head node (default .31, not .41)."""

from __future__ import annotations

import os
import re
import time

import ray
from ray.util.placement_group import PlacementGroup

from verl.single_controller.ray.base import RayWorkerGroup

from verl.utils.vamverl_env import NCCL_TIMEOUT_MIN, RANK0_NODE_IP, get

# .41 = Ray head + LM Studio → OOM if also FSDP rank0 (full checkpoint load).
_DEFAULT_FSDP_RANK0_IP = "192.168.88.31"


def resolve_rank0_node_ip() -> str:
    return (get(RANK0_NODE_IP) or _DEFAULT_FSDP_RANK0_IP).strip()


def _worker_env_from_runtime() -> dict[str, str]:
    """Propagate cluster env to Ray actors (verl only sets RANK/WORLD_SIZE by default)."""
    keys = (
        "VAMVERL_ROOT",
        "PYTHONPATH",
        "MODEL_PATH",
        "WAN21_DIR",
        "WAN22_DIR",
        "TOKENIZER_PATH",
        "HF_HUB_OFFLINE",
        NCCL_TIMEOUT_MIN,
        RANK0_NODE_IP,
        "HEAD_HOST",
    )
    return {k: get(k) for k in keys if get(k)}


def _pg_bundle_node_id(pg: PlacementGroup, bundle_idx: int = 0) -> str | None:
    table = ray.util.placement_group_table(pg)
    node_ids = table.get("bundles_to_node_id") or []
    if bundle_idx < len(node_ids):
        return node_ids[bundle_idx]
    return None


def _node_id_to_ip(node_id: str) -> str | None:
    for node in ray.nodes():
        if node.get("NodeID") == node_id and node.get("Alive"):
            return node.get("NodeManagerAddress")
    return None


def pg_node_ip(pg: PlacementGroup, bundle_idx: int = 0) -> str | None:
    node_id = _pg_bundle_node_id(pg, bundle_idx)
    if node_id is None:
        return None
    return _node_id_to_ip(node_id)


def _ip_matches(node_ip: str | None, target_ip: str) -> bool:
    if not node_ip:
        return False
    if node_ip == target_ip:
        return True
    head_host = os.environ.get("HEAD_HOST", "spark-0a0b")
    if head_host and node_ip == head_host:
        return True
    # Allow short form like "41" → match 192.168.88.41
    if target_ip.isdigit() and node_ip.endswith(f".{target_ip}"):
        return True
    if target_ip.startswith(".") and node_ip.endswith(target_ip):
        return True
    return False


def reorder_pgs_rank0_first(pgs: list[PlacementGroup], rank0_ip: str) -> list[PlacementGroup]:
    """Put the PG on rank0_ip first so torch dist rank 0 loads checkpoints on that node."""
    head: list[PlacementGroup] = []
    others: list[PlacementGroup] = []
    for pg in pgs:
        ip = pg_node_ip(pg)
        if _ip_matches(ip, rank0_ip):
            head.append(pg)
        else:
            others.append(pg)
    if not head:
        ips = [pg_node_ip(pg) for pg in pgs]
        raise RuntimeError(
            f"No Ray placement group on rank0 node {rank0_ip!r}. "
            f"PG node IPs: {ips}. Check Ray cluster / {RANK0_NODE_IP}."
        )
    ordered = head + others
    print(
        f"[VamVerl] rank0 pinned to {rank0_ip}: PG order → {[pg_node_ip(p) for p in ordered]}",
        flush=True,
    )
    return ordered


class DreamZeroRayWorkerGroup(RayWorkerGroup):
    """RayWorkerGroup that assigns torch dist rank 0 to VAMVERL_RANK0_NODE_IP (default .31)."""

    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
        from ray.util import list_named_actors

        use_gpu = resource_pool.use_gpu
        strategy = "STRICT_PACK" if bin_pack else "PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy)
        pgs = reorder_pgs_rank0_first(pgs, resolve_rank0_node_ip())

        world_size = resource_pool.world_size
        self._world_size = world_size
        num_gpus = 1 / resource_pool.max_collocate_count

        rank = -1
        for pg_idx, local_world_size in enumerate(resource_pool.store):
            pg = pgs[pg_idx]
            assert local_world_size <= pg.bundle_count, (
                f"when generating for {self.name_prefix}, "
                f"local_world_size {local_world_size} > pg.bundle_count {pg.bundle_count}"
            )
            for local_rank in range(local_world_size):
                rank += 1
                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "WG_PREFIX": self.name_prefix,
                    "WG_BACKEND": "ray",
                    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                    "RAY_LOCAL_RANK": str(local_rank),
                }
                env_vars.update(_worker_env_from_runtime())
                if rank != 0:
                    env_vars["MASTER_ADDR"] = self._master_addr
                    env_vars["MASTER_PORT"] = self._master_port

                cia_name = type(ray_cls_with_init.cls).__name__
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)
                cia_name = match.group(1) if match else cia_name
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"

                ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})
                if detached:
                    ray_cls_with_init.update_options({"lifetime": "detached"})

                worker = ray_cls_with_init(
                    placement_group=pg,
                    placement_group_bundle_idx=local_rank,
                    use_gpu=use_gpu,
                    num_gpus=num_gpus,
                )
                self._workers.append(worker)
                self._worker_names.append(name)

                if rank == 0:
                    register_center_actor = None
                    for _ in range(360):
                        if f"{self.name_prefix}_register_center" not in list_named_actors():
                            time.sleep(1)
                        else:
                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
                            break
                    assert register_center_actor is not None, (
                        f"failed to get register_center_actor: {self.name_prefix}_register_center "
                        f"in {list_named_actors(all_namespaces=True)}"
                    )
                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
                    self._master_addr, self._master_port = (
                        rank_zero_info["MASTER_ADDR"],
                        rank_zero_info["MASTER_PORT"],
                    )
