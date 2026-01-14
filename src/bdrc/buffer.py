# src/bdrc/buffer.py

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

@dataclass
class TransitionBatch:
    states: torch.Tensor
    actions_raw: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    costs_c1: torch.Tensor
    costs_c2: torch.Tensor
    dones: torch.Tensor


class OnPolicyBuffer:
    def __init__(self, state_dim: int, device: str = "cpu"):
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.reset()
    
    def reset(self) -> None:
        self._states: List[np.ndarray] = []
        self._actions_raw: List[np.ndarray] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []
        self._rewards: List[float] = []
        self._costs_c1: List[float] = []
        self._costs_c2: List[float] = []
        self._dones: List[bool] = []
        
        # 预分配numpy数组以减少内存碎片
        self._prealloc_size = 1000  # 预分配大小
        self._states_prealloc = np.zeros((self._prealloc_size, self.state_dim), dtype=np.float32)
        self._current_idx = 0
    
    def add(
        self,
        state_vec: np.ndarray,
        action_raw: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        cost_c1: float,
        cost_c2: float,
        done: bool,
    ) -> None:
        """添加过渡到buffer，使用预分配内存"""
        # 如果超出预分配大小，动态扩展
        if self._current_idx >= len(self._states_prealloc):
            # 扩展预分配数组
            new_size = len(self._states_prealloc) * 2
            new_states = np.zeros((new_size, self.state_dim), dtype=np.float32)
            new_states[:len(self._states_prealloc)] = self._states_prealloc
            self._states_prealloc = new_states
        
        # 存储到预分配数组
        self._states_prealloc[self._current_idx] = state_vec
        
        # 仍然存储到列表（用于get_batch）
        self._states.append(state_vec.copy())
        self._actions_raw.append(np.asarray(action_raw, dtype=np.float32))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))
        self._rewards.append(float(reward))
        self._costs_c1.append(float(cost_c1))
        self._costs_c2.append(float(cost_c2))
        self._dones.append(bool(done))
        
        self._current_idx += 1
    
    def __len__(self) -> int:
        return len(self._rewards)
    
    def get_batch(self) -> TransitionBatch:
        """获取batch并直接移动到设备内存"""
        assert len(self) > 0, "buffer为空"
        
        # 批量转换为tensor
        states = torch.from_numpy(np.asarray(self._states, dtype=np.float32)).to(self.device)
        actions_raw = torch.from_numpy(np.asarray(self._actions_raw, dtype=np.float32)).to(self.device)
        log_probs = torch.from_numpy(np.asarray(self._log_probs, dtype=np.float32)).to(self.device)
        values = torch.from_numpy(np.asarray(self._values, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.asarray(self._rewards, dtype=np.float32)).to(self.device)
        costs_c1 = torch.from_numpy(np.asarray(self._costs_c1, dtype=np.float32)).to(self.device)
        costs_c2 = torch.from_numpy(np.asarray(self._costs_c2, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.asarray(self._dones, dtype=np.float32)).to(self.device)

        return TransitionBatch(
            states=states,
            actions_raw=actions_raw,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            costs_c1=costs_c1,
            costs_c2=costs_c2,
            dones=dones,
        )