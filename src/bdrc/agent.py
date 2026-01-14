# src/bdrc/agent.py

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.bdrc.policy import BDRCPolicy, PolicyConfig
from src.bdrc.critic import ValueNetwork, CriticConfig
from src.bdrc.buffer import OnPolicyBuffer, TransitionBatch
from src.bdrc.cmdp_spec import BDRCAction
from src.bdrc.config import LagrangeConfig

@dataclass
class AgentConfig:
    state_dim: int = 12
    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    # 新增 PPO 超参数
    ppo_clip_ratio: float = 0.2      
    gae_lambda: float = 0.95         # GAE λ 系数
    ppo_epochs: int = 5              # 每次更新的 PPO epoch 数
    ppo_batch_size: int = 64         # PPO 每个 epoch 的 minibatch 大小
    lagrange: LagrangeConfig = LagrangeConfig()
    disable_violation_advantage: bool = False  # True 则仅用 reward 优势


class BDRCAgent:
    def __init__(
        self,
        cfg: AgentConfig,
        policy_cfg: PolicyConfig,
        critic_cfg: CriticConfig,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        # 只创建一次 policy，并同时记录 policy_type
        self.policy = BDRCPolicy(policy_cfg).to(self.device)
        self.policy_type = policy_cfg.policy_type  # 方便日志/metrics 区分
        # 三个 value network：分别估计 reward、cost1、cost2 的 value
        self.value_r = ValueNetwork(critic_cfg).to(self.device)
        self.value_c1 = ValueNetwork(critic_cfg).to(self.device)
        self.value_c2 = ValueNetwork(critic_cfg).to(self.device)
        
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)
        self.value_opt = optim.Adam(
            list(self.value_r.parameters())
            + list(self.value_c1.parameters())
            + list(self.value_c2.parameters()),
            lr=cfg.lr_value,
        )
        
        # 拉格朗日乘子初始化
        lag_cfg = self.cfg.lagrange
        lambda_c1_init = max(lag_cfg.lambda_init_c1, 1e-6)
        lambda_c2_init = max(lag_cfg.lambda_init_c2, 1e-6)
        self.mu_c1 = torch.log(
            torch.tensor(lambda_c1_init, dtype=torch.float32, device=self.device)
        )
        self.mu_c2 = torch.log(
            torch.tensor(lambda_c2_init, dtype=torch.float32, device=self.device)
        )
        self.lambda_c1 = torch.exp(self.mu_c1).detach()
        self.lambda_c2 = torch.exp(self.mu_c2).detach()
        
        # 成本的滑动平均，用于 "Budget-Tracking Lagrange Update"
        self.cost_c1_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.cost_c2_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def select_action(self, state_vec: np.ndarray, return_log_prob: bool = False):
        """
        基于当前策略采样动作
        返回: (action:BDRCAction, log_prob:float/tensor, value:float/tensor, raw_action_np:np.ndarray)
        """
        device = self.device
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        
        # policy sampling
        action, log_prob_t, raw_action_t = self.policy.sample_action_tensor(state_t)
        
        # value estimate (reward value head)
        with torch.no_grad():
            v = self.value_r(state_t.unsqueeze(0)).squeeze(0).squeeze(-1)
        
        if return_log_prob:
            return action, log_prob_t, v, raw_action_t.detach().cpu().numpy()
        else:
            return action, float(log_prob_t.item()), float(v.item()), raw_action_t.detach().cpu().numpy()

    def train(self) -> None:
        """设置为训练模式"""
        self.policy.train()
        self.value_r.train()
        self.value_c1.train()
        self.value_c2.train()

    def eval(self) -> None:
        """设置为评估模式"""
        self.policy.eval()
        self.value_r.eval()
        self.value_c1.eval()
        self.value_c2.eval()

    def compute_gae_returns(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = rewards.shape[0]
        if dones is None:
            dones = torch.zeros(T, device=rewards.device)
        
        advantages = torch.zeros(T, device=rewards.device)
        returns = torch.zeros(T, device=rewards.device)
        
        gamma = float(self.cfg.gamma)
        gae_lambda = float(self.cfg.gae_lambda)
        
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_not_done = 0.0
            else:
                next_value = values[t + 1]
                next_not_done = 1.0 - dones[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_not_done - values[t]
            gae = delta + gamma * gae_lambda * next_not_done * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages

    def update_with_trajectories(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        costs_c1: torch.Tensor,
        costs_c2: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> dict:
        device = self.device
        
        # 确保数据在正确的设备上
        states = states.to(device).float()
        actions = actions.to(device).float()
        old_log_probs = old_log_probs.to(device).float()
        rewards = rewards.to(device).float()
        costs_c1 = costs_c1.to(device).float()
        costs_c2 = costs_c2.to(device).float()
        
        if dones is not None:
            dones = dones.to(device).float()
        
        # ---- 1) 计算旧的价值估计 ----
        with torch.no_grad():
            V_r_old = self.value_r(states).squeeze(-1)
            V_c1_old = self.value_c1(states).squeeze(-1)
            V_c2_old = self.value_c2(states).squeeze(-1)
            
            # 计算GAE优势
            returns_r, adv_r = self.compute_gae_returns(rewards, V_r_old, dones)
            returns_c1, adv_c1 = self.compute_gae_returns(costs_c1, V_c1_old, dones)
            returns_c2, adv_c2 = self.compute_gae_returns(costs_c2, V_c2_old, dones)
            
            # 拉格朗日组合（只惩罚正的成本优势）；若禁用 violation-only，则仅用 reward 优势
            if self.cfg.disable_violation_advantage or not self.cfg.lagrange.enabled:
                advantages = adv_r
            else:
                lambda_c1 = torch.exp(self.mu_c1).detach()
                lambda_c2 = torch.exp(self.mu_c2).detach()
                advantages = adv_r - lambda_c1 * torch.relu(adv_c1) - lambda_c2 * torch.relu(adv_c2)
            
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 裁剪returns以稳定critic
            r_ret_clip = 10.0
            c_ret_clip = 10.0
            returns_r = torch.clamp(returns_r, -r_ret_clip, r_ret_clip)
            returns_c1 = torch.clamp(returns_c1, 0.0, c_ret_clip)
            returns_c2 = torch.clamp(returns_c2, 0.0, c_ret_clip)
        
        # ---- 2) PPO更新 ----
        clip_ratio = float(self.cfg.ppo_clip_ratio)
        value_clip = float(self.cfg.ppo_clip_ratio)  # 价值裁剪范围同策略裁剪范围
        ent_coef = float(self.cfg.entropy_coef)
        v_coef = float(self.cfg.value_coef)
        
        batch_size = states.size(0)
        minibatch_size = int(self.cfg.ppo_batch_size)
        epochs = int(self.cfg.ppo_epochs)
        
        # 训练统计
        pol_loss_sum = 0.0
        v_loss_sum = 0.0
        ent_sum = 0.0
        kl_sum = 0.0
        clipfrac_sum = 0.0
        n_mb = 0
        
        for _ in range(epochs):
            # 随机打乱数据
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_returns_r = returns_r[idx]
                mb_returns_c1 = returns_c1[idx]
                mb_returns_c2 = returns_c2[idx]
                mb_Vr_old = V_r_old[idx]
                mb_Vc1_old = V_c1_old[idx]
                mb_Vc2_old = V_c2_old[idx]
                
                # ---- 策略损失 ----
                mean, log_std = self.policy(mb_states)
                std = torch.exp(log_std)
                dist = torch.distributions.Independent(
                    torch.distributions.Normal(mean, std), 1
                )
                
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy
                
                # PPO诊断
                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean()
                    clipfrac = (torch.abs(ratio - 1.0) > clip_ratio).float().mean()
                
                # ---- 价值损失 ----
                Vr = self.value_r(mb_states).squeeze(-1)
                Vc1 = self.value_c1(mb_states).squeeze(-1)
                Vc2 = self.value_c2(mb_states).squeeze(-1)
                
                # 带裁剪的价值损失
                Vr_clip = mb_Vr_old + torch.clamp(Vr - mb_Vr_old, -value_clip, value_clip)
                Vc1_clip = mb_Vc1_old + torch.clamp(Vc1 - mb_Vc1_old, -value_clip, value_clip)
                Vc2_clip = mb_Vc2_old + torch.clamp(Vc2 - mb_Vc2_old, -value_clip, value_clip)
                
                vloss_r = 0.5 * torch.max(
                    (Vr - mb_returns_r).pow(2),
                    (Vr_clip - mb_returns_r).pow(2)
                ).mean()
                
                vloss_c1 = 0.5 * torch.max(
                    (Vc1 - mb_returns_c1).pow(2),
                    (Vc1_clip - mb_returns_c1).pow(2)
                ).mean()
                
                vloss_c2 = 0.5 * torch.max(
                    (Vc2 - mb_returns_c2).pow(2),
                    (Vc2_clip - mb_returns_c2).pow(2)
                ).mean()
                
                value_loss = vloss_r + vloss_c1 + vloss_c2
                
                # ---- 总损失和优化 ----
                loss = policy_loss + v_coef * value_loss
                
                self.policy_opt.zero_grad(set_to_none=True)
                self.value_opt.zero_grad(set_to_none=True)
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_r.parameters())
                    + list(self.value_c1.parameters())
                    + list(self.value_c2.parameters()),
                    self.cfg.max_grad_norm,
                )
                
                self.policy_opt.step()
                self.value_opt.step()
                
                # 累积统计
                pol_loss_sum += policy_loss.item()
                v_loss_sum += value_loss.item()
                ent_sum += entropy.item()
                kl_sum += approx_kl.item()
                clipfrac_sum += clipfrac.item()
                n_mb += 1
        
        # ---- 3) 更新拉格朗日乘子 ----
        if self.cfg.lagrange.enabled and self.cfg.lagrange.update_mu:
            with torch.no_grad():
                mean_c1 = costs_c1.mean()
                mean_c2 = costs_c2.mean()
                
                # 预算模式：ema=滑动平均；hard=直接用当前批次均值
                if getattr(self.cfg.lagrange, "budget_mode", "ema") == "hard":
                    self.cost_c1_ema = mean_c1
                    self.cost_c2_ema = mean_c2
                else:
                    beta = float(self.cfg.lagrange.cost_ema_beta)
                    self.cost_c1_ema = (1.0 - beta) * self.cost_c1_ema + beta * mean_c1
                    self.cost_c2_ema = (1.0 - beta) * self.cost_c2_ema + beta * mean_c2
                
                clip_range = float(self.cfg.lagrange.lambda_update_clip)
                grad_mu1 = torch.clamp(
                    self.cost_c1_ema - self.cfg.lagrange.cost_budget_c1,
                    -clip_range, clip_range
                )
                grad_mu2 = torch.clamp(
                    self.cost_c2_ema - self.cfg.lagrange.cost_budget_c2,
                    -clip_range, clip_range
                )
                
                self.mu_c1 += float(self.cfg.lagrange.lambda_lr_c1) * grad_mu1
                self.mu_c2 += float(self.cfg.lagrange.lambda_lr_c2) * grad_mu2
                
                lambda_max = float(self.cfg.lagrange.lambda_max)
                self.lambda_c1 = torch.clamp(torch.exp(self.mu_c1), 0.0, lambda_max)
                self.lambda_c2 = torch.clamp(torch.exp(self.mu_c2), 0.0, lambda_max)
        
        # ---- 4) 返回统计 ----
        denom = max(1, n_mb)
        return {
            "loss_total": (pol_loss_sum / denom) + v_coef * (v_loss_sum / denom),
            "loss_policy": pol_loss_sum / denom,
            "loss_value": v_loss_sum / denom,
            "entropy": ent_sum / denom,
            "approx_kl": kl_sum / denom,
            "clip_frac": clipfrac_sum / denom,
            "returns_mean": returns_r.mean().item(),
            "rewards_sum": rewards.sum().item(),
            "cost_c1_sum": costs_c1.sum().item(),
            "cost_c2_sum": costs_c2.sum().item(),
            "lambda_c1": self.lambda_c1.item(),
            "lambda_c2": self.lambda_c2.item(),
        }

    # 保持与现有代码的兼容性
    def update(self, batch: TransitionBatch) -> dict:
        """兼容原有的update接口"""
        return self.update_with_trajectories(
            states=batch.states,
            actions=batch.actions_raw,
            old_log_probs=batch.log_probs,
            rewards=batch.rewards,
            costs_c1=batch.costs_c1,
            costs_c2=batch.costs_c2,
            dones=batch.dones,
        )

    def state_dict(self) -> dict:
        """获取完整状态字典"""
        return {
            "policy": self.policy.state_dict(),
            "value_r": self.value_r.state_dict(),
            "value_c1": self.value_c1.state_dict(),
            "value_c2": self.value_c2.state_dict(),
            "policy_opt": self.policy_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "lagrange": {
                "mu_c1": float(self.mu_c1.item()),
                "mu_c2": float(self.mu_c2.item()),
                "lambda_c1": float(self.lambda_c1.item()),
                "lambda_c2": float(self.lambda_c2.item()),
                "cost_c1_ema": float(self.cost_c1_ema.item()),
                "cost_c2_ema": float(self.cost_c2_ema.item()),
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """加载状态字典"""
        if "policy" in state:
            self.policy.load_state_dict(state["policy"])
        if "value_r" in state:
            self.value_r.load_state_dict(state["value_r"])
        if "value_c1" in state:
            self.value_c1.load_state_dict(state["value_c1"])
        if "value_c2" in state:
            self.value_c2.load_state_dict(state["value_c2"])
        if "policy_opt" in state:
            self.policy_opt.load_state_dict(state["policy_opt"])
        if "value_opt" in state:
            self.value_opt.load_state_dict(state["value_opt"])
        
        lag = state.get("lagrange", {})
        if "mu_c1" in lag:
            self.mu_c1 = torch.tensor(lag["mu_c1"], dtype=torch.float32, device=self.device)
        if "mu_c2" in lag:
            self.mu_c2 = torch.tensor(lag["mu_c2"], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            self.lambda_c1 = torch.exp(self.mu_c1).detach()
            self.lambda_c2 = torch.exp(self.mu_c2).detach()
        
        if "cost_c1_ema" in lag:
            self.cost_c1_ema = torch.tensor(lag["cost_c1_ema"], dtype=torch.float32, device=self.device)
        if "cost_c2_ema" in lag:
            self.cost_c2_ema = torch.tensor(lag["cost_c2_ema"], dtype=torch.float32, device=self.device)
    
    # 保持向后兼容的方法名
    to_state_dict = state_dict
    load_from_state_dict = load_state_dict