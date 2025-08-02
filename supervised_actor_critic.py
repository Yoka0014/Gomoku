"""
訓練データの着手と勝敗から, Actor-Criticの学習則で方策と価値を教師あり学習するためのスクリプト
"""
import multiprocessing as mp

import torch
from torch import nn
from torch.utils.data import DataLoader

from gomoku import Position
from dataset import PositionDataset
from dual_net import DualNet


OUTCOME_WIN = 1
OUTCOME_LOSS = 0
OUTCOME_DRAW = 0.5


class SupervisedActorCriticConfig:
    def __init__(self):
        self.initial_model_path = "params/DQN/dqn_model_39999.pth"
        self.train_dataset_path = "data/train_data_qnet.txt"
        self.test_dataset_path = "data/test_data_qnet.txt"

        self.model_out_path = "dualnet.pth"
        self.train_loss_history_path = "train_loss_history.txt"
        self.test_loss_history_path = "test_loss_history.txt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_workers = 8

        self.board_size = 9
        self.learning_rate = 0.001
        self.l2_penalty = 1.0e-4

        self.batch_size = 4096

        self.max_epoch = 1000

        # テスト損失がpatience回連続して改善しない場合に学習を打ち切る.
        self.patience = 10

        # 方策ヘッドと価値ヘッドのみの学習を行う場合はTrueにする.
        self.transfer_learning = False

        assert(self.initial_model_path is not None or not self.transfer_learning)

    def init_pos(self):
        """
        初期局面を生成する.
        デフォルトはPositionクラスのコンストラクタ.

        この関数を書き換えることで任意局面を初期局面として設定できる.
        """
        return Position(self.board_size, nn_input=True)
    

def loss_func(model: DualNet, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float, float]:
    """
    モデルの損失を計算する関数.
    """
    pos_tensor, move_tensor, outcome_tensor = batch

    p, v_logit = model(pos_tensor)
    v = nn.functional.sigmoid(v_logit).detach()

    value_loss = nn.functional.binary_cross_entropy_with_logits(v_logit.squeeze(1), outcome_tensor)

    log_p = nn.functional.log_softmax(p, dim=1)
    nll = nn.functional.nll_loss(log_p, move_tensor, reduction='none')
    advantage = outcome_tensor - v.squeeze(1) + 0.5
    policy_loss = (nll * advantage).mean()

    policy_entropy = -torch.sum(log_p * torch.exp(log_p), dim=1).mean()

    return policy_loss, value_loss, policy_entropy
    

def model_step(model: DualNet, optimizer, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float]:
    """
    モデルの1ステップ分の更新を実行し, 方策と価値の損失を返す.
    """
    policy_loss, value_loss, policy_entropy = loss_func(model, batch)
    loss = value_loss + policy_loss  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def evaluate_model(config: SupervisedActorCriticConfig, model: DualNet, dataloader: DataLoader) -> tuple[float, float]:
    """
    モデルの評価を行い, 平均方策損失と平均価値損失を返す.
    """
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(tensor.to(config.device) for tensor in batch)
            policy_loss, value_loss, _ = loss_func(model, batch)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    model.train()
    return total_policy_loss / num_batches, total_value_loss / num_batches


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = SupervisedActorCriticConfig()

    # モデルとオプティマイザの初期化
    model = DualNet(config.board_size)
    if config.initial_model_path is not None:
        model.load_state_dict(torch.load(config.initial_model_path))
    
    if config.transfer_learning:
        model.fix_shared_weights()
        model.init_action_head_weights()
        model.init_value_head_weights()
    elif config.initial_model_path is None:
        model.unfix_shared_weights()
        model.init_weights()

    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_penalty)

    print("loading dataset...")

    train_dataset = PositionDataset(config.train_dataset_path)
    test_dataset = PositionDataset(config.test_dataset_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, persistent_workers=True)

    print(f"the number of training samples: {len(train_dataset)} positions")
    print(f"the number of test samples: {len(test_dataset)} positions")
    print(f"batch size: {config.batch_size}\n")

    print("evaluating initial model...")

    policy_loss, value_loss = evaluate_model(config, model, test_dataloader)

    print(f"Initial test loss: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}\n")

    best_test_loss = policy_loss + value_loss

    train_loss_history = []
    test_loss_history = []
    patience_counter = 0
    test_loss_history.append((policy_loss, value_loss))

    print("start training...")

    for epoch in range(config.max_epoch):
        print(f"Epoch: [{epoch + 1}/{config.max_epoch}]")

        for batch_id, batch in enumerate(train_dataloader):
            batch = tuple(tensor.to(config.device) for tensor in batch)
            policy_loss, value_loss = model_step(model, optimizer, batch)
            train_loss_history.append((policy_loss, value_loss))

            if (batch_id + 1) % 100 == 0:
                print(f"Batch [{batch_id + 1}/{len(train_dataloader)}]: "
                      f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")
                
        # エポックごとにテストデータで評価
        policy_loss, value_loss = evaluate_model(config, model, test_dataloader)
        test_loss_history.append((policy_loss, value_loss))

        print(f"Test loss: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

        current_test_loss = policy_loss + value_loss
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            patience_counter = 0

            # モデルの保存
            torch.save(model.state_dict(), config.model_out_path)
            print(f"Model saved at epoch {epoch + 1} with test loss {best_test_loss:.4f}")
        else:
            patience_counter += 1

            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

        with open(config.train_loss_history_path, 'w') as f:
            f.write(str(train_loss_history))

        with open(config.test_loss_history_path, 'w') as f:
            f.write(str(test_loss_history))


