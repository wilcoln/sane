import torch
import torch.nn.functional as F

from src.settings import settings


def regret(loss, expert_loss, reduce=True):
    # result = torch.exp(-expert_loss) * loss
    # result = (torch.exp(-expert_loss) / (eps + expert_loss)) * loss
    # result = (torch.exp(-expert_loss) * loss / (eps + expert_loss)) * loss
    # result = torch.exp(-expert_loss) * torch.norm(loss - expert_loss)
    temp = F.relu(loss - expert_loss + settings.margin_regret)
    result = temp * temp
    # result = (F.relu(loss - expert_loss) / torch.norm(loss - expert_loss)) * loss
    if reduce:
        return result.mean()
    return result


# Test regret
if __name__ == '__main__':
    loss = torch.FloatTensor([1, 1, 0, 0, 1, 1])
    expert_loss = torch.FloatTensor([0, 0, 0, 1, 1, 100])
    print(regret(loss, expert_loss, reduce=False))
    print(regret(loss, expert_loss))
