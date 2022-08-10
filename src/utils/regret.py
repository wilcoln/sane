import torch


def regret(loss, expert_loss, reduce=True):
    result = torch.exp(-expert_loss) * loss
    if reduce:
        return result.mean()
    return result


# Test regret
if __name__ == '__main__':
    loss = torch.FloatTensor([1, 1, 0, 0, 1])
    expert_loss = torch.FloatTensor([0, 0, 0, .1, .1])
    print(regret(loss, expert_loss, reduce=False))
    print(regret(loss, expert_loss))
